'''
多分位 LSTM 分位回归训练脚本
支持 1%, 5%, 50%, 95%, 99% 分位预测
'''

import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径（确保能够 import src.training.*）
# Path(__file__) = .../src/training/train.py -> parents[2] 指向仓库根目录
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from joblib import dump

# 直接定义模型类，避免导入问题
class LSTMQuantile(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2, num_quantiles=5, dropout=0.2):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden, num_quantiles)

    def forward(self, x):
        h, _ = self.rnn(x)          # (B, T, H)
        out = self.fc(h[:, -1, :])  # (B, num_quantiles)
        return out

def multi_quantile_loss(pred, y, quantiles=[0.01, 0.05, 0.5, 0.95, 0.99], 
                               quantile_weights=None, sample_weights=None):
    """
    带分位权重的多分位损失函数
    pred: (B, num_quantiles) 每个分位的预测值
    y: (B,) 真实值
    quantiles: 分位列表
    quantile_weights: 每个分位的权重，用于调整不同分位的重要性
    sample_weights: 样本权重
    """
    if quantile_weights is None:
        quantile_weights = [1.0] * len(quantiles)
    
    losses = []
    for i, q in enumerate(quantiles):
        error = y - pred[:, i]
        loss = torch.maximum(q * error, (q - 1) * error)
        # 应用分位权重
        weighted_loss = loss * quantile_weights[i]
        losses.append(weighted_loss)

    # per-sample loss (B,)
    per_sample = torch.stack(losses, dim=1).mean(dim=1)

    if sample_weights is None:
        return per_sample.mean()
    else:
        w = sample_weights.float()
        if w.sum() == 0:
            return per_sample.mean()
        return (per_sample * w).sum() / w.sum()
    
def compute_coverage(pred, y, quantiles):
    """
    计算各分位的覆盖率
    pred: tensor of shape (B, num_quantiles)
    y: tensor of shape (B,)
    quantiles: list of quantile values
    """
    coverage = {}
    for i, q in enumerate(quantiles):
        if q < 0.5:
            cov = (y <= pred[:, i]).float().mean().item()
        else:
            cov = (y >= pred[:, i]).float().mean().item()
        coverage[f'q{int(q*100):02d}'] = cov
    return coverage

class SeqDS(Dataset):
    def __init__(self, X, y, weights=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        if weights is None:
            self.w = torch.ones(len(self.y), dtype=torch.float32)
        else:
            self.w = torch.tensor(weights, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.w[i]

def time_split(n, tr=0.7, va=0.15):
    t = int(n * tr); v = int(n * (tr + va))
    return slice(0, t), slice(t, v), slice(v, n)


def main(args):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 定义分位点
    quantiles = [0.01, 0.05, 0.5, 0.95, 0.99]
    num_quantiles = len(quantiles)

    print(f"🚀 开始多分位训练 - 分位点: {quantiles}")
    print(f"📊 设备: {device}")

    # 1) 读取数据
    if not os.path.exists(args.npz):
        print(f"❌ 数据文件不存在: {args.npz}")
        print("请先运行数据预处理脚本生成 seq_lbk60_h1.npz")
        return
    
    pack = np.load(args.npz, allow_pickle=True)
    X = pack["X"]            # (N, T, F)
    y = pack["y"]            # (N,)
    dates = pack["dates"]    # 日期索引

    print(f"📈 数据形状: X{X.shape}, y{y.shape}")

    # 2) 按时间切分（严格时序）
    tr, va, te = time_split(len(X), tr=args.train_ratio, va=args.val_ratio)

    # 3) 创建模型目录
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    # 4) 标准化（只在train拟合）
    # 在拟合 StandardScaler 之前清理 Inf / NaN / 非有限值
    def _sanitize_2d(arr2d):
        """把 2D 数组中的 inf/nan 替换为每列中位数（若整列无效则填 0）。"""
        A = arr2d.astype(float).copy()
        # 标记非有限位置
        mask_finite = np.isfinite(A)
        if mask_finite.all():
            # 没有异常值
            col_med = np.nanmedian(A, axis=0)
            col_med = np.where(np.isnan(col_med), 0.0, col_med)
            return A, col_med

        # 将非有限置为 NaN，方便 nanmedian 计算
        A[~mask_finite] = np.nan
        col_med = np.nanmedian(A, axis=0)
        # 若某列全是 NaN，nanmedian 返回 nan，替换为 0
        col_med = np.where(np.isnan(col_med), 0.0, col_med)

        # 填充 NaN 为对应列中位数
        inds = np.where(np.isnan(A))
        if len(inds[0]) > 0:
            A[inds] = np.take(col_med, inds[1])

        return A, col_med

    Xtr_2d = X[tr].reshape(len(X[tr]), -1).astype(np.float64)
    Xtr_clean, col_meds = _sanitize_2d(Xtr_2d)

    scaler = StandardScaler().fit(Xtr_clean)
    dump(scaler, "models/scaler.pkl") 

    def scale3d(A):
        # A: (N, T, F)
        A2 = A.reshape(len(A), -1).astype(np.float64).copy()
        # 替换非有限值为训练集的列中位数
        mask_finite = np.isfinite(A2)
        if not mask_finite.all():
            inds = np.where(~mask_finite)
            if len(inds[0]) > 0:
                A2[inds] = np.take(col_meds, inds[1])
        B = scaler.transform(A2)
        return B.reshape(A.shape)
    
    Xtr, Xva, Xte = scale3d(X[tr]), scale3d(X[va]), scale3d(X[te])
    ytr, yva, yte = y[tr], y[va], y[te]

    # === 异常检测与样本权重 ===
    # 支持方法: 'zscore' or 'rolling_vol'
    def detect_anomalies_series(ret_series, method='zscore', z_thresh=3.0, win=60):
        """
        更稳健的异常检测：
        - 当 rolling window 大于可用样本，退化为全量均值/Std
        - 避免 std==0 导致 z 恒为0，使用 eps 下限
        - 返回 (mask, score_array)
        """
        s = pd.Series(ret_series)
        n = len(s)
        # sanitize window
        try:
            win = int(win) if win is not None else None
        except Exception:
            win = None

        # choose effective rolling window
        if win is None or win < 2 or win > n:
            use_roll = False
        else:
            use_roll = True

        eps = 1e-8
        if method == 'zscore':
            if use_roll:
                roll_mean = s.rolling(window=win, min_periods=1).mean()
                roll_std = s.rolling(window=win, min_periods=1).std().fillna(0.0)
                mu = roll_mean.values
                sd = roll_std.values
            else:
                mu = np.full(n, s.mean(), dtype=float)
                sd = np.full(n, s.std(ddof=0) if n > 0 else 0.0, dtype=float)

            sd = np.where(sd < eps, eps, sd)
            z = (s.values - mu) / sd
            scores = np.abs(z)
            mask = scores > z_thresh
            return mask, scores
        else:
            # rolling volatility: use rolling std as volatility proxy
            if use_roll:
                roll_vol = s.rolling(window=win, min_periods=1).std().fillna(0.0).values
            else:
                roll_vol = np.full(n, s.std(ddof=0) if n > 0 else 0.0, dtype=float)
            roll_vol = np.where(roll_vol < eps, eps, roll_vol)
            mu = s.rolling(window=win, min_periods=1).mean().values if use_roll else np.full(n, s.mean(), dtype=float)
            z_eq = (s.values - mu) / roll_vol
            scores = np.abs(z_eq)
            mask = scores > z_thresh
            return mask, scores

    # 假设 y 表示未来回报或目标，可以在原始 ytr 上计算异常
    anomaly_mask = np.zeros(len(ytr), dtype=bool)
    anomaly_scores = np.zeros(len(ytr), dtype=float)
    try:
        mask, scores = detect_anomalies_series(ytr, method=args.anomaly_method, z_thresh=args.z_threshold, win=args.rolling_window)
        anomaly_mask = np.asarray(mask)
        anomaly_scores = np.asarray(scores)

        # 调试输出：展示前若干值，方便用户确认 rolling_window 被应用
        print("[调试] ytr[:10] =", np.round(ytr[:10], 6))
        print("[调试] anomaly_scores[:10] =", np.round(anomaly_scores[:10], 6))
        print("[调试] anomaly_mask sum =", int(anomaly_mask.sum()))
        print("[调试] anomaly indices (first 20):", np.where(anomaly_mask)[0][:20])
    except Exception as e:
        print("⚠️ 异常检测失败，跳过异常权重：", e)

    # 将异常样本权重缩小（默认 scale=0.1）
    base_weights = np.ones(len(ytr), dtype=float)
    if anomaly_mask.sum() > 0:
        # 使用乘法缩放，保底 0.01
        scaled = np.clip(base_weights[anomaly_mask] * args.weight_scale, 0.01, 1.0)
        base_weights[anomaly_mask] = scaled
        print("[调试] unique weights after scaling:", np.unique(base_weights)[:10])

    # 统计并保存 anomaly stats
    stats = {
        'n_train': int(len(ytr)),
        'n_anomalies': int(anomaly_mask.sum()),
        'anomaly_ratio': float(anomaly_mask.sum() / max(1, len(ytr)))
    }
    import json
    Path('data').mkdir(exist_ok=True)
    with open('data/anomaly_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"🔎 异常样本检测: {stats['n_anomalies']} / {stats['n_train']} ({stats['anomaly_ratio']:.3%}) — 权重缩放: {args.weight_scale}")
    
    # 处理日期索引
    if hasattr(dates, 'tolist'):
        dates = dates.tolist()
    dat_te = dates[te.stop:] if te.stop < len(dates) else dates[te]

    print(f"📊 训练集: {len(Xtr)}, 验证集: {len(Xva)}, 测试集: {len(Xte)}")

    # 5) DataLoader
    train_loader = DataLoader(SeqDS(Xtr, ytr, weights=base_weights), batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader   = DataLoader(SeqDS(Xva, yva), batch_size=args.batch*2, shuffle=False)
    test_loader  = DataLoader(SeqDS(Xte, yte), batch_size=args.batch*2, shuffle=False)

    # 6) 模型与优化器
    model = LSTMQuantile(
        in_dim=X.shape[-1], 
        hidden=args.hidden, 
        layers=args.layers,
        num_quantiles=num_quantiles, 
        dropout=args.dropout
    ).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"🧠 模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 7) 训练（早停）
    best = 1e9; wait = 0
    train_losses, val_losses = [], []
    
# 根据观察到的覆盖率偏差设置分位权重
    # 覆盖率偏低的给更高权重，覆盖率接近的给正常权重
    quantile_weights = [2.0, 1.5, 1.0, 1.5, 2.0]  # 对应 [q01, q05, q50, q95, q99]
    
    for epoch in range(args.epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if len(batch) == 3:
                xb, yb, wb = batch
            else:
                xb, yb = batch; wb = torch.ones(len(yb))
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            pred = model(xb)
            # 使用带权重的损失函数
            loss = multi_quantile_loss(pred, yb, quantiles=quantiles, 
                                              quantile_weights=quantile_weights, 
                                              sample_weights=wb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        # 验证也使用相同的权重
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    xb, yb, wb = batch
                else:
                    xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = multi_quantile_loss(pred, yb, quantiles=quantiles,
                                                  quantile_weights=quantile_weights)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 早停
        if val_loss < best:
            best = val_loss
            wait = 0
            torch.save(model.state_dict(), "models/best_lstm_quantile.pt")
            print(f"✅ Epoch {epoch+1:3d} | 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} *")
        else:
            wait += 1
            print(f"⏳ Epoch {epoch+1:3d} | 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")

        if wait >= args.patience:
            print(f"⏹️  早停于第 {epoch+1} 轮")
            break

    # 8) 测试
    print("\n🧪 测试阶段")
    model.load_state_dict(torch.load("models/best_lstm_quantile.pt", map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                xb, yb, wb = batch
            else:
                xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            y_true.append(yb.cpu())
            y_pred.append(pred.cpu())

    # 这里 y_true 和 y_pred 已经是 CPU tensor 列表，直接拼接
    y_true = torch.cat(y_true)  # 已经是 tensor，不需要再转换
    y_pred = torch.cat(y_pred)  # 已经是 tensor，不需要再转换

    # 9) 计算覆盖率 - 直接传入张量，不需要再次包装
    coverage = compute_coverage(y_pred, y_true, quantiles)
    
    print("\n📊 测试集覆盖率:")
    for q_name, cov in coverage.items():
        expected = float(q_name[1:]) / 100.0
        if expected > 0.5:
            expected = 1 - expected
        print(f"  {q_name}: {cov:.3f} (期望: {expected:.3f})")

    # 10) 保存测试集预测结果
    test_df = pd.DataFrame({
        'date': dat_te[:len(y_true)],
        'y_true': y_true,
        'q01': y_pred[:, 0],
        'q05': y_pred[:, 1], 
        'q50': y_pred[:, 2],
        'q95': y_pred[:, 3],
        'q99': y_pred[:, 4]
    })
    
    test_df.to_csv("data/test_preds.csv", index=False)
    print(f"💾 测试集预测已保存至 data/test_preds.csv")

    # --- 调用 LightGBM 训练脚本以训练表格模型并保存评估 ---
    try:
        print("\n🔗 开始训练 LightGBM 模型（表格特征） via src.training.train_lightgbm")
        from src.training import train_lightgbm as tl
        tl_args = argparse.Namespace(features='data/features.parquet', outdir='models', quantiles=quantiles)
        tl.main(tl_args)
        print("✅ LightGBM 训练完成，评估结果保存在 models/lgb_eval_summary.json")
    except Exception as e:
        print(f"⚠️ 无法调用 train_lightgbm: {e}")

    # 保存 ensemble 配置（包括当前 LSTM 模型路径与 LightGBM 汇总路径）
    ensemble_cfg = {
        'lstm_model': os.path.abspath('models/best_lstm_quantile.pt'),
        'lgb_eval_summary': os.path.abspath('models/lgb_eval_summary.json') if os.path.exists('models/lgb_eval_summary.json') else None,
        'scaler': os.path.abspath('models/scaler.pkl') if os.path.exists('models/scaler.pkl') else None,
        'lgb_scaler': os.path.abspath('models/lgb_scaler.pkl') if os.path.exists('models/lgb_scaler.pkl') else None
    }
    with open('models/ensemble_config.json', 'w', encoding='utf-8') as f:
        json.dump(ensemble_cfg, f, indent=2, ensure_ascii=False)
    print("🔧 ensemble_config 已保存至 models/ensemble_config.json")

    # 11) 绘制损失曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('多分位损失')
        plt.title('多分位LSTM训练过程')
        plt.legend()
        plt.grid(True)
        plt.savefig('models/training_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📈 训练损失曲线已保存至 models/training_loss.png")
    except ImportError:
        print("⚠️  无法导入matplotlib，跳过绘图")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="data/seq_lbk60_h1.npz")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    # 新增异常检测相关参数
    parser.add_argument("--anomaly-method", type=str, default="zscore", help="异常检测方法: zscore 或 rolling_vol")
    parser.add_argument("--z-threshold", type=float, default=3.0, help="z-score 判定阈值")
    parser.add_argument("--rolling-window", type=int, default=60, help="滚动窗口大小")
    parser.add_argument("--weight-scale", type=float, default=0.1, help="异常样本权重缩放")
    args = parser.parse_args()

    main(args)