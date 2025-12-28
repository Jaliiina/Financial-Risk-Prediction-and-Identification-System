# retrain_independent.py
#!/usr/bin/env python3
"""
完全独立的重新训练脚本 - 不依赖项目导入
使用7个特征重新训练所有模型
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# ==================== 模型定义 ====================

class LSTMQuantile(nn.Module):
    def __init__(self, in_dim=7, hidden=64, layers=2, num_quantiles=5, dropout=0.2):
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
        h, _ = self.rnn(x)
        out = self.fc(h[:, -1, :])
        return out

class EnsembleLSTM(nn.Module):
    def __init__(self, in_dim=7, hidden=64, layers=2, num_quantiles=5, dropout=0.2,
                 num_trend_classes=3, num_risk_classes=3):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        
        self.shared_proj = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.quantile_head = nn.Linear(hidden // 2, num_quantiles)
        self.vol_head = nn.Linear(hidden // 2, 1)
        self.cvar_head = nn.Linear(hidden // 2, 1)
        self.trend_head = nn.Linear(hidden // 2, num_trend_classes)
        self.risk_head = nn.Linear(hidden // 2, num_risk_classes)

    def forward(self, x):
        h, _ = self.rnn(x)
        last_hidden = h[:, -1, :]
        shared_features = self.shared_proj(last_hidden)
        
        return {
            'quantiles': self.quantile_head(shared_features),
            'vol': self.vol_head(shared_features).squeeze(-1),
            'cvar': self.cvar_head(shared_features).squeeze(-1),
            'trend_logits': self.trend_head(shared_features),
            'risk_logits': self.risk_head(shared_features)
        }

# ==================== 数据准备 ====================

def make_features_7(csv_path="data/market.csv"):
    """使用7个特征"""
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    
    # 只使用7个特征
    df["ret"] = np.log(df["Close"]).diff()
    rsi = RSIIndicator(df["Close"], window=14)
    macd = MACD(df["Close"])
    bb = BollingerBands(df["Close"], window=20, window_dev=2)
    
    df["rsi"] = rsi.rsi()
    df["macd"] = macd.macd()
    df["macd_sig"] = macd.macd_signal()
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    
    df = df.dropna()
    return df

def make_sequences_7(df, lookback=60, horizon=1):
    """使用7个特征列"""
    feat_cols = ["ret", "rsi", "macd", "macd_sig", "bb_high", "bb_low", "Volume"]
    
    print(f"使用7个特征列: {feat_cols}")
    
    X_raw = df[feat_cols].copy()
    y = df["ret"].shift(-horizon).dropna()
    X_raw = X_raw.iloc[:-horizon]
    X = X_raw.values
    y = y.values
    xs, ys, idx = [], [], []
    for i in range(lookback, len(X)):
        xs.append(X[i-lookback:i])
        ys.append(y[i])
        idx.append(X_raw.index[i])
    return np.array(xs), np.array(ys), pd.Index(idx, name="Date")

# ==================== 损失函数 ====================

def multi_quantile_loss(pred, y, quantiles=[0.01, 0.05, 0.5, 0.95, 0.99]):
    """多分位损失函数"""
    losses = []
    for i, q in enumerate(quantiles):
        error = y - pred[:, i]
        loss = torch.maximum(q * error, (q - 1) * error)
        losses.append(loss)
    
    total_loss = torch.stack(losses, dim=1).mean(dim=1).mean()
    return total_loss

# ==================== 训练函数 ====================

def train_lstm_model(X_train, y_train, X_val, y_val, device, model_name="lstm"):
    """训练LSTM模型"""
    print(f"训练{model_name}模型...")
    
    model = LSTMQuantile(in_dim=7, hidden=64, num_quantiles=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        
        # 训练
        x_batch = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_batch = torch.tensor(y_train, dtype=torch.float32).to(device)
        
        pred = model(x_batch)
        loss = multi_quantile_loss(pred, y_batch)
        
        loss.backward()
        optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            x_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
            val_pred = model(x_val_tensor)
            val_loss = multi_quantile_loss(val_pred, y_val_tensor)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"models/{model_name}_7features.pt")
            print(f"✅ {model_name} Epoch {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f} *")
        else:
            patience_counter += 1
            if epoch % 10 == 0:
                print(f"⏳ {model_name} Epoch {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        
        if patience_counter >= patience:
            print(f"⏹️ {model_name} 早停于第 {epoch} 轮")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(f"models/{model_name}_7features.pt", map_location=device))
    return model

def train_lightgbm_models(X_train, y_train, quantiles=[0.01, 0.05, 0.5, 0.95, 0.99]):
    """训练LightGBM模型"""
    print("训练LightGBM模型...")
    
    lgb_models = {}
    
    for q in quantiles:
        print(f"训练LightGBM分位数模型: {q}")
        
        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=q,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        lgb_models[q] = model
        
        # 保存单个模型
        joblib.dump(model, f"models/lgb_q{int(q*100):02d}_7features.pkl")
    
    # 保存所有模型
    joblib.dump(lgb_models, "models/ensemble_lgb_7features.pkl")
    return lgb_models

# ==================== 主训练函数 ====================

def main():
    print("🎯 开始使用7个特征重新训练所有模型")
    
    # 检查数据文件
    csv_path = "data/market.csv"
    if not os.path.exists(csv_path):
        print(f"❌ 数据文件不存在: {csv_path}")
        return
    
    # 创建模型目录
    os.makedirs('models', exist_ok=True)
    
    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 生成数据
    df = make_features_7(csv_path)
    X, y, dates = make_sequences_7(df, lookback=60)
    
    print(f"数据形状: X{X.shape}, y{y.shape}")
    
    # 数据分割
    train_idx = int(0.7 * len(X))
    val_idx = int(0.85 * len(X))
    
    X_train, X_val = X[:train_idx], X[train_idx:val_idx]
    y_train, y_val = y[:train_idx], y[train_idx:val_idx]
    
    # 创建标准化器
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(len(X_train), -1)
    X_val_2d = X_val.reshape(len(X_val), -1)
    
    scaler.fit(X_train_2d)
    X_train_scaled = scaler.transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_2d).reshape(X_val.shape)
    
    # 保存标准化器
    joblib.dump(scaler, "models/scaler_7features.pkl")
    joblib.dump(scaler, "models/ensemble_scaler_seq_7features.pkl")
    
    print(f"✅ 标准化器创建完成，特征维度: {scaler.n_features_in_}")
    
    # 训练LSTM模型
    lstm_model = train_lstm_model(X_train_scaled, y_train, X_val_scaled, y_val, device, "lstm")
    
    # 训练LightGBM模型（使用展平的特征）
    X_train_lgb = X_train_2d
    X_val_lgb = X_val_2d
    lgb_models = train_lightgbm_models(X_train_lgb, y_train)
    
    # 保存配置
    config = {
        'model_weights': {'lstm': 0.5, 'lightgbm': 0.5},
        'quantiles': [0.01, 0.05, 0.5, 0.95, 0.99],
        'lookback': 60,
        'features': 7,
        'feature_columns': ["ret", "rsi", "macd", "macd_sig", "bb_high", "bb_low", "Volume"]
    }
    joblib.dump(config, "models/ensemble_config_7features.pkl")
    
    print("🎉 重新训练完成！")
    print("📁 新的模型文件:")
    print("  - models/lstm_7features.pt")
    print("  - models/ensemble_lgb_7features.pkl") 
    print("  - models/scaler_7features.pkl")
    print("  - models/ensemble_config_7features.pkl")
    
    # 验证模型
    print("\n🧪 验证模型...")
    lstm_model.eval()
    with torch.no_grad():
        x_test = torch.tensor(X_val_scaled[:5], dtype=torch.float32).to(device)
        pred = lstm_model(x_test)
        print(f"LSTM预测形状: {pred.shape}")
    
    # 测试LightGBM
    lgb_pred = lgb_models[0.05].predict(X_val_lgb[:5])
    print(f"LightGBM预测形状: {lgb_pred.shape}")

if __name__ == "__main__":
    main()