import os
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# 简单的特征中文映射表，可根据项目需要扩展
FEATURE_CN_MAP = {
    'ret': '收益率',
    'rsi': 'RSI(14)',
    'macd': 'MACD',
    'macd_sig': 'MACD 信号',
    'macd_signal': 'MACD 信号',
    'bb_high': '布林带上轨',
    'bb_low': '布林带下轨',
    'Close': '收盘价',
    'Open': '开盘价',
    'High': '最高价',
    'Low': '最低价',
    'Volume': '成交量',
    'volatility_5': '波动率(5)',
    'volatility_20': '波动率(20)',
    'momentum_5': '动量(5)',
    'momentum_10': '动量(10)',
    'momentum_20': '动量(20)',
    'bb_width': '布林带宽度',
    'bb_position': '布林带位置',
    'rsi_slope': 'RSI 斜率',
    'macd_diff': 'MACD 差值',
    'macd_slope': 'MACD 斜率',
    'volume_ma': '成交量均线',
    'volume_ratio': '成交量比率',
    'high_low_ratio': '高/低 比率',
    'close_open_ratio': '收盘/开盘 比率',
}


def get_feature_cn(col_name: str):
    """返回特征的中文名；使用映射表，若未命中则用简单规则生成中文描述。"""
    if col_name in FEATURE_CN_MAP:
        return FEATURE_CN_MAP[col_name]
    # 常见模式处理
    if col_name.startswith('momentum_'):
        try:
            w = int(col_name.split('_')[-1])
            return f'动量({w})'
        except Exception:
            return '动量'
    if col_name.startswith('volatility_'):
        try:
            w = int(col_name.split('_')[-1])
            return f'波动率({w})'
        except Exception:
            return '波动率'
    if col_name.startswith('rsi'):
        return 'RSI'
    if 'macd' in col_name:
        return 'MACD'
    if 'bb' in col_name:
        return '布林带指标'
    if 'volume' in col_name.lower() or 'vol' in col_name.lower():
        return '成交量'
    # fallback: 将下划线替换为空格并首字母大写（仍为英文或拼音）
    return col_name.replace('_', ' ')


def granger_test_feature(df, feature_col, target_col='ret', maxlag=5, alpha=0.05):
    """
    对单个特征与未来收益做 Granger 因果检验。
    假定我们测试 feature_t -> return_{t+1}
    返回字典：feature, best_pvalue, best_lag, pass_windows, n_windows, pass_ratio, status
    status: '通过检验' / '无效' / '不稳定'
    """
    s = df.copy()
    if target_col not in s.columns or feature_col not in s.columns:
        return None

    # 构造目标：未来1期收益
    target = s[target_col].shift(-1).iloc[:-1]
    feature = s[feature_col].iloc[:-1]

    # 如果数据太短，直接返回无效
    if len(target) < max(3, maxlag + 2):
        return {
            'feature': feature_col,
            'best_pvalue': 1.0,
            'best_lag': None,
            'pass_windows': 0,
            'n_windows': 0,
            'pass_ratio': 0.0,
            'status': '无效',
        }

    arr = np.column_stack([target.values, feature.values])

    try:
        res = grangercausalitytests(arr, maxlag=maxlag, verbose=False)
    except Exception:
        # 若检验失败，标记为无效
        return {
            'feature': feature_col,
            'best_pvalue': 1.0,
            'best_lag': None,
            'pass_windows': 0,
            'n_windows': 0,
            'pass_ratio': 0.0,
            'status': '无效',
        }

    best_p = 1.0
    best_lag = None
    for lag, out in res.items():
        try:
            pval = out[0]['ssr_ftest'][1]
        except Exception:
            pval = 1.0
        if pval < best_p:
            best_p = pval
            best_lag = lag

    return {
        'feature': feature_col,
        'best_pvalue': float(best_p),
        'best_lag': int(best_lag) if best_lag is not None else None,
    }


def stability_check(df, feature_col, target_col='ret', maxlag=5, alpha=0.05, n_windows=3):
    """
    在多个时间窗口上运行 Granger 检验以评估稳定性。
    返回 pass_windows, n_windows, pass_ratio
    """
    s = df.copy()
    N = len(s)
    if N < 30:
        return 0, 0, 0.0

    # 等分为 n_windows 段（尽量保证每段长度不小于 maxlag+5）
    windows = []
    sizes = [N // n_windows] * n_windows
    # adjust remainder
    for i in range(N % n_windows):
        sizes[i] += 1
    idx = 0
    for size in sizes:
        windows.append(s.iloc[idx: idx + size])
        idx += size

    pass_cnt = 0
    tot = 0
    for w in windows:
        if len(w) < max(3, maxlag + 2):
            continue
        tot += 1
        try:
            r = granger_test_feature(w, feature_col, target_col=target_col, maxlag=maxlag, alpha=alpha)
            if r and r.get('best_pvalue', 1.0) < alpha:
                pass_cnt += 1
        except Exception:
            continue

    pass_ratio = pass_cnt / tot if tot > 0 else 0.0
    return pass_cnt, tot, pass_ratio


def generate_causality_report(csv_path="data/market.csv", lookback=None, maxlag=5, alpha=0.05, n_windows=3,
                              save_path="data/feature_causality.json"):
    """
    生成特征因果关系报告，返回排序后的特征检测结果列表，并保存为 JSON。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    # 计算特征（与 app.make_features 保持一致的逻辑）
    df = df.copy()
    df['ret'] = np.log(df['Close']).diff()
    # 简化：默认使用一些常见指标如果存在
    try:
        from ta.momentum import RSIIndicator
        from ta.trend import MACD
        from ta.volatility import BollingerBands
        rsi = RSIIndicator(df['Close'], window=14)
        macd = MACD(df['Close'])
        bb = BollingerBands(df['Close'], window=20, window_dev=2)
        df['rsi'] = rsi.rsi()
        df['macd'] = macd.macd()
        df['macd_sig'] = macd.macd_signal()
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
    except Exception:
        # 如果 ta 库不可用，跳过高级指标
        pass

    # 如果提供了 lookback，则只使用最近 lookback 的数据进行检验
    if lookback is not None:
        try:
            lookback = int(lookback)
            if lookback <= 0:
                lookback = None
        except Exception:
            lookback = None

    if lookback is not None and lookback < len(df):
        df_used = df.tail(lookback).copy()
    else:
        df_used = df.copy()

    # 选取用于检验的候选特征（排除ret自身）
    candidate_cols = [c for c in df_used.columns if c != 'ret']
    results = []
    for col in candidate_cols:
        try:
            # ensure maxlag is not larger than available samples
            effective_maxlag = min(maxlag, max(1, (len(df_used) // 5)))
            base = granger_test_feature(df_used, col, target_col='ret', maxlag=effective_maxlag, alpha=alpha)
            if base is None:
                continue
            pass_cnt, tot, pass_ratio = stability_check(df_used, col, target_col='ret', maxlag=effective_maxlag, alpha=alpha, n_windows=n_windows)
            if pass_ratio >= 0.66:
                status = '通过检验'
            elif pass_ratio == 0:
                status = '无效'
            else:
                status = '不稳定'

            r = {
                'feature': col,
                'feature_cn': get_feature_cn(col),
                'best_pvalue': base.get('best_pvalue', 1.0),
                'best_lag': base.get('best_lag', None),
                'pass_windows': int(pass_cnt),
                'n_windows': int(tot),
                'pass_ratio': float(pass_ratio),
                'status': status
            }
            results.append(r)
        except Exception as e:
            # 忽略单列异常
            continue

    # 排序：首先查看通过检验的，再按pass_ratio降序
    results = sorted(results, key=lambda x: (0 if x['status']=='通过检验' else 1, -x['pass_ratio']))

    # 保存报告
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({'meta': {'maxlag': maxlag, 'alpha': alpha, 'n_windows': n_windows, 'lookback': lookback}, 'results': results}, f, ensure_ascii=False, indent=2)

    # 额外保存通过检验的有效特征清单
    effective = [r['feature'] for r in results if r.get('status') == '通过检验']
    try:
        eff_path = os.path.join(os.path.dirname(save_path), 'effective_features.json')
        with open(eff_path, 'w', encoding='utf-8') as ef:
            json.dump({'effective_features': effective}, ef, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return {'meta': {'maxlag': maxlag, 'alpha': alpha, 'n_windows': n_windows}, 'results': results}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/market.csv')
    parser.add_argument('--maxlag', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--n_windows', type=int, default=3)
    parser.add_argument('--out', type=str, default='data/feature_causality.json')
    args = parser.parse_args()
    print('Generating causality report...')
    rep = generate_causality_report(args.csv, maxlag=args.maxlag, alpha=args.alpha, n_windows=args.n_windows, save_path=args.out)
    print('Saved to', args.out)
