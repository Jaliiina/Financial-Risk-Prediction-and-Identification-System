'''
# 基于 Close 计算对数收益 ret，再算 RSI、MACD、Bollinger 上下轨，并 dropna 清理头部空值。
##  RSI Relative Strength Index 相对强弱指数:比较一段时间内价格上涨的平均幅度和下跌的平均幅度，来衡量市场的超买或超卖状态

    RSI > 70: 通常被认为市场处于超买状态，价格可能即将回调或反转下跌。
    RSI < 30: 通常被认为市场处于超卖状态，价格可能即将反弹或反转上涨。
##  macd Moving Average Convergence Divergence移动平均收敛散度：比较短期和长期的价格趋势，来判断当前的动量是在加速还是减速。
    类似为价格变化加速度

    当 macd 为正时，说明短期趋势强于长期趋势，市场处于多头（上涨）动量中。
    当 macd 为负时，说明短期趋势弱于长期趋势，市场处于空头（下跌）动量中。
    macd 值的变化反映了动量的变化。如果 macd 从负数变为正数，是一个潜在的买入信号；反之则是潜在的卖出信号。

## macd信号线 macd_sig
    平滑后macd线
    它的作用是过滤掉 macd 的短期波动，更清晰地显示其趋势。
    当 macd 线上穿 macd_sig 线时，产生金叉 (Golden Cross)，是一个看涨信号。
    当 macd 线下穿 macd_sig 线时，产生死叉 (Death Cross)，是一个看跌信号。

## bollinger 反映波动范围与价格偏离程度 识别超卖超买区域
    上轨：价格的压力位或高估区域 当价格触及或向上穿越上轨时，通常被认为是超买信号。这意味着价格上涨得过快、过猛，短期内可能会有回调或至少是横盘整理的需求。这是一个潜在的卖出或观望信号。
    下轨：价格的支撑位或低估区域 当价格触及或向下穿越下轨时，通常被认为是超卖信号。这意味着价格下跌得过快、过猛，短期内可能会有反弹或企稳的需求。这是一个潜在的买入或观望信号。

    "触及" 不一定马上反转，有时强势的趋势会让价格 "沿着轨道" 运行一段时间。因此，这通常需要结合其他指标（如 RSI）来确认。

## volume 反应交易活跃程度

# 以滑窗切片：lookback 天历史 → 预测 horizon 天后的次日收益作为标签（与 VaR 的"下一日损益"口径对齐）。
默认特征列包含 ret、rsi、macd、macd_sig、bb_high、bb_low、Volume。

输入X：过去 lookback 天的特征矩阵
标签y：第 t + horizon 天的收益

得到data/seq_lbk60_h1.npz

扩展特征工程：加入宏观、情绪、资金流特征
'''

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import os
from pathlib import Path

def check_data_files():
    """检查必要的数据文件是否存在"""
    required_files = {
        "market": "data/market.csv",
        "macro": "data/macro.csv", 
        "sentiment": "data/news_sentiment.csv",
        "flow": "data/money_flow.csv"
    }
    
    missing_files = []
    available_files = {}
    
    for data_type, file_path in required_files.items():
        if os.path.exists(file_path):
            try:
                # 尝试读取文件以验证完整性
                df = pd.read_csv(file_path, nrows=1)
                if len(df) > 0:
                    available_files[data_type] = file_path
                    print(f"✓ {data_type} data found: {file_path}")
                else:
                    missing_files.append(data_type)
                    print(f"✗ {data_type} data file is empty: {file_path}")
            except Exception as e:
                missing_files.append(data_type)
                print(f"✗ {data_type} data file corrupted: {file_path} - {e}")
        else:
            missing_files.append(data_type)
            print(f"✗ {data_type} data file missing: {file_path}")
    
    if missing_files:
        print(f"Warning: Missing {len(missing_files)} data files: {missing_files}")
        print("Please run fetch.py first to generate the required data files")
    
    return available_files

def align_data_dates(market_df, macro_df, sentiment_df, flow_df):
    """对齐所有数据集的日期范围"""
    # 使用市场数据的日期范围作为基准
    market_dates = market_df.index
    
    # 重新索引所有数据到市场数据的日期
    if macro_df is not None:
        macro_aligned = macro_df.reindex(market_dates)
        macro_aligned = macro_aligned.ffill()
    else:
        macro_aligned = None
        
    if sentiment_df is not None:
        sentiment_aligned = sentiment_df.reindex(market_dates) 
        sentiment_aligned = sentiment_aligned.ffill()
    else:
        sentiment_aligned = None
        
    if flow_df is not None:
        flow_aligned = flow_df.reindex(market_dates)
        flow_aligned = flow_aligned.ffill()
    else:
        flow_aligned = None
    
    print(f"Aligned all data to market date range: {len(market_dates)} days")
    return macro_aligned, sentiment_aligned, flow_aligned

def load_additional_data(symbol: str = "SPY") -> tuple:
    """加载额外的数据源 - 使用本地生成的文件"""
    print("Loading additional data from local files...")
    
    available_files = check_data_files()
    macro_df, sentiment_df, flow_df = None, None, None
    
    # 加载宏观数据
    if "macro" in available_files:
        try:
            macro_df = pd.read_csv(available_files["macro"], parse_dates=["Date"], index_col="Date")
            print(f"✓ Loaded macro data: {macro_df.shape}")
        except Exception as e:
            print(f"✗ Failed to load macro data: {e}")
    
    # 加载情绪数据
    if "sentiment" in available_files:
        try:
            sentiment_df = pd.read_csv(available_files["sentiment"], parse_dates=["Date"], index_col="Date")
            print(f"✓ Loaded sentiment data: {sentiment_df.shape}")
        except Exception as e:
            print(f"✗ Failed to load sentiment data: {e}")
    
    # 加载资金流数据
    if "flow" in available_files:
        try:
            flow_df = pd.read_csv(available_files["flow"], parse_dates=["Date"], index_col="Date")
            print(f"✓ Loaded flow data: {flow_df.shape}")
        except Exception as e:
            print(f"✗ Failed to load flow data: {e}")
    
    return macro_df, sentiment_df, flow_df

def add_macro_features(df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """添加宏观特征"""
    if macro_df is None:
        print("No macro data available, skipping macro features")
        return df
        
    # 合并宏观数据
    df = df.merge(macro_df, left_index=True, right_index=True, how="left")
    
    # 宏观特征工程
    print("Adding macro features...")
    
    # 利率变化特征
    if "interest_rate" in df.columns:
        df["interest_rate_change"] = df["interest_rate"].pct_change(5)  # 5日利率变化
        df["interest_rate_momentum"] = df["interest_rate"].diff(3)      # 利率动量
    
    # 收益率曲线特征
    if "treasury_10y" in df.columns and "treasury_2y" in df.columns:
        df["yield_curve"] = df["treasury_10y"] - df["treasury_2y"]      # 收益率曲线（10Y-2Y）
        df["yield_curve_change"] = df["yield_curve"].pct_change(5)      # 收益率曲线变化
    
    # 通胀特征
    if "cpi" in df.columns:
        df["inflation_gap"] = df["cpi"] / df["cpi"].rolling(252).mean() - 1  # 年化通胀缺口
        df["inflation_momentum"] = df["cpi"].pct_change(30)                  # 月度通胀动量
    
    # PMI特征
    if "pmi" in df.columns:
        df["pmi_momentum"] = df["pmi"].diff(3)                          # PMI 3个月动量
        df["pmi_trend"] = df["pmi"].rolling(10).mean()                  # PMI趋势
    
    # 美元指数特征
    if "dollar_index" in df.columns:
        df["dollar_strength"] = df["dollar_index"].pct_change(10)       # 美元指数10日变化
        df["dollar_momentum"] = df["dollar_index"].diff(5)              # 美元动量
    
    # 失业率特征
    if "unemployment" in df.columns:
        df["unemployment_change"] = df["unemployment"].diff(3)          # 失业率变化
        df["unemployment_trend"] = df["unemployment"].rolling(6).mean() # 失业率趋势
    
    # 工业生产特征
    if "industrial_production" in df.columns:
        df["production_growth"] = df["industrial_production"].pct_change(30)  # 生产增长率
    
    # 零售销售特征
    if "retail_sales" in df.columns:
        df["retail_growth"] = df["retail_sales"].pct_change(30)         # 零售增长率
    
    # 货币供应特征
    if "money_supply_m2" in df.columns:
        df["money_growth"] = df["money_supply_m2"].pct_change(30)       # 货币供应增长率
    
    # 向前填充缺失的宏观数据
    df = df.ffill()
    
    print(f"Added macro features. Total columns: {len(df.columns)}")
    return df

def add_sentiment_features(df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """添加情绪特征"""
    if sentiment_df is None:
        print("No sentiment data available, skipping sentiment features")
        return df
        
    # 合并情绪数据
    df = df.merge(sentiment_df, left_index=True, right_index=True, how="left")
    
    # 情绪特征工程
    print("Adding sentiment features...")
    
    if "sentiment_score" in df.columns:
        # 情绪动量特征
        df["sentiment_ma5"] = df["sentiment_score"].rolling(5).mean()           # 情绪5日移动平均
        df["sentiment_ma10"] = df["sentiment_score"].rolling(10).mean()         # 情绪10日移动平均
        df["sentiment_momentum"] = df["sentiment_score"].diff(3)                # 情绪3日动量
        df["sentiment_acceleration"] = df["sentiment_momentum"].diff(2)         # 情绪加速度
        
        # 情绪波动率特征
        df["sentiment_volatility_5d"] = df["sentiment_score"].rolling(5).std()   # 情绪5日波动率
        df["sentiment_volatility_10d"] = df["sentiment_score"].rolling(10).std() # 情绪10日波动率
        
        # 情绪极值标记
        if len(df) > 20:
            df["sentiment_extreme_bull"] = df["sentiment_score"] > df["sentiment_score"].quantile(0.9)
            df["sentiment_extreme_bear"] = df["sentiment_score"] < df["sentiment_score"].quantile(0.1)
            df["sentiment_high_vol"] = df["sentiment_volatility_5d"] > df["sentiment_volatility_5d"].quantile(0.8)
        else:
            df["sentiment_extreme_bull"] = False
            df["sentiment_extreme_bear"] = False
            df["sentiment_high_vol"] = False
    
    # 新闻量特征
    if "news_volume" in df.columns:
        df["news_volume_ma"] = df["news_volume"].rolling(5).mean()              # 新闻量移动平均
        df["news_volume_spike"] = df["news_volume"] > df["news_volume"].rolling(10).mean() * 1.5
    
    df = df.ffill()
    print(f"Added sentiment features. Total columns: {len(df.columns)}")
    return df

def add_flow_features(df: pd.DataFrame, flow_df: pd.DataFrame) -> pd.DataFrame:
    """添加资金流特征"""
    if flow_df is None:
        print("No flow data available, skipping flow features")
        return df
        
    # 合并资金流数据
    df = df.merge(flow_df, left_index=True, right_index=True, how="left", suffixes=("", "_flow"))
    
    # 资金流特征工程
    print("Adding flow features...")
    
    # 净流入特征
    if "net_flow" in df.columns:
        df["net_flow_ratio"] = df["net_flow"] / (df["Volume"] + 1e-8)           # 净流入比率
        df["net_flow_momentum"] = df["net_flow"].rolling(5).mean()              # 净流入5日动量
        df["net_flow_trend"] = df["net_flow"].rolling(10).mean()                # 净流入趋势
        df["net_flow_volatility"] = df["net_flow"].rolling(10).std()            # 净流入波动率
    
    # 大单流量特征
    if "large_order_flow" in df.columns:
        df["large_flow_ratio"] = df["large_order_flow"] / (df["Volume"] + 1e-8) # 大单流量比率
        df["large_flow_momentum"] = df["large_order_flow"].rolling(5).mean()    # 大单流量动量
        df["large_flow_extreme"] = df["large_order_flow"] > df["large_order_flow"].quantile(0.9)
    
    # 机构流量特征
    if "institutional_flow" in df.columns:
        df["institutional_ratio"] = df["institutional_flow"] / (df["Volume"] + 1e-8)
        df["institutional_momentum"] = df["institutional_flow"].rolling(5).mean()
    
    # 零售流量特征
    if "retail_flow" in df.columns:
        df["retail_ratio"] = df["retail_flow"] / (df["Volume"] + 1e-8)
        df["retail_momentum"] = df["retail_flow"].rolling(5).mean()
    
    # 成交量异常检测
    if "Volume" in df.columns:
        df["volume_anomaly"] = df["Volume"] / df["Volume"].rolling(20).mean()   # 成交量异常
        df["volume_momentum"] = df["Volume"].pct_change(5)                      # 成交量动量
        df["high_volume"] = df["volume_anomaly"] > 1.5                          # 高成交量标记
    
    df = df.ffill()
    print(f"Added flow features. Total columns: {len(df.columns)}")
    return df

def get_available_features(df):
    """动态获取可用的特征列"""
    # 基础技术特征
    base_tech_features = ["ret", "rsi", "macd", "macd_sig", "bb_high", "bb_low", "bb_width", "Volume"]
    
    # 宏观特征
    macro_features = [
        "interest_rate_change", "interest_rate_momentum", "yield_curve", "yield_curve_change",
        "inflation_gap", "inflation_momentum", "pmi_momentum", "pmi_trend", 
        "dollar_strength", "dollar_momentum", "unemployment_change", "unemployment_trend",
        "production_growth", "retail_growth", "money_growth"
    ]
    
    # 情绪特征
    sentiment_features = [
        "sentiment_ma5", "sentiment_ma10", "sentiment_momentum", "sentiment_acceleration",
        "sentiment_volatility_5d", "sentiment_volatility_10d", "sentiment_extreme_bull",
        "sentiment_extreme_bear", "sentiment_high_vol", "news_volume_ma", "news_volume_spike"
    ]
    
    # 资金流特征
    flow_features = [
        "net_flow_ratio", "net_flow_momentum", "net_flow_trend", "net_flow_volatility",
        "large_flow_ratio", "large_flow_momentum", "large_flow_extreme",
        "institutional_ratio", "institutional_momentum", "retail_ratio", "retail_momentum",
        "volume_anomaly", "volume_momentum", "high_volume"
    ]
    
    # 检查哪些特征列实际存在
    available_features = []
    
    for feature in base_tech_features + macro_features + sentiment_features + flow_features:
        if feature in df.columns:
            available_features.append(feature)
    
    print(f"Using {len(available_features)} available features")
    print("Available features by category:")
    print(f"  Technical: {[f for f in base_tech_features if f in df.columns]}")
    print(f"  Macro: {[f for f in macro_features if f in df.columns]}")
    print(f"  Sentiment: {[f for f in sentiment_features if f in df.columns]}")
    print(f"  Flow: {[f for f in flow_features if f in df.columns]}")
    
    return available_features

def make_features(csv_path="data/market.csv", symbol="SPY"):
    """生成完整的特征表 - 使用fetch.py生成的数据文件"""
    print("=== Starting Feature Engineering ===")
    
    # 检查数据文件
    available_files = check_data_files()
    if "market" not in available_files:
        raise FileNotFoundError("Market data file not found. Please run fetch.py first.")
    
    print("Loading market data...")
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    print(f"Loaded market data: {df.shape}")
    
    # 基本数据验证
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in market data: {missing_columns}")
    
    # 1. 基础技术指标
    df["ret"] = np.log(df["Close"]).diff()

    # 技术指标计算
    print("Calculating technical indicators...")
    rsi = RSIIndicator(df["Close"], window=14)
    macd = MACD(df["Close"])
    bb = BollingerBands(df["Close"], window=20, window_dev=2)

    df["rsi"] = rsi.rsi()
    df["macd"] = macd.macd()
    df["macd_sig"] = macd.macd_signal()
    df["macd_diff"] = df["macd"] - df["macd_sig"]  # MACD差值
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["Close"]  # 布林带宽度
    df["bb_position"] = (df["Close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"])  # 布林带位置

    # 2. 加载额外数据并对齐日期
    print("Loading and aligning additional data sources...")
    macro_df, sentiment_df, flow_df = load_additional_data(symbol)
    
    # 对齐日期范围到市场数据
    macro_df, sentiment_df, flow_df = align_data_dates(df, macro_df, sentiment_df, flow_df)
    
    # 3. 添加新特征维度
    df = add_macro_features(df, macro_df)
    df = add_sentiment_features(df, sentiment_df)
    df = add_flow_features(df, flow_df)
    # 在清理缺失值前生成多任务目标列（分位数 / VaR / CVaR / volatility / trend / risk_label）
    try:
        df = create_multi_targets(df, lookback=60, horizon=1)
    except Exception:
        # 如果 create_multi_targets 尚未定义（兼容旧版本），跳过并在后续流程中不使用多任务列
        print("create_multi_targets not available or failed; continuing without multi-target columns")

    # 清理缺失值
    initial_shape = df.shape
    df = df.dropna()
    final_shape = df.shape
    
    print(f"Feature engineering complete. Dropped {initial_shape[0] - final_shape[0]} rows due to NaN.")
    print(f"Final feature table: {df.shape}")
       
    # 预处理特征：将布尔列转换为数值类型
    df = preprocess_features(df)
    
    # 显示特征统计
    print("\n=== Feature Summary ===")
    tech_features = [col for col in df.columns if col in ["ret", "rsi", "macd", "macd_sig", "bb_high", "bb_low", "bb_width"]]
    macro_features = [col for col in df.columns if "interest" in col or "yield" in col or "inflation" in col or "pmi" in col or "dollar" in col or "unemployment" in col or "production" in col or "retail" in col or "money" in col]
    sentiment_features = [col for col in df.columns if "sentiment" in col or "news" in col]
    flow_features = [col for col in df.columns if "flow" in col or "volume_anomaly" in col]
    
    print(f"Technical features: {len(tech_features)}")
    print(f"Macro features: {len(macro_features)}") 
    print(f"Sentiment features: {len(sentiment_features)}")
    print(f"Flow features: {len(flow_features)}")
    print(f"Total features: {len(df.columns)}")
    
    return df

def preprocess_features(df):
    """预处理特征：将布尔列转换为数值类型"""
    # 找出所有的布尔列
    bool_columns = df.select_dtypes(include=['bool']).columns.tolist()
    
    if bool_columns:
        print(f"Converting boolean columns to numeric: {bool_columns}")
        # 将布尔值转换为 0 和 1
        df[bool_columns] = df[bool_columns].astype(int)
    
    return df


def create_multi_targets(df: pd.DataFrame, lookback: int = 60, horizon: int = 1,
                         quantiles: list = [0.01, 0.05, 0.5, 0.95, 0.99]):
    """为多任务学习生成目标列：
    - 基于历史滚动窗口估计的分位数（作为弱监督的分位数标签）
    - VaR95 / VaR99（分别对应 q05, q01）
    - CVaR95（过去 lookback 内小于等于 5% 分位值的均值）
    - volatility（历史滚动波动率的向前位移）
    - trend（基于 q50 是否为正）
    - risk_label（基于 VaR95 的绝对值分箱：0=low,1=mid,2=high）
    备注：使用历史滚动统计作为标签近似；可根据需要替换为未来窗口统计。
    """
    df = df.copy()

    # 1) 滚动分位数（历史窗口）并向前位移到预测位置
    rolling_ret = df['ret'].rolling(lookback)
    for q in quantiles:
        col = f"q_{int(q*100):02d}"
        try:
            df[col] = rolling_ret.quantile(q).shift(-horizon)
        except Exception:
            # 兼容性回退：使用 apply+np.quantile
            df[col] = df['ret'].rolling(lookback).apply(lambda x, qq=q: np.quantile(x, qq) if len(x) > 0 else np.nan).shift(-horizon)

    # 2) VaR / CVaR / volatility / trend / risk_label
    # VaR95 / VaR99
    if 'q_05' in df.columns:
        df['VaR95'] = df['q_05']
    else:
        df['VaR95'] = df.get('q_05', np.nan)

    if 'q_01' in df.columns:
        df['VaR99'] = df['q_01']
    else:
        df['VaR99'] = df.get('q_01', np.nan)

    # CVaR95: 在历史窗口内所有小于等于 5% 分位值的均值（向前位移）
    def _cvar95(arr):
        if len(arr) == 0:
            return np.nan
        q05 = np.quantile(arr, 0.05)
        tail = arr[arr <= q05]
        if len(tail) == 0:
            return q05
        return float(np.mean(tail))

    df['CVaR95'] = df['ret'].rolling(lookback).apply(lambda x: _cvar95(x) if len(x) > 0 else np.nan).shift(-horizon)

    # 历史波动率（这里使用历史滚动 20 日标准差并向前位移）
    df['volatility'] = df['ret'].rolling(20).std().shift(-horizon)

    # 趋势：基于中位数 q_50 的符号（不额外 shift，因为 q_50 已被 shift）
    if 'q_50' in df.columns:
        df['trend'] = (df['q_50'] > 0).astype(int)
    else:
        df['trend'] = (df['ret'].shift(-horizon) > 0).astype(int)

    # 风险标签（基于 VaR95 绝对值分箱）0=low,1=mid,2=high
    df['risk_label'] = pd.cut(df['VaR95'].abs().fillna(0), bins=[-np.inf, 0.02, 0.05, np.inf], labels=[0, 1, 2]).astype(int)

    print("Created multi-task target columns:", [c for c in df.columns if c.startswith('q_') or c in ['VaR95','VaR99','CVaR95','volatility','trend','risk_label']])
    return df

def make_sequences(df, lookback=60, horizon=1, feat_cols=None,
                   h_vol=5, trend_thresh=0.0, tau_y=0.015, tau_r=0.03):
    """生成序列数据"""
    # 如果未指定特征列，动态获取可用的特征
    if feat_cols is None:
        feat_cols = get_available_features(df)
    
    print(f"Creating sequences with {len(feat_cols)} features...")
    
    X_raw = df[feat_cols].copy()
    # 主目标：下一日收益（用于多分位回归训练）
    y_full = df["ret"].shift(-horizon)
    # 为避免结尾 NaN，截断对应行
    X_raw = X_raw.iloc[:-horizon]
    y_full = y_full.iloc[:-horizon]

    # 其他任务标签：未来波动（5日）、风险灯（3类）、趋势（3类）
    ret_vals = df["ret"].values  # 原始对数收益
    N = len(df)

    # 未来波动：基于未来 h_vol 天的标准差（不 annualize，以与 return 单位一致）
    vols = np.full(N, np.nan)
    for i in range(N):
        start = i + 1
        end = i + 1 + h_vol
        if end <= N:
            vols[i] = np.std(ret_vals[start:end], ddof=0)

    # 近期历史波动（用于风险灯判定），使用过去20日滚动 std
    recent_vol = df["ret"].rolling(20).std().fillna(method='bfill').values

    # 未来趋势标签（基于下一日收益的符号/阈值）
    trend_full = np.full(N, 1, dtype=int)  # 0=down,1=flat,2=up
    y_next = df["ret"].shift(-horizon).values
    for i in range(N):
        val = y_next[i]
        if np.isnan(val):
            trend_full[i] = 1
        else:
            if val < -trend_thresh:
                trend_full[i] = 0
            elif val > trend_thresh:
                trend_full[i] = 2
            else:
                trend_full[i] = 1

    # 风险灯：使用与推理/展示一致的绝对阈值（tau_y, tau_r）
    # 0=green,1=yellow,2=red
    risk_full = np.full(N, 0, dtype=int)
    for i in range(N):
        yv = y_next[i]
        if np.isnan(yv):
            risk_full[i] = 0
        else:
            if yv <= -tau_r:
                risk_full[i] = 2
            elif yv <= -tau_y:
                risk_full[i] = 1
            else:
                risk_full[i] = 0

    # 截断到与 X_raw / y_full 对齐的长度
    L = len(X_raw)
    vols_aligned = vols[:L]
    risk_aligned = risk_full[:L]
    trend_aligned = trend_full[:L]

    X = X_raw.values
    y = y_full.values

    xs, ys, ys_vol, ys_risk, ys_trend, idx = [], [], [], [], [], []
    for i in range(lookback, len(X)):
        xs.append(X[i-lookback:i])
        ys.append(y[i])
        ys_vol.append(vols_aligned[i])
        ys_risk.append(risk_aligned[i])
        ys_trend.append(trend_aligned[i])
        idx.append(X_raw.index[i])

    return (np.array(xs), np.array(ys), np.array(ys_vol), np.array(ys_risk), np.array(ys_trend), pd.Index(idx, name="Date"))

if __name__ == "__main__":
    # 确保数据目录存在
    Path("data").mkdir(exist_ok=True)
    
    # 检查是否需要先运行fetch.py
    available_files = check_data_files()
    if "market" not in available_files:
        print("\n❌ Market data not found. Please run fetch.py first to generate data files.")
        print("Run: python fetch.py --symbol SPY --start 2015-01-01")
        exit(1)
    
    # 生成特征
    df = make_features("data/market.csv", symbol="SPY")
    print(f"Feature table shape: {df.shape}")
    print(f"Date range: {df.index.min().date()} → {df.index.max().date()}")

    # 生成序列数据（含多任务标签）
    print("\n=== Creating Sequences (multi-task) ===")
    X, y, y_vol, y_risk, y_trend, idx = make_sequences(df, lookback=60, horizon=1)
    print(f"Sequences: {X.shape}, y: {y.shape}, y_vol: {y_vol.shape}, y_risk: {y_risk.shape}, y_trend: {y_trend.shape}")
    print(f"Sequence date range: {idx.min().date()} → {idx.max().date()}")

    # 基本健康检查
    assert len(X) == len(y) == len(y_vol) == len(y_risk) == len(y_trend) == len(idx)
    # 检查 NaN（允许 X 中存在少量 NaN，但最好在 preprocess/dropna 已处理）
    if np.isnan(X).any():
        raise ValueError("Feature matrix X contains NaN values after preprocessing")
    if np.isnan(y).any():
        raise ValueError("Target y contains NaN values")

    print("✓ Data health check passed")

    # 保存结果
    print("\n=== Saving Results ===")
    df.to_parquet("data/features.parquet")
    np.savez_compressed(
        "data/seq_lbk60_h1.npz",
        X=X,
        y=y,
        y_vol=y_vol,
        y_risk=y_risk,
        y_trend=y_trend,
        dates=idx.astype(str).values
    )
    print("Saved to:")
    print("  - data/features.parquet")
    print("  - data/seq_lbk60_h1.npz (keys: X, y, y_vol, y_risk, y_trend, dates)")
    
    # 显示数据统计
    print(f"\n=== Data Statistics ===")
    print(f"Features shape: {df.shape}")
    print(f"Sequences: {X.shape} (samples × timesteps × features)")
    print(f"Labels: {y.shape}")
    print(f"Memory usage: {X.nbytes / 1024 / 1024:.2f} MB")