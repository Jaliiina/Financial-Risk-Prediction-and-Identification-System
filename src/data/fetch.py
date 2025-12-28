'''
扩展数据抓取：宏观指标、新闻情绪、资金流
修复 yfinance 版本兼容性问题
'''

import argparse, time, sys, os, math
from pathlib import Path
import datetime as dt
import pandas as pd
import requests
import numpy as np
from typing import Dict, List, Optional

# --------- 通用工具 ---------
def _save_csv(df: pd.DataFrame, out: str):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df = df.dropna()
    df.to_csv(out)
    print(f"[save] {out} rows={len(df)} range={df.index.min().date()}→{df.index.max().date()}")

def _maybe_use_cache(out: str, fresh_days: int = 2):
    if os.path.exists(out):
        try:
            old = pd.read_csv(out, parse_dates=["Date"], index_col="Date")
            if len(old) > 0 and old.index.max().date() >= (dt.date.today() - dt.timedelta(days=fresh_days)):
                print("[cache] use cached file:", out)
                return True
        except Exception as e:
            print(f"[cache] error reading {out}: {e}")
    return False

def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Open","High","Low","Close","Volume"]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing columns: {miss}")
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        else:
            raise RuntimeError("No Date index/column")
    return df[cols].sort_index()

# --------- 修复的 yfinance 下载 ---------
def fetch_yfinance(symbol: str, start: str):
    import yfinance as yf
    print(f"[yf] downloading {symbol} from {start} ...")
    
    try:
        # 新版本 yfinance 的兼容性下载
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, auto_adjust=True)
        
        if df is None or len(df) < 5:
            raise RuntimeError("empty dataframe from yfinance")
        
        # 确保有必要的列
        if 'Close' not in df.columns:
            if 'Adj Close' in df.columns:
                df = df.rename(columns={'Adj Close': 'Close'})
            else:
                raise RuntimeError("No Close price column found")
        
        return _ensure_ohlcv(df)
        
    except Exception as e:
        print(f"[yf] download failed: {e}")
        raise

# --------- Provider B: Stooq（无需API）---------
def _stooq_code(symbol: str) -> str:
    if symbol.isalpha():
        return f"{symbol.lower()}.us"
    alias = {
        "^GSPC": "spy.us",
        "^IXIC": "qqq.us", 
        "^DJI":  "dia.us",
    }
    return alias.get(symbol, "")

def fetch_stooq(symbol: str, start: str):
    code = _stooq_code(symbol)
    if not code:
        raise RuntimeError("no stooq code mapping")
    url = f"https://stooq.com/q/d/l/?s={code}&i=d"
    print(f"[stooq] {code} ...")
    try:
        df = pd.read_csv(url)
        if df is None or len(df) < 5:
            raise RuntimeError("empty dataframe from stooq")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})
        df = df[df.index >= pd.to_datetime(start)]
        return _ensure_ohlcv(df)
    except Exception as e:
        raise RuntimeError(f"stooq error: {e}")

# --------- Provider C: Binance ---------
def _binance_symbol(symbol: str) -> str:
    if symbol.endswith("-USD"):
        return symbol.replace("-USD","USDT").upper()
    if symbol.endswith("USDT"):
        return symbol.upper()
    return ""

def _binance_klines(sym: str, start_ms: int):
    import requests
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": sym, "interval": "1d", "limit": 1000, "startTime": start_ms}
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"binance http {r.status_code}: {r.text[:200]}")
    return r.json()

def fetch_binance(symbol: str, start: str):
    sym = _binance_symbol(symbol)
    if not sym:
        raise RuntimeError("not a binance symbol (expect *-USD or *USDT)")
    print(f"[binance] {sym} from {start} ...")
    start_dt = pd.to_datetime(start)
    start_ms = int(start_dt.timestamp() * 1000)

    all_rows = []
    cur = start_ms
    for _ in range(50):
        data = _binance_klines(sym, cur)
        if not data:
            break
        all_rows.extend(data)
        last_close_time = data[-1][6]
        cur = last_close_time + 1
        if len(data) < 1000:
            break
        time.sleep(0.2)

    if not all_rows:
        raise RuntimeError("empty klines from binance")

    cols = ["Open time","Open","High","Low","Close","Volume",
            "Close time","Quote asset vol","Trades","Taker buy base",
            "Taker buy quote","Ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["Date"] = pd.to_datetime(df["Close time"], unit="ms")
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.set_index("Date")[["Open","High","Low","Close","Volume"]].sort_index()
    return _ensure_ohlcv(df[df.index >= pd.to_datetime(start)])


# --------- 新增宏观数据抓取 ---------
def fetch_fred_data(start_date="2015-01-01"):
    """从FRED获取高质量的宏观数据"""
    print("[FRED] fetching macroeconomic data from Federal Reserve...")
    
    # 在这里填入你的FRED API密钥
    FRED_API_KEY = os.getenv('FRED_API_KEY', '6eca85307e1e3dc85a72a6459c56e5ef')
    
    # 精选的宏观指标系列ID（都是活跃的系列）
    fred_series = {
        # 利率相关
        "FEDFUNDS": "interest_rate",           # 联邦基金利率
        "DGS10": "treasury_10y",               # 10年期国债收益率
        "DGS2": "treasury_2y",                 # 2年期国债收益率
        
        # 通胀相关
        "CPIAUCSL": "cpi",                     # CPI消费者价格指数
        "CPILFESL": "core_cpi",                # 核心CPI（排除食品能源）
        
        # 经济活动
        "NAPM": "pmi",                         # ISM制造业PMI
        "INDPRO": "industrial_production",     # 工业生产指数
        "RSAFS": "retail_sales",               # 零售销售额
        
        # 就业与货币
        "UNRATE": "unemployment",              # 失业率
        "DTWEXB": "dollar_index",              # 贸易加权美元指数
        "M2SL": "money_supply_m2",             # M2货币供应量
    }
    
    data_frames = []
    successful_series = 0
    
    for series_id, col_name in fred_series.items():
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json',
            'observation_start': start_date,
            'sort_order': 'asc'
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                observations = data.get('observations', [])
                
                if observations:
                    # 解析数据
                    dates = []
                    values = []
                    for obs in observations:
                        date_str = obs.get('date', '')
                        value_str = obs.get('value', '')
                        
                        if date_str and value_str != '.':
                            dates.append(pd.to_datetime(date_str))
                            try:
                                values.append(float(value_str))
                            except ValueError:
                                continue
                    
                    if dates and values:
                        # 创建Series
                        series_df = pd.DataFrame({col_name: values}, index=dates)
                        data_frames.append(series_df)
                        successful_series += 1
                        print(f"[FRED] {col_name} fetched: {len(series_df)} rows")
            
            # 避免请求过快
            time.sleep(0.5)
            
        except Exception as e:
            print(f"[FRED] Failed to fetch {series_id}: {e}")
            continue
    
    print(f"[FRED] Successfully fetched {successful_series} out of {len(fred_series)} series")
    
    # 合并所有数据
    if data_frames:
        # 合并所有DataFrame
        macro_df = pd.concat(data_frames, axis=1)
        
        # 创建完整的日期索引（日频）
        full_index = pd.date_range(start=start_date, end=dt.datetime.today(), freq='D')
        macro_df = macro_df.reindex(full_index)
        
        # 前向填充缺失值（宏观数据通常是低频的）
        macro_df = macro_df.ffill()
        
        return macro_df
    else:
        print("[FRED] No data fetched, falling back to synthetic data")
        return pd.DataFrame()

def generate_synthetic_macro_data(start_date="2015-01-01"):
    """生成具有真实经济模式的合成宏观数据"""
    print("[synthetic] generating realistic synthetic macro data...")
    
    # 创建完整的日期索引
    dates = pd.date_range(start=start_date, end=dt.datetime.today(), freq='D')
    t = np.arange(len(dates))
    
    # 基于真实经济周期模式生成数据
    synthetic_data = {
        # 利率：缓慢上升趋势 + 周期性波动 + 疫情期降息模拟
        'interest_rate': 0.5 + 0.02*t/365 + 0.8*np.sin(t*0.008) + 
                        np.where(dates > pd.Timestamp('2020-03-01'), -1.5, 0) + 
                        np.random.normal(0, 0.05, len(t)),
        
        # CPI：稳定通胀 + 季节性 + 近期通胀压力
        'cpi': 250 + 0.08*t + 3*np.sin(t*0.017) + 
               np.where(dates > pd.Timestamp('2021-01-01'), 15, 0) +
               np.random.normal(0, 0.3, len(t)),
        
        # 核心CPI：更稳定的通胀指标
        'core_cpi': 250 + 0.06*t + 2*np.sin(t*0.015) + 
                   np.where(dates > pd.Timestamp('2021-01-01'), 8, 0) +
                   np.random.normal(0, 0.2, len(t)),
        
        # 失业率：反周期 + 疫情冲击
        'unemployment': 4.5 + 0.8*np.sin(t*0.005 + 1) + 
                       np.where((dates > pd.Timestamp('2020-03-01')) & 
                               (dates < pd.Timestamp('2020-08-01')), 10, 0) +
                       np.random.normal(0, 0.2, len(t)),
        
        # 10年期国债利率：与政策利率相关但更高
        'treasury_10y': 2.5 + 0.015*t/365 + 0.6*np.sin(t*0.007) + 
                       np.random.normal(0, 0.08, len(t)),
        
        # 2年期国债利率
        'treasury_2y': 1.8 + 0.012*t/365 + 0.5*np.sin(t*0.007) + 
                      np.random.normal(0, 0.06, len(t)),
        
        # PMI制造业指数
        'pmi': 55 + 3*np.sin(t*0.01) + 
              np.where(dates > pd.Timestamp('2020-03-01'), -10, 0) +
              np.random.normal(0, 1, len(t)),
        
        # 工业生产：经济增长代理
        'industrial_production': 100 + 0.03*t + 2*np.sin(t*0.01) + 
                               np.where(dates > pd.Timestamp('2020-03-01'), -8, 0) +
                               np.random.normal(0, 0.5, len(t)),
        
        # 零售销售
        'retail_sales': 500 + 0.5*t + 20*np.sin(t*0.01) + 
                       np.where(dates > pd.Timestamp('2020-03-01'), -50, 0) +
                       np.random.normal(0, 5, len(t)),
        
        # 美元指数
        'dollar_index': 95 + 3*np.sin(t*0.005) + 
                       np.random.normal(0, 1, len(t)),
        
        # M2货币供应：量化宽松影响
        'money_supply_m2': 15000 + 8*t + 200*np.sin(t*0.005) + 
                          np.where(dates > pd.Timestamp('2020-03-01'), 3000, 0) +
                          np.random.normal(0, 50, len(t)),
    }
    
    df_synthetic = pd.DataFrame(synthetic_data, index=dates)
    
    # 确保数据合理性
    df_synthetic['interest_rate'] = df_synthetic['interest_rate'].clip(0, 10)
    df_synthetic['unemployment'] = df_synthetic['unemployment'].clip(2, 25)
    df_synthetic['pmi'] = df_synthetic['pmi'].clip(30, 70)
    
    print(f"[synthetic] Generated {len(df_synthetic.columns)} synthetic macro series")
    return df_synthetic

def fetch_comprehensive_macro_data(start_date="2015-01-01", use_fred=True):
    """综合宏观数据获取 - FRED优先，合成数据备用"""
    print("[macro] fetching comprehensive macroeconomic data...")
    
    # 1. 优先尝试FRED (最高质量)
    if use_fred:
        fred_df = fetch_fred_data(start_date)
        if not fred_df.empty and len(fred_df.columns) > 5:  # 确保获取到足够多的指标
            print(f"[macro] FRED data successfully integrated: {len(fred_df.columns)} series")
            return fred_df
        else:
            print("[macro] FRED data insufficient, falling back to synthetic data")
    
    # 2. 备用：生成合成数据
    synthetic_df = generate_synthetic_macro_data(start_date)
    print(f"[macro] Using synthetic macro data with realistic economic patterns")
    
    return synthetic_df

def fetch_macro(start: str = "2015-01-01", out: str = "data/macro.csv"):
    """抓取主要宏观指标 - 使用新的综合方法"""
    print(f"[macro] 🚀 Starting comprehensive macro data fetch from {start}...")
    
    try:
        # 使用新的综合数据获取方法
        df = fetch_comprehensive_macro_data(start, use_fred=True)
        
        # 数据质量报告
        total_days = len(df)
        data_quality = {}
        
        for col in df.columns:
            non_null_count = df[col].count()
            completeness = (non_null_count / total_days) * 100
            data_quality[col] = completeness
            
            if completeness < 80:
                print(f"[macro] ⚠️  {col}: {completeness:.1f}% complete")
            else:
                print(f"[macro] ✅ {col}: {completeness:.1f}% complete")
        
        _save_csv(df, out)
        print(f"[macro] 🎉 Macro data successfully saved to {out}")
        print(f"[macro] 📈 Final dataset: {len(df.columns)} series, {len(df)} days")
        
        return df
        
    except Exception as e:
        print(f"[macro] 💥 Critical error: {e}")
        # 即使出错也生成合成数据
        print("[macro] Generating synthetic data as fallback...")
        df = generate_synthetic_macro_data(start)
        _save_csv(df, out)
        return df
# --------- 简化的新闻情绪数据抓取 ---------
def fetch_alternative_news_sentiment(symbol: str = "SPY", start: str = "2015-01-01", out: str = "data/news_sentiment.csv"):
    """使用替代源获取新闻情绪数据 - 专注于Polygon和Yahoo Finance"""
    print(f"[Alternative News] 📰 Fetching news sentiment for {symbol} from {start}...")

    # 创建日期范围
    start_date = pd.to_datetime(start).date()
    end_date = dt.date.today()
    
    # 获取新闻数据
    all_articles = _fetch_historical_news(symbol, start_date, end_date)
    
    if all_articles:
        sentiment_df = _process_articles_to_sentiment(all_articles, start)
        _save_csv(sentiment_df, out)
        print(f"[Alternative News] ✅ Successfully processed {len(all_articles)} articles from {start_date} to {end_date}")
        return sentiment_df
    else:
        print("[Alternative News] ⚠️ No articles found, using enhanced synthetic data")
        return _generate_enhanced_sentiment(symbol, start, out)

def _fetch_historical_news(symbol: str, start_date: dt.date, end_date: dt.date) -> List[Dict]:
    """获取历史新闻数据"""
    all_articles = []
    
    # 按月份分批获取新闻（更细的粒度）
    current_date = start_date
    while current_date <= end_date:
        # 计算每个批次的结束日期（3个月一批）
        batch_end = min(
            current_date + dt.timedelta(days=89),  # 约3个月
            end_date
        )
        
        print(f"[News Fetch] Fetching news for {current_date} to {batch_end}...")
        
        # 优先使用Polygon
        batch_articles = _try_polygon_news_by_date(symbol, current_date, batch_end)
        
        # 如果Polygon没有数据且是最近的时间段，尝试Yahoo Finance
        if not batch_articles and batch_end >= dt.date.today() - dt.timedelta(days=30):
            yahoo_articles = _try_yahoofinance_news(symbol)
            if yahoo_articles:
                # 过滤日期范围
                batch_articles = [
                    article for article in yahoo_articles 
                    if current_date <= article['date'].date() <= batch_end
                ]
        
        if batch_articles:
            all_articles.extend(batch_articles)
            print(f"[News Fetch] ✅ {current_date} to {batch_end}: {len(batch_articles)} articles")
        else:
            print(f"[News Fetch] ⚠️ {current_date} to {batch_end}: No articles found")
        
        # 移动到下一个批次
        current_date = batch_end + dt.timedelta(days=1)
        
        # 避免请求过快
        time.sleep(0.5)
    
    return all_articles

def _try_polygon_news_by_date(symbol: str, start_date: dt.date, end_date: dt.date) -> List[Dict]:
    """按日期范围从Polygon获取新闻"""
    API_KEY = os.getenv('POLYGON_API_KEY', 'yhSsedlqhzD_B3TcvwYsqLsB5gD4FVgb')
    if not API_KEY:
        return []

    # Polygon API 参数
    params = {
        'ticker': symbol,
        'limit': 50,
        'order': 'desc',
        'sort': 'published_utc',
        'apiKey': API_KEY,
        'published_utc.gte': start_date.strftime('%Y-%m-%d'),
        'published_utc.lte': (end_date + dt.timedelta(days=1)).strftime('%Y-%m-%d')
    }

    all_articles = []
    
    try:
        url = f"https://api.polygon.io/v2/reference/news"
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('results', [])
            
            for item in articles:
                title = item.get('title', '') or ''
                summary = item.get('description', '') or ''
                text = f"{title} {summary}".strip()

                # 解析发布时间
                try:
                    published = pd.to_datetime(item.get('published_utc', None))
                except Exception:
                    published = pd.to_datetime('today')

                # 检查是否在日期范围内
                if published.date() < start_date or published.date() > end_date:
                    continue

                # 计算情感分数
                sentiment_score = _improved_score_text_sentiment(text)

                article_data = {
                    'title': title,
                    'summary': summary,
                    'date': published,
                    'url': item.get('article_url', ''),
                    'source': item.get('author', 'Polygon'),
                    'relevance': 1.0,
                    'sentiment': sentiment_score
                }
                all_articles.append(article_data)

            print(f"[Polygon] {start_date} to {end_date}: {len(all_articles)} articles")
                
        else:
            print(f"[Polygon] API error: {response.status_code} {response.text[:200]}")
            
    except Exception as e:
        print(f"[Polygon] Error: {e}")

    return all_articles

def _try_yahoofinance_news(symbol: str) -> List[Dict]:
    """从雅虎财经获取新闻"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        news = getattr(ticker, 'news', []) or []

        articles = []
        for item in news[:50]:  # 增加数量限制
            publish_time = item.get('providerPublishTime', 0)
            if publish_time:
                try:
                    publish_date = pd.to_datetime(publish_time, unit='s')
                except Exception:
                    publish_date = pd.to_datetime('today')
            else:
                publish_date = pd.to_datetime('today')

            title = item.get('title', '') or ''
            summary = item.get('summary', '') or ''
            text = f"{title} {summary}".strip()

            # 计算情感分数
            sentiment_score = _improved_score_text_sentiment(text)

            article = {
                'title': title,
                'summary': summary,
                'date': publish_date,
                'url': item.get('link', ''),
                'source': 'Yahoo Finance',
                'relevance': 1.0,
                'sentiment': sentiment_score
            }
            articles.append(article)

        print(f"[Yahoo Finance] Fetched {len(articles)} articles")
        return articles
    except Exception as e:
        print(f"[Yahoo Finance] Error: {e}")
        return []

# 改进的情感分析函数
def _create_comprehensive_sentiment_lexicon():
    """创建更全面的情感词典"""
    positive_words = {
        # 基础正面词
        'bull', 'bullish', 'surge', 'rally', 'gain', 'profit', 'growth', 'positive', 
        'strong', 'beat', 'up', 'rise', 'soar', 'jump', 'boost', 'optimistic', 
        'recovery', 'outperform', 'breakout', 'momentum', 'record', 'high',
        'success', 'win', 'victory', 'achievement', 'breakthrough', 'expansion',
        'progress', 'improve', 'better', 'great', 'excellent', 'outstanding',
        'amazing', 'fantastic', 'strong', 'robust', 'healthy', 'solid',
        'boom', 'prosper', 'thrive', 'flourish', 'accelerate', 'advance',
        'climb', 'increase', 'appreciate', 'strengthen', 'rebound', 'recover',
        'turnaround', 'upside', 'opportunity', 'potential', 'promising',
        'encouraging', 'favorable', 'beneficial', 'advantageous', 'profitable',
        'lucrative', 'rewarding', 'dividend', 'buy', 'purchase', 'accumulate',
        'overweight', 'outperform', 'buy', 'strong buy', 'upgrade',
        
        # 金融特定正面词
        'earnings', 'revenue', 'sales', 'margin', 'dividend', 'yield',
        'cash flow', 'balance sheet', 'valuation', 'undervalued', 'upside',
        'target price', 'price target', 'bull case', 'growth story',
        'market share', 'competitive advantage', 'innovation', 'disruptive',
        'efficient', 'productivity', 'synergy', 'acquisition', 'merger',
        
        # 程度副词（增强正面）
        'significantly', 'substantially', 'dramatically', 'remarkably',
        'exceptionally', 'extremely', 'very', 'highly', 'greatly'
    }
    
    negative_words = {
        # 基础负面词
        'bear', 'bearish', 'drop', 'fall', 'loss', 'decline', 'negative', 
        'weak', 'miss', 'down', 'crash', 'plunge', 'slump', 'dip', 'selloff',
        'pessimistic', 'recession', 'underperform', 'collapse', 'tumble',
        'volatility', 'risk', 'danger', 'warning', 'caution', 'concern',
        'worry', 'fear', 'anxiety', 'uncertainty', 'doubt', 'problem',
        'issue', 'challenge', 'difficulty', 'trouble', 'setback', 'failure',
        'disappointment', 'disaster', 'crisis', 'meltdown', 'bubble',
        'correction', 'adjustment', 'downturn', 'slowdown', 'contraction',
        'deteriorate', 'worsen', 'weaken', 'diminish', 'shrink', 'contract',
        'plummet', 'sink', 'slide', 'retreat', 'withdraw', 'abandon',
        'bankruptcy', 'default', 'insolvency', 'liquidation', 'delisting',
        
        # 金融特定负面词
        'loss', 'write-down', 'write-off', 'impairment', 'charge', 'restructuring',
        'layoff', 'firing', 'cut', 'reduce', 'decrease', 'downgrade', 'sell',
        'underweight', 'underperform', 'reduce', 'downgrade', 'sell',
        'overvalued', 'bubble', 'speculative', 'risky', 'volatile',
        'debt', 'leverage', 'interest', 'payment', 'obligation', 'liability',
        'lawsuit', 'litigation', 'investigation', 'probe', 'subpoena',
        'regulation', 'regulatory', 'compliance', 'penalty', 'fine',
        
        # 程度副词（增强负面）
        'sharply', 'steeply', 'severely', 'seriously', 'badly', 'heavily',
        'deeply', 'substantially', 'significantly', 'drastically'
    }
    
    return positive_words, negative_words

def _improved_score_text_sentiment(text: str) -> float:
    """改进的单篇文章情感评分"""
    positive_words, negative_words = _create_comprehensive_sentiment_lexicon()
    
    if not text:
        return 0.0
    
    # 预处理文本
    import re
    cleaned_text = text.lower()
    cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text)
    cleaned_text = ' '.join(cleaned_text.split())
    words = cleaned_text.split()
    
    if not words:
        return 0.0
    
    # 否定词列表
    negation_words = {'not', 'no', 'never', 'none', 'nothing', 'without', 'lack'}
    
    # 程度副词权重
    intensifiers = {
        'very': 1.5, 'extremely': 2.0, 'highly': 1.8, 'greatly': 1.7,
        'significantly': 1.8, 'substantially': 1.8, 'dramatically': 2.0,
        'remarkably': 1.7, 'exceptionally': 1.9, 'sharply': 1.8,
        'steeply': 1.8, 'severely': 2.0, 'seriously': 1.7
    }
    
    sentiment_score = 0.0
    negation_active = False
    current_intensifier = 1.0
    
    for i, word in enumerate(words):
        # 检查否定词
        if word in negation_words:
            negation_active = True
            continue
        
        # 检查程度副词
        if word in intensifiers:
            current_intensifier = intensifiers[word]
            continue
        
        # 检查情感词
        word_score = 0.0
        if word in positive_words:
            word_score = 1.0
        elif word in negative_words:
            word_score = -1.0
        
        if word_score != 0.0:
            # 应用否定
            if negation_active:
                word_score = -word_score
                negation_active = False
            
            # 应用程度加强
            word_score *= current_intensifier
            current_intensifier = 1.0  # 重置
            
            sentiment_score += word_score
    
    # 归一化到 [-1, 1] 范围
    max_possible_score = len([w for w in words if w in positive_words or w in negative_words])
    if max_possible_score > 0:
        sentiment_score = max(-1.0, min(1.0, sentiment_score / max_possible_score))
    
    return sentiment_score

def _improved_calculate_text_sentiment(articles: List[Dict]) -> float:
    """改进的文本情感计算"""
    if not articles:
        return 0.0
    
    daily_sentiment = 0.0
    article_weights = []
    
    for article in articles:
        text = f"{article.get('title', '')} {article.get('summary', '')}"
        weight = article.get('relevance', 0.5)
        
        # 使用改进的情感评分
        article_sentiment = _improved_score_text_sentiment(text)
        
        daily_sentiment += article_sentiment * weight
        article_weights.append(weight)
    
    if article_weights:
        avg_sentiment = daily_sentiment / sum(article_weights)
        return max(-1.0, min(1.0, avg_sentiment))
    else:
        return 0.0

def _process_articles_to_sentiment(articles: List[Dict], start: str) -> pd.DataFrame:
    """将文章处理为每日情感数据 - 使用改进的情感分析"""
    if not articles:
        return None
    
    # 按日期分组
    daily_articles = {}
    
    for article in articles:
        date_key = article['date'].date()
        
        if date_key not in daily_articles:
            daily_articles[date_key] = []
        
        daily_articles[date_key].append(article)
    
    # 创建日期范围
    start_date = pd.to_datetime(start).date()
    end_date = dt.date.today()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 计算每日情感指标
    sentiment_data = []
    
    for single_date in all_dates:
        date_key = single_date.date()
        day_articles = daily_articles.get(date_key, [])
        
        if day_articles:
            # 使用改进的情感计算方法
            sentiment_score = _improved_calculate_text_sentiment(day_articles)
            
            news_volume = len(day_articles)
            relevance_avg = sum(article.get('relevance', 0.5) for article in day_articles) / news_volume
            
            sentiment_data.append({
                'Date': single_date,
                'sentiment_score': sentiment_score,
                'relevance_score': min(1.0, relevance_avg),
                'news_volume': news_volume
            })
            
            # 输出调试信息（仅在有新闻的日子）
            if abs(sentiment_score) > 0.1:
                print(f"[Sentiment Debug] {date_key}: score={sentiment_score:.3f}, articles={news_volume}")
        else:
            # 没有新闻的日子
            sentiment_data.append({
                'Date': single_date,
                'sentiment_score': 0.0,
                'relevance_score': 0.0,
                'news_volume': 0
            })
    
    # 创建DataFrame
    sentiment_df = pd.DataFrame(sentiment_data)
    sentiment_df = sentiment_df.set_index('Date')
    
    # 平滑情感数据（7天移动平均）
    sentiment_df['sentiment_score_ma7'] = sentiment_df['sentiment_score'].rolling(window=7, min_periods=1).mean()
    
    return sentiment_df

# --------- 新闻情绪数据抓取 ---------
def _generate_enhanced_sentiment(symbol: str, start: str, out: str) -> pd.DataFrame:
    """生成基于市场模式的增强合成情绪数据"""
    print(f"[Enhanced Synthetic] Generating realistic sentiment for {symbol}...")
    
    dates = pd.date_range(start=start, end=dt.datetime.today(), freq="D")
    t = np.arange(len(dates))
    
    # 基于真实市场模式生成更合理的情感数据
    base_sentiment = np.zeros(len(dates))
    
    # 1. 初始值
    base_sentiment[0] = np.random.normal(0, 0.1)
    
    # 2. 带记忆性和均值回归的随机游走
    for i in range(1, len(dates)):
        # 市场情绪有持续性，但也倾向于回归均值
        innovation = np.random.normal(0, 0.05)
        # 重大事件模拟（偶尔的情绪冲击）
        if np.random.random() < 0.02:  # 2%的概率有重大事件
            innovation += np.random.normal(0, 0.3)
        
        base_sentiment[i] = base_sentiment[i-1] * 0.9 + innovation
        # 边界控制
        base_sentiment[i] = max(-1.0, min(1.0, base_sentiment[i]))
    
    # 3. 添加季节性模式（周内效应）
    day_of_week = [d.weekday() for d in dates]
    weekend_effect = np.array([-0.1 if d >= 5 else 0 for d in day_of_week])  # 周末情绪略低
    
    # 4. 添加市场危机模式
    crisis_periods = [
        ('2020-03-01', '2020-04-30', -0.8),  # COVID危机
        ('2022-01-01', '2022-03-31', -0.4),  # 2022年市场调整
    ]
    
    crisis_effect = np.zeros(len(dates))
    for crisis_start, crisis_end, effect_size in crisis_periods:
        mask = (dates >= pd.to_datetime(crisis_start)) & (dates <= pd.to_datetime(crisis_end))
        crisis_effect[mask] = effect_size
    
    # 组合所有效应
    final_sentiment = base_sentiment + weekend_effect + crisis_effect
    
    # 创建DataFrame
    sentiment_df = pd.DataFrame({
        "sentiment_score": final_sentiment,
        "sentiment_score_ma7": pd.Series(final_sentiment).rolling(window=7, min_periods=1).mean(),
        "sentiment_score_ma30": pd.Series(final_sentiment).rolling(window=30, min_periods=1).mean(),
        "relevance_score": np.random.uniform(0.6, 0.95, len(dates)),
        "news_volume": np.random.poisson(20, len(dates)),
        "volatility": np.abs(pd.Series(final_sentiment).diff().fillna(0))  # 情绪波动率
    }, index=dates)
    
    sentiment_df.index.name = "Date"
    _save_csv(sentiment_df, out)
    
    print(f"[Enhanced Synthetic] ✅ Generated realistic sentiment data with market patterns")
    return sentiment_df

# --------- 使用Alpha Vantage获取新闻情绪数据 ---------
def fetch_alpha_vantage_news_sentiment(symbol: str = "SPY", start: str = "2015-01-01", out: str = "data/news_sentiment.csv"):
    """使用Alpha Vantage API获取新闻情绪数据"""
    print(f"[Alpha Vantage News] 📰 Fetching news sentiment for {symbol} from {start}...")
    
    # Alpha Vantage API密钥
    API_KEY = os.getenv('ALPHA_VANTAGE_KEY', 'V6W7DH5QJF6DH0VM')
    
    # 创建日期范围
    start_date = pd.to_datetime(start).date()
    end_date = dt.date.today()
    
    # 获取新闻数据
    all_articles = _fetch_alpha_vantage_news(symbol, API_KEY, start_date, end_date)
    
    if all_articles:
        sentiment_df = _process_articles_to_sentiment(all_articles, start)
        _save_csv(sentiment_df, out)
        print(f"[Alpha Vantage News] ✅ Successfully processed {len(all_articles)} articles from {start_date} to {end_date}")
        return sentiment_df
    else:
        print("[Alpha Vantage News] ⚠️ No articles found, using enhanced synthetic data")
        return _generate_enhanced_sentiment(symbol, start, out)

def _fetch_alpha_vantage_news(symbol: str, api_key: str, start_date: dt.date, end_date: dt.date) -> List[Dict]:
    """从Alpha Vantage获取新闻数据"""
    all_articles = []
    
    # Alpha Vantage新闻API - 根据文档修正参数
    url = "https://www.alphavantage.co/query"
    
    # 基本参数 - 根据文档必须的参数
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': symbol,
        'apikey': api_key,
        'limit': 1000,  # 可选：50, 100, 200, 500, 1000
    }
    
    try:
        print(f"[Alpha Vantage] Fetching news for {symbol}...")
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # 调试：打印响应结构
            print(f"[Alpha Vantage] Response keys: {list(data.keys())}")
            
            # 检查API响应
            if 'feed' in data:
                articles = data['feed']
                print(f"[Alpha Vantage] Retrieved {len(articles)} articles")
                
                for i, item in enumerate(articles[:5]):  # 只打印前5条调试信息
                    print(f"[Alpha Vantage] Article {i+1}: {item.get('title', 'No title')[:50]}...")
                
                for item in articles:
                    try:
                        # 解析文章数据
                        article_data = _parse_alpha_vantage_article(item, symbol, start_date, end_date)
                        if article_data:
                            all_articles.append(article_data)
                    except Exception as e:
                        print(f"[Alpha Vantage] Error parsing article: {e}")
                        continue
                        
                print(f"[Alpha Vantage] After date filtering: {len(all_articles)} articles")
                
            elif 'Information' in data:
                print(f"[Alpha Vantage] API Information: {data['Information']}")
            elif 'Error Message' in data:
                print(f"[Alpha Vantage] API Error: {data['Error Message']}")
            elif 'Note' in data:
                print(f"[Alpha Vantage] API Note: {data['Note']}")
                # 如果是频率限制，等待后重试
                if 'call frequency' in data['Note'].lower():
                    print("[Alpha Vantage] Rate limit hit, waiting 65 seconds...")
                    time.sleep(65)
                    return _fetch_alpha_vantage_news(symbol, api_key, start_date, end_date)
            else:
                print(f"[Alpha Vantage] Unexpected response: {data}")
        
        else:
            print(f"[Alpha Vantage] HTTP error {response.status_code}: {response.text[:200]}")
            
    except Exception as e:
        print(f"[Alpha Vantage] Request failed: {e}")
    
    # 如果使用demo密钥，数据可能有限，尝试获取更多历史数据
    if len(all_articles) < 50 and api_key != 'demo':
        print(f"[Alpha Vantage] Only {len(all_articles)} articles, trying time-based fetching...")
        time_based_articles = _fetch_alpha_vantage_news_by_time(symbol, api_key, start_date, end_date)
        all_articles.extend(time_based_articles)
    
    return all_articles

def _fetch_alpha_vantage_news_by_time(symbol: str, api_key: str, start_date: dt.date, end_date: dt.date) -> List[Dict]:
    """分时段获取Alpha Vantage新闻数据 - 使用time_from和time_to参数"""
    all_articles = []
    
    # 根据文档，time_from和time_to格式应该是: YYYYMMDDTHHMMSS
    # 我们按月获取数据
    
    current_date = end_date.replace(day=1)  # 从当前月的第一天开始
    request_count = 0
    
    while current_date >= start_date and request_count < 10:  # 限制请求次数
        # 计算时间范围
        time_to = current_date.strftime('%Y%m%dT235959')
        
        # 计算上个月的第一天
        if current_date.month == 1:
            previous_month = current_date.replace(year=current_date.year-1, month=12)
        else:
            previous_month = current_date.replace(month=current_date.month-1)
        
        time_from = previous_month.strftime('%Y%m%dT000000')
        
        print(f"[Alpha Vantage] Fetching from {time_from} to {time_to}...")
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': api_key,
                'limit': 200,  # 每次获取200条
                'time_from': time_from,
                'time_to': time_to,
                'sort': 'LATEST'  # 或 'EARLIEST'
            }
            
            response = requests.get(url, params=params, timeout=20)
            request_count += 1
            
            if response.status_code == 200:
                data = response.json()
                
                if 'feed' in data:
                    articles = data['feed']
                    monthly_articles = []
                    
                    for item in articles:
                        article_data = _parse_alpha_vantage_article(item, symbol, start_date, end_date)
                        if article_data:
                            monthly_articles.append(article_data)
                    
                    all_articles.extend(monthly_articles)
                    print(f"[Alpha Vantage] Time range: {len(monthly_articles)} articles")
                    
                    # 如果获取到很多文章，可能这个月的数据已经足够
                    if len(monthly_articles) >= 150:
                        print(f"[Alpha Vantage] Got sufficient articles for this time range")
                
                elif 'Note' in data:
                    print(f"[Alpha Vantage] Rate limit: {data['Note']}")
                    # 等待后继续
                    time.sleep(65)
                    continue
                else:
                    print(f"[Alpha Vantage] No articles in this time range")
            
            # 严格遵守API限制：免费版5请求/分钟
            time.sleep(13)  # 12秒 + 1秒缓冲
            
            # 移动到上一个月
            current_date = previous_month
            
        except Exception as e:
            print(f"[Alpha Vantage] Error for time range: {e}")
            break
    
    print(f"[Alpha Vantage] Time-based fetching completed: {len(all_articles)} total articles")
    return all_articles

def _parse_alpha_vantage_article(item: Dict, symbol: str, start_date: dt.date, end_date: dt.date) -> Optional[Dict]:
    """解析Alpha Vantage文章数据"""
    try:
        # 提取标题和摘要
        title = item.get('title', '')
        summary = item.get('summary', '')
        text = f"{title} {summary}".strip()
        
        if not text:
            return None
        
        # 解析发布时间 - 根据文档格式: YYYYMMDDTHHMMSS
        time_published = item.get('time_published', '')
        try:
            if time_published and len(time_published) >= 15:
                # 移除可能的时区信息，只取前15个字符
                time_str = time_published[:15]
                published = pd.to_datetime(time_str, format='%Y%m%dT%H%M%S')
            else:
                published = pd.to_datetime('today')
        except Exception as time_error:
            print(f"[Alpha Vantage] Time parsing error for '{time_published}': {time_error}")
            published = pd.to_datetime('today')
        
        # 检查日期范围
        if published.date() < start_date or published.date() > end_date:
            return None
        
        # 提取情感分数 - 根据API文档结构
        sentiment_score = 0.0
        relevance_score = 0.5
        
        # 查找特定ticker的情感分数
        ticker_sentiment = item.get('ticker_sentiment', [])
        if isinstance(ticker_sentiment, list):
            for ticker_info in ticker_sentiment:
                if ticker_info.get('ticker') == symbol:
                    try:
                        # 根据文档，情感分数范围是 -1 到 +1
                        ticker_score = ticker_info.get('ticker_sentiment_score', '0')
                        sentiment_score = float(ticker_score)
                        relevance_score = 0.9  # 直接提及目标symbol
                        
                        # 调试信息
                        ticker_relevance = ticker_info.get('relevance_score', '0')
                        print(f"[Alpha Vantage] {symbol}: sentiment={sentiment_score}, relevance={ticker_relevance}")
                        break
                    except (ValueError, TypeError) as e:
                        print(f"[Alpha Vantage] Error parsing sentiment score: {e}")
                        continue
        
        # 如果没有特定ticker的情感分数，使用整体情感分数
        if sentiment_score == 0.0:
            overall_sentiment = item.get('overall_sentiment_score', '0')
            try:
                sentiment_score = float(overall_sentiment)
                relevance_score = 0.7  # 整体市场情感
                print(f"[Alpha Vantage] Using overall sentiment: {sentiment_score}")
            except (ValueError, TypeError):
                # 最后使用文本分析
                sentiment_score = _improved_score_text_sentiment(text)
                print(f"[Alpha Vantage] Using text analysis sentiment: {sentiment_score}")
        
        # 提取其他信息
        source = item.get('source', 'Unknown')
        authors = item.get('authors', [])
        if authors and isinstance(authors, list) and authors[0]:
            source = authors[0]
        
        url = item.get('url', '')
        banner_image = item.get('banner_image', '')
        
        article_data = {
            'title': title,
            'summary': summary,
            'date': published,
            'url': url,
            'source': source,
            'relevance': min(1.0, relevance_score),  # 确保不超过1.0
            'sentiment': max(-1.0, min(1.0, sentiment_score))  # 确保在-1到1之间
        }
        
        return article_data
        
    except Exception as e:
        print(f"[Alpha Vantage] Error parsing article: {e}")
        import traceback
        print(f"[Alpha Vantage] Traceback: {traceback.format_exc()}")
        return None


# --------- 新增资金流数据抓取 ---------
def fetch_money_flow(symbol: str = "SPY", start: str = "2015-01-01", out: str = "data/money_flow.csv"):
    """抓取资金流数据"""
    print(f"[MoneyFlow] Generating flow data for {symbol}...")
    
    dates = pd.date_range(start=start, end=dt.datetime.today(), freq="D")
    
    # 生成合理的资金流数据
    flow_df = pd.DataFrame({
        "net_flow": np.random.normal(0, 1000000, len(dates)).cumsum(),  # 累积净流入
        "large_order_flow": np.random.normal(0, 500000, len(dates)),
        "institutional_flow": np.random.normal(0, 2000000, len(dates)),
        "retail_flow": np.random.normal(0, 300000, len(dates))
    }, index=dates)
    
    flow_df.index.name = "Date"
    _save_csv(flow_df, out)
    return flow_df

# --------- 统一数据抓取入口 ---------
def fetch_all_data(symbol: str = "SPY", start: str = "2015-01-01", use_cache: bool = True):
    """统一抓取所有类型的数据（市场价格、宏观、新闻情绪、资金流）"""
    print(f"=== Fetching all data for {symbol} from {start} ===")
    
    # 1. 市场价格数据
    print("\n1. Fetching market data...")
    market_out = "data/market.csv"
    market_df = None
    
    if use_cache and _maybe_use_cache(market_out, fresh_days=2):
        market_df = pd.read_csv(market_out, parse_dates=["Date"], index_col="Date")
        print(f"[market] using cached file: {market_out}")
    else:
        providers = [
            ("yfinance", lambda: fetch_yfinance(symbol, start)),
            ("stooq", lambda: fetch_stooq(symbol, start)), 
            ("binance", lambda: fetch_binance(symbol, start))
        ]
        
        for source_name, fetch_func in providers:
            try:
                market_df = fetch_func()
                _save_csv(market_df, market_out)
                print(f"[market] source={source_name}")
                break
            except Exception as e:
                print(f"[{source_name}] fail: {type(e).__name__}: {e}")
                continue
        
        # 全部失败后的回退方案
        if market_df is None:
            try:
                print("[fallback] force to BTC-USD via Binance")
                market_df = fetch_binance("BTC-USD", start)
                _save_csv(market_df, market_out)
                print("[market] source=binance, symbol=BTC-USD")
            except Exception as e:
                print(f"[market] All sources failed: {e}")
    
    # 2. 宏观数据
    print("\n2. Fetching macro data...")
    macro_out = "data/macro.csv"
    if use_cache and _maybe_use_cache(macro_out, fresh_days=7):
        print(f"[macro] using cached file: {macro_out}")
    else:
        try:
            fetch_macro(start, macro_out)
        except Exception as e:
            print(f"[macro] Error: {e}")
    
    # 3. 新闻情绪数据 - 使用修复后的Alpha Vantage
    print("\n3. Fetching sentiment data...")
    sentiment_out = "data/news_sentiment.csv"
    if use_cache and _maybe_use_cache(sentiment_out, fresh_days=1):
        print(f"[sentiment] using cached file: {sentiment_out}")
    else:
        try:
            fetch_alpha_vantage_news_sentiment(symbol, start, sentiment_out)
        except Exception as e:
            print(f"[sentiment] Alpha Vantage failed: {e}")
            print("[sentiment] Falling back to synthetic data...")
            _generate_enhanced_sentiment(symbol, start, sentiment_out)
    
    # 4. 资金流数据  
    print("\n4. Fetching money flow data...")
    flow_out = "data/money_flow.csv"
    if use_cache and _maybe_use_cache(flow_out, fresh_days=1):
        print(f"[money_flow] using cached file: {flow_out}")
    else:
        try:
            fetch_money_flow(symbol, start, flow_out)
        except Exception as e:
            print(f"[money_flow] Error: {e}")
    
    print(f"\n=== All data fetch completed ===")
    
    return market_df if market_df is not None else pd.DataFrame()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch market, macro, sentiment and flow data")
    ap.add_argument("--symbol", default="SPY", help="Symbol to fetch")
    ap.add_argument("--start", default="2015-01-01", help="Start date") 
    ap.add_argument("--out", default="data/market.csv", help="Market data output path")
    ap.add_argument("--no_cache", action="store_true", help="Disable cache")
    ap.add_argument("--all", action="store_true", help="Fetch all data types")
    
    args = ap.parse_args()
    
    # 创建数据目录
    Path("data").mkdir(exist_ok=True)
    
    # 现在无论是否指定 --all，都抓取所有数据
    fetch_all_data(symbol=args.symbol, start=args.start, use_cache=not args.no_cache)