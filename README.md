# 金融风险预测系统

AruraFin是一个集成了数据抓取、特征分析和多任务深度学习模型的金融风险预测系统，旨在为用户提供全面的金融市场数据分析与风险评估能力。

## 一、系统概述
本系统主要包含三大核心模块：
- 多源数据抓取模块：自动获取市场价格、宏观经济、新闻情绪和资金流数据
- 特征因果分析模块：分析各类金融特征与目标变量的因果关系
- 多任务深度学习模块：构建并训练用于金融风险预测的深度学习模型

## 二、核心功能

### 1. 数据自动抓取
- 支持多源市场价格数据获取（yfinance、stooq、binance）
- 宏观经济数据采集与更新
- 新闻情绪分析数据获取（含合成数据 fallback 机制）
- 资金流数据跟踪
- 本地缓存机制，减少重复请求

### 2. 特征因果分析

- 基于多窗口滑动分析的特征因果检验

- 自动计算特征最优滞后项与显著性

- 提供特征稳定性评估（通过检验/不稳定）

### 3. 多任务风险预测
- 支持多目标联合训练（分位数、波动率、CVaR、趋势、风险等级）
- 动态任务权重调整机制
- 早停策略防止过拟合
- 自动保存与加载最优模型

## 三、快速开始
### 1. 数据抓取
```python

from src.data.fetch import fetch_all_data

# 抓取SPY从2015年开始的所有数据
market_df = fetch_all_data(
    symbol="SPY",
    start="2015-01-01",
    use_cache=True  # 使用缓存加速
)
```

### 2. 模型训练
```python

from src.app.app_flask import train_multitask_model
import torch

# 准备训练数据（X_train, train_targets等）
# ...

# 初始化模型
model = YourModel()  # 替换为实际模型类

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model = train_multitask_model(
    model,
    X_train, train_targets,
    X_val, val_targets,
    device,
    model_name="RiskPredictor",
    epochs=100,
    patience=15
)
```

## 四、特征分析结果

根据特征因果分析（`feature_causality.json`），主要金融特征表现如下：

|特征|中文名称|最佳滞后项|检验结果|
|---|---|---|---|
|Open|开盘价|3|通过检验|
|Low|最低价|3|通过检验|
|Close|收盘价|3|通过检验|
|Volume|成交量|2|通过检验|
|rsi|RSI(14)|-|不稳定|
|macd|MACD|-|-|
*`feature_causality.json`* *注：完整结果请查看  文件*

## 五、配置说明
- 数据缓存有效期：市场数据(2天)、宏观数据(7天)、情绪数据(1天)、资金流数据(1天)
- 模型训练参数：
    - 优化器：Adam (学习率0.001，权重衰减1e-4)
    - 学习率调度：ReduceLROnPlateau
    - 分位数：[0.01, 0.05, 0.5, 0.95, 0.99]（极端分位权重更高）
    - 早停耐心值：15个epoch

## 六、项目结构

```text

root/
├── data/
│   ├── feature_causality.json  # 特征因果分析结果
│   └── ...（数据文件）
├── src/
│   ├── data/
│   │   └── fetch.py            # 数据抓取模块
│   └── app/
│       └── app_flask.py        # 模型训练模块
└── models/                     # 模型保存目录
```

## 七、注意事项
- 数据抓取可能受限于第三方API的访问限制
- 模型训练需要适当的计算资源，建议使用GPU加速
- 历史数据仅供参考，不构成任何投资建议
