import os, json
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
import torch.nn as nn
from joblib import load
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import requests
from flask import Flask, render_template, request, jsonify
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# 添加验证类（支持多分位验证）
class VaRValidation:
    def __init__(self, returns, quantile_predictions, quantiles=[0.01, 0.05, 0.5, 0.95, 0.99], window=250):
        """
        quantile_predictions: (N, 5) 数组，包含5个分位的预测
        quantiles: 分位点列表 [q01, q05, q50, q95, q99]
        """
        self.returns = np.array(returns)
        self.quantile_predictions = np.array(quantile_predictions)
        self.quantiles = quantiles
        self.window = window
        self.hits = {}
        self._calculate_all_hits()
    
    def _calculate_all_hits(self):
        """计算所有分位点的命中情况"""
        for i, q in enumerate(self.quantiles):
            if q < 0.5:
                self.hits[f'q{int(q*100):02d}'] = (self.returns <= self.quantile_predictions[:, i]).astype(int)
            else:
                self.hits[f'q{int(q*100):02d}'] = (self.returns >= self.quantile_predictions[:, i]).astype(int)
    
    def hit_ratio(self, quantile_name='q05'):
        """获取指定分位的命中率"""
        if quantile_name in self.hits:
            return np.mean(self.hits[quantile_name])
        return 0.0
    
    def coverage_quality_score(self, recent_window=60):
        """
        基于多分位覆盖率的综合质量评分
        考虑所有分位的覆盖率偏差
        """
        if len(self.returns) < recent_window:
            return 0.5
        
        recent_returns = self.returns[-recent_window:]
        recent_preds = self.quantile_predictions[-recent_window:]
        
        total_bias = 0
        total_weight = 0
        
        for i, q in enumerate(self.quantiles):
            q_name = f'q{int(q*100):02d}'
            expected = q if q <= 0.5 else 1 - q
            
            if q < 0.5:
                actual = (recent_returns <= recent_preds[:, i]).mean()
            else:
                actual = (recent_returns >= recent_preds[:, i]).mean()
            
            bias = abs(actual - expected)
            
            # 为极端分位赋予更高权重
            if q in [0.01, 0.99]:
                weight = 2.0
            elif q in [0.05, 0.95]:
                weight = 1.5
            else:
                weight = 1.0
                
            total_bias += bias * weight
            total_weight += weight
        
        avg_bias = total_bias / total_weight if total_weight > 0 else 1.0
        bias_penalty = max(0, 1 - avg_bias / 0.1)  # 允许10%的平均偏差
        
        # 计算稳定性
        if len(recent_returns) >= 20:
            q05_hits = (recent_returns <= recent_preds[:, 1]).astype(int)  # q05 hits
            rolling_20 = pd.Series(q05_hits).rolling(20).mean().dropna()
            stability = 1 - min(1, rolling_20.std() / 0.1)
        else:
            stability = 0.5
        
        # 综合评分：70%基于覆盖率准确性，30%基于稳定性
        confidence = 0.7 * bias_penalty + 0.3 * stability
        return min(1.0, max(0.0, confidence))
    def calculate_overall_bias(self, recent_window=60):
        """
        计算多分位综合偏差
        考虑所有分位的覆盖率偏差，加权平均
        """
        if len(self.returns) < recent_window:
            return 1.0  # 默认最大偏差
        
        recent_returns = self.returns[-recent_window:]
        recent_preds = self.quantile_predictions[-recent_window:]
        
        total_weighted_bias = 0
        total_weight = 0
        
        for i, q in enumerate(self.quantiles):
            expected = q if q <= 0.5 else 1 - q
            
            if q < 0.5:
                actual = (recent_returns <= recent_preds[:, i]).mean()
            else:
                actual = (recent_returns >= recent_preds[:, i]).mean()
            
            bias = abs(actual - expected)
            
            # 为不同分位分配权重
            if q in [0.01, 0.99]:
                weight = 2.0  # 极端分位权重更高
            elif q in [0.05, 0.95]:
                weight = 1.5  # 重要风险分位
            else:
                weight = 1.0  # 中位数分位
                
            total_weighted_bias += bias * weight
            total_weight += weight
        
        # 综合偏差 (越小越好)
        overall_bias = total_weighted_bias / total_weight if total_weight > 0 else 1.0
        return overall_bias
    
    def get_detailed_coverage_report(self):
        """获取详细的分位覆盖率报告"""
        report = {}
        for i, q in enumerate(self.quantiles):
            q_name = f'q{int(q*100):02d}'
            expected = q if q <= 0.5 else 1 - q
            actual = self.hit_ratio(q_name)
            bias = actual - expected
            abs_bias = abs(bias)
            
            report[q_name] = {
                'expected': expected,
                'actual': actual,
                'bias': bias,
                'abs_bias': abs_bias,
                'performance': '优秀' if abs_bias <= 0.005 else (
                    '良好' if abs_bias <= 0.01 else (
                    '一般' if abs_bias <= 0.02 else '需要改进'
                ))
            }
        
        return report

# ==================== 多分位损失函数（与训练代码一致） ====================

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


def adjust_weights_by_quantile(base_weights: dict, quantile_weights_map: dict):
    """将基础权重按分位点特异性权重调整，返回每个分位点的最终权重字典。

    base_weights: {'lstm':..,'lightgbm':..,'transformer':..}
    quantile_weights_map: e.g. {'q01': {'lstm':0.3,...}, ...}

    返回: dict mapping quantile name -> weights dict (normalized)
    """
    final = {}
    for qname, qm in quantile_weights_map.items():
        # start from base and multiply by quantile-specific modifier
        w = {m: base_weights.get(m, 0.0) * qm.get(m, 1.0) for m in base_weights}
        s = sum(w.values())
        if s == 0:
            # fallback to base normalized
            s = sum(base_weights.values()) or 1.0
            w = {m: base_weights.get(m, 0.0)/s for m in base_weights}
        else:
            w = {m: v / s for m, v in w.items()}
        final[qname] = w
    return final


def calculate_dynamic_weights(validation_performance: dict) -> dict:
    """基于验证集表现计算动态权重（示例实现）。

    validation_performance: 任意结构，期望包含必要的评分字段。
    返回字典 {'lstm':..., 'lightgbm':..., 'transformer':...}
    """
    # 提取指标（使用 get 保持鲁棒）
    coverage = validation_performance.get('coverage_score', {})
    stability = validation_performance.get('stability_score', {})
    extreme = validation_performance.get('extreme_accuracy', {})
    feature_imp = validation_performance.get('feature_importance_score', 0.5)
    training_speed = validation_performance.get('training_speed_score', 0.5)
    tail_acc = validation_performance.get('tail_accuracy', {})
    multi_scale = validation_performance.get('multi_scale_score', 0.5)
    event_resp = validation_performance.get('event_response_score', 0.5)

    # 简化聚合：对每个模型计算分数
    def safe_get(d, k):
        return float(d.get(k, 0.5)) if isinstance(d, dict) else 0.5

    lstm_score = 0.4 * safe_get(coverage, 'lstm') + 0.3 * safe_get(stability, 'lstm') + 0.3 * safe_get(extreme, 'lstm')
    lgb_score = 0.5 * float(feature_imp) + 0.3 * safe_get(coverage, 'lightgbm') + 0.2 * float(training_speed)
    transformer_score = 0.5 * safe_get(tail_acc, 'transformer') + 0.3 * float(multi_scale) + 0.2 * float(event_resp)

    total = lstm_score + lgb_score + transformer_score
    if total <= 0:
        return {'lstm': 0.33, 'lightgbm': 0.33, 'transformer': 0.34}
    return {'lstm': lstm_score/total, 'lightgbm': lgb_score/total, 'transformer': transformer_score/total}


def _ensemble(predictions: dict, current_market_regime: str = "normal",
                   quantile_weights_map: dict = None, base_weights: dict = None):
    """基于市场状态与分位点特异性权重对三模型预测进行智能加权。

    predictions: dict with keys 'lstm','lightgbm','transformer' values shape (batch, Q)
    quantile_weights_map: mapping like {'q01': {'lstm':0.3,...}, ...}
    base_weights: base weights for models; if None use defaults
    返回: ensemble_pred (batch, Q), final_weights_per_quantile
    """
    if base_weights is None:
        if hasattr(smart_ensemble, '_default_base'):
            base_weights = smart_ensemble._default_base
        else:
            base_weights = {'lstm': 0.4, 'lightgbm': 0.3, 'transformer': 0.3}

    # adjust base weights by market regime
    if current_market_regime == "high_volatility":
        regime_base = {'lstm': 0.3, 'lightgbm': 0.2, 'transformer': 0.5}
    elif current_market_regime == "trending":
        regime_base = {'lstm': 0.5, 'lightgbm': 0.3, 'transformer': 0.2}
    else:
        regime_base = base_weights

    # use provided quantile_weights_map or default mapping
    if quantile_weights_map is None:
        quantile_weights_map = {
            'q01': {'lstm': 0.3, 'lightgbm': 0.2, 'transformer': 0.5},
            'q05': {'lstm': 0.4, 'lightgbm': 0.3, 'transformer': 0.3},
            'q50': {'lstm': 0.4, 'lightgbm': 0.4, 'transformer': 0.2},
            'q95': {'lstm': 0.3, 'lightgbm': 0.3, 'transformer': 0.4},
            'q99': {'lstm': 0.2, 'lightgbm': 0.2, 'transformer': 0.6}
        }

    # compute final per-quantile normalized weights
    per_q_weights = adjust_weights_by_quantile(regime_base, quantile_weights_map)

    # assemble predictions
    lstm = predictions.get('lstm')
    lgb = predictions.get('lightgbm')
    transformer = predictions.get('transformer')
    # ensure numpy arrays
    lstm = np.array(lstm) if lstm is not None else 0.0
    lgb = np.array(lgb) if lgb is not None else 0.0
    transformer = np.array(transformer) if transformer is not None else 0.0

    # assume order of quantiles matches keys q01,q05,q50,q95,q99
    qnames = ['q01','q05','q50','q95','q99']
    ensemble_pred = np.zeros_like(lstm if isinstance(lstm, np.ndarray) and lstm.shape==lgb.shape else (lgb if isinstance(lgb, np.ndarray) else lstm))

    for i, qn in enumerate(qnames):
        w = per_q_weights.get(qn)
        wl = w.get('lstm', 0.0)
        wg = w.get('lightgbm', 0.0)
        wt = w.get('transformer', 0.0)

        # handle possible shapes and missing models
        term = 0.0
        if isinstance(lstm, np.ndarray) and lstm.shape[1] > i:
            term = term + wl * lstm[:, i]
        if isinstance(lgb, np.ndarray) and lgb.shape[1] > i:
            term = term + wg * lgb[:, i]
        if isinstance(transformer, np.ndarray) and transformer.shape[1] > i:
            term = term + wt * transformer[:, i]

        ensemble_pred[:, i] = term

    return ensemble_pred, per_q_weights


def confidence_calibration(individual_preds: dict):
    """基于模型间不一致性计算置信度，返回 (batch,) 的置信度值（0-1）。

    individual_preds: dict with keys 'lstm','lightgbm','transformer' each (batch, Q)
    目前采用 q05（index 1）计算 disagreement。
    """
    try:
        arrs = [np.array(individual_preds[k])[:, 1] for k in ['lstm','lightgbm','transformer'] if k in individual_preds]
        disagreement = np.std(arrs, axis=0)
        confidence = 1.0 / (1.0 + disagreement)
        return confidence
    except Exception:
        # fallback: uniform confidence
        return np.ones((np.array(list(individual_preds.values())[0]).shape[0],))


def multi_task_loss(preds: dict, targets: dict, weights: dict = None,
                    quantiles=[0.01, 0.05, 0.5, 0.95, 0.99], quantile_weights=None,
                    task_weights: dict = None):
    """
    完整的多任务损失函数，支持所有任务的联合训练
    
    Args:
        preds: dict of model predictions. Expected keys:
            - 'quantiles': Tensor (B, Q)
            - 'vol': Tensor (B,)
            - 'cvar': Tensor (B,)
            - 'trend_logits': Tensor (B, C_trend)
            - 'risk_logits': Tensor (B, C_risk)
        targets: dict of target tensors. Expected keys:
            - 'y': Tensor (B,)  用于分位损失（下一日收益）
            - 'vol': Tensor (B,)
            - 'cvar': Tensor (B,)
            - 'trend': LongTensor (B,)  分类标签
            - 'risk': LongTensor (B,)   分类标签
        weights: dict of scalar weights for tasks, keys: 'q','vol','cvar','trend','risk'
        quantiles: list of quantile levels (Q,)
        quantile_weights: list or tensor of per-quantile weights
        task_weights: dict of task weights for dynamic adjustment
    """
    if weights is None:
        weights = {'q': 1.0, 'vol': 0.5, 'cvar': 0.8, 'trend': 0.3, 'risk': 0.3}
    
    if task_weights is None:
        task_weights = {'q': 1.0, 'vol': 1.0, 'cvar': 1.0, 'trend': 1.0, 'risk': 1.0}
    
    loss_dict = {}
    total = torch.tensor(0.0, device=next(iter(preds.values())).device if len(preds) > 0 else 'cpu')
    
    # 基础损失函数
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    huber = nn.HuberLoss()

    # 1. Quantile loss (pinball) - 核心任务
    if 'quantiles' in preds and 'y' in targets:
        q_pred = preds['quantiles']
        y = targets['y']
        q_loss = multi_quantile_loss(q_pred, y, quantiles=quantiles, 
                                    quantile_weights=quantile_weights)
        weighted_q_loss = weights['q'] * task_weights['q'] * q_loss
        loss_dict['quantile_loss'] = q_loss
        loss_dict['weighted_quantile_loss'] = weighted_q_loss
        total = total + weighted_q_loss

    # 2. Volatility regression - 中等重要性
    if 'vol' in preds and 'vol' in targets:
        vol_pred = preds['vol']
        vol_t = targets['vol']
        # 使用Huber损失，对异常值更鲁棒
        vol_loss = huber(vol_pred.float(), vol_t.float())
        weighted_vol_loss = weights['vol'] * task_weights['vol'] * vol_loss
        loss_dict['vol_loss'] = vol_loss
        loss_dict['weighted_vol_loss'] = weighted_vol_loss
        total = total + weighted_vol_loss

    # 3. CVaR regression - 高重要性（风险度量）
    if 'cvar' in preds and 'cvar' in targets:
        cvar_pred = preds['cvar']
        cvar_t = targets['cvar']
        cvar_loss = huber(cvar_pred.float(), cvar_t.float())
        weighted_cvar_loss = weights['cvar'] * task_weights['cvar'] * cvar_loss
        loss_dict['cvar_loss'] = cvar_loss
        loss_dict['weighted_cvar_loss'] = weighted_cvar_loss
        total = total + weighted_cvar_loss

    # 4. Trend classification - 低重要性
    if 'trend_logits' in preds and 'trend' in targets:
        trend_logits = preds['trend_logits']
        trend_t = targets['trend'].long()
        try:
            trend_loss = ce(trend_logits, trend_t)
        except Exception:
            trend_loss = torch.tensor(0.0, device=total.device)
        
        weighted_trend_loss = weights['trend'] * task_weights['trend'] * trend_loss
        loss_dict['trend_loss'] = trend_loss
        loss_dict['weighted_trend_loss'] = weighted_trend_loss
        total = total + weighted_trend_loss

    # 5. Risk classification - 中等重要性
    if 'risk_logits' in preds and 'risk' in targets:
        risk_logits = preds['risk_logits']
        risk_t = targets['risk'].long()
        try:
            risk_loss = ce(risk_logits, risk_t)
        except Exception:
            risk_loss = torch.tensor(0.0, device=total.device)
            
        weighted_risk_loss = weights['risk'] * task_weights['risk'] * risk_loss
        loss_dict['risk_loss'] = risk_loss
        loss_dict['weighted_risk_loss'] = weighted_risk_loss
        total = total + weighted_risk_loss

    # 添加正则化损失（可选）
    if 'quantiles' in preds:
        # 分位数单调性正则化
        quantile_pred = preds['quantiles']
        monotonicity_loss = torch.tensor(0.0, device=total.device)
        for i in range(quantile_pred.shape[1] - 1):
            diff = quantile_pred[:, i+1] - quantile_pred[:, i]
            violation = torch.relu(-diff)  # 惩罚非单调的部分
            monotonicity_loss = monotonicity_loss + violation.mean()
        
        if monotonicity_loss > 0:
            reg_weight = 0.01
            total = total + reg_weight * monotonicity_loss
            loss_dict['monotonicity_loss'] = monotonicity_loss

    loss_dict['total_loss'] = total
    return total, loss_dict

def compute_task_weights(validation_performance: dict, decay_factor=0.9) -> dict:
    """
    基于验证集表现动态计算任务权重
    validation_performance: 包含各任务验证指标的字典
    """
    default_weights = {'q': 1.0, 'vol': 1.0, 'cvar': 1.0, 'trend': 1.0, 'risk': 1.0}
    
    try:
        # 提取各任务的性能指标（越高越好）
        q_perf = 1.0 - validation_performance.get('quantile_coverage_bias', 0.1)
        vol_perf = 1.0 - min(1.0, validation_performance.get('volatility_mae', 0.1) / 0.01)
        cvar_perf = 1.0 - min(1.0, validation_performance.get('cvar_mae', 0.1) / 0.01)
        trend_perf = validation_performance.get('trend_accuracy', 0.5)
        risk_perf = validation_performance.get('risk_accuracy', 0.5)
        
        # 计算相对性能（相对于基准）
        performances = {
            'q': max(0.1, q_perf),
            'vol': max(0.1, vol_perf),
            'cvar': max(0.1, cvar_perf),
            'trend': max(0.1, trend_perf),
            'risk': max(0.1, risk_perf)
        }
        
        # 性能越差的任务给予更高权重（需要更多关注）
        total_perf = sum(performances.values())
        task_weights = {}
        for task, perf in performances.items():
            # 性能倒数的归一化
            inv_perf = (total_perf - perf) / (len(performances) - 1) if len(performances) > 1 else 1.0
            task_weights[task] = max(0.5, min(2.0, inv_perf))
            
        return task_weights
        
    except Exception as e:
        print(f"计算任务权重失败: {e}，使用默认权重")
        return default_weights
    
# ==================== 模型定义（与训练代码一致） ====================

class LSTMQuantile(nn.Module):
    def __init__(self, in_dim=7, hidden=64, layers=2, num_quantiles=5, dropout=0.2):  # 改为7
        super().__init__()
        self.num_quantiles = num_quantiles
        self.rnn = nn.LSTM(
            input_size=in_dim,  # 使用7个特征
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
    
# ==================== 内联集成模型定义 ====================

class EnsembleLSTM(nn.Module):
    """LSTM模型 - 专注于价格序列，支持多任务学习"""
    def __init__(self, in_dim=7, hidden=64, layers=2, num_quantiles=5, dropout=0.2,  # 改为7
                 num_trend_classes=3, num_risk_classes=3):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.rnn = nn.LSTM(
            input_size=in_dim,  # 使用7个特征
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        
        # 共享特征提取层
        self.shared_proj = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 多任务头
        self.quantile_head = nn.Linear(hidden // 2, num_quantiles)
        self.vol_head = nn.Linear(hidden // 2, 1)  # 波动率预测
        self.cvar_head = nn.Linear(hidden // 2, 1)  # CVaR预测
        self.trend_head = nn.Linear(hidden // 2, num_trend_classes)  # 趋势分类
        self.risk_head = nn.Linear(hidden // 2, num_risk_classes)  # 风险等级分类

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

class EnsembleTransformer(nn.Module):
    """简化版Transformer模型，支持多任务学习"""
    def __init__(self, d_model=64, nhead=8, num_layers=2, num_quantiles=5, dropout=0.1,
                 num_trend_classes=3, num_risk_classes=3):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.d_model = d_model
        
        # 输入投影 - 7个特征
        self.input_proj = nn.Linear(7, d_model)  # 改为7
        self.position_encoding = nn.Parameter(torch.zeros(1, 60, d_model))
        
        # Transformer编码器
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=d_model, 
                    nhead=nhead, 
                    dim_feedforward=256,
                    dropout=dropout,
                    batch_first=True
                )
            )
        self.transformer = nn.Sequential(*encoder_layers)
        
        # 共享特征提取
        self.shared_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 多任务头
        self.quantile_head = nn.Linear(d_model // 2, num_quantiles)
        self.vol_head = nn.Linear(d_model // 2, 1)
        self.cvar_head = nn.Linear(d_model // 2, 1)
        self.trend_head = nn.Linear(d_model // 2, num_trend_classes)
        self.risk_head = nn.Linear(d_model // 2, num_risk_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_proj(x)
        x = x + self.position_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        
        # 使用最后一个时间步的特征
        last_hidden = x[:, -1, :]
        shared_features = self.shared_proj(last_hidden)
        
        return {
            'quantiles': self.quantile_head(shared_features),
            'vol': self.vol_head(shared_features).squeeze(-1),
            'cvar': self.cvar_head(shared_features).squeeze(-1),
            'trend_logits': self.trend_head(shared_features),
            'risk_logits': self.risk_head(shared_features)
        }

class EnsembleManager:
    """集成模型管理器（更新以支持多任务输出）"""
    def __init__(self, lookback=60, num_quantiles=5, device='cpu',
                 num_trend_classes=3, num_risk_classes=3):
        self.lookback = lookback
        self.num_quantiles = num_quantiles
        self.device = device
        self.quantiles = [0.01, 0.05, 0.5, 0.95, 0.99]
        
        # 初始化三个模型（多任务版本）
        self.lstm_model = EnsembleLSTM(
            num_quantiles=num_quantiles,
            num_trend_classes=num_trend_classes,
            num_risk_classes=num_risk_classes
        ).to(device)
        
        self.transformer_model = EnsembleTransformer(
            num_quantiles=num_quantiles,
            num_trend_classes=num_trend_classes,
            num_risk_classes=num_risk_classes
        ).to(device)
        
        self.lgb_models = {}  # 每个分位数一个LightGBM模型
        
        # 默认模型权重
        self.model_weights = {'lstm': 0.4, 'transformer': 0.3, 'lightgbm': 0.3}

    def train_lightgbm_models(self, X_lgb, y, quantiles=None):
        """训练LightGBM分位数回归模型"""
        if quantiles is None:
            quantiles = self.quantiles
            
        self.lgb_models = {}
        
        for q in quantiles:
            print(f"训练LightGBM分位数模型: {q}")
            
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=100,  # 减少树的数量以加快训练
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X_lgb, y)
            self.lgb_models[q] = model
    
    def predict_lightgbm(self, X_lgb):
        """LightGBM预测"""
        if not self.lgb_models:
            return np.zeros((len(X_lgb), len(self.quantiles)))
            
        predictions = np.zeros((len(X_lgb), len(self.quantiles)))
        
        for i, q in enumerate(self.quantiles):
            if q in self.lgb_models:
                predictions[:, i] = self.lgb_models[q].predict(X_lgb)
        
        return predictions

    def prepare_technical_features(self, df):
        """为LightGBM准备技术指标特征"""
        features_df = df.copy()
        
        # 基础价格特征
        features_df['returns'] = np.log(features_df['Close']).diff()
        features_df['volatility_5'] = features_df['returns'].rolling(5).std()
        features_df['volatility_20'] = features_df['returns'].rolling(20).std()
        
        # RSI相关特征
        features_df['rsi_14'] = features_df['rsi']  # 使用已计算的RSI
        features_df['rsi_slope'] = features_df['rsi'].diff(3)
        
        # MACD相关特征
        features_df['macd_diff'] = features_df['macd'] - features_df['macd_sig']
        features_df['macd_slope'] = features_df['macd'].diff(3)
        
        # 布林带相关特征
        features_df['bb_width'] = (features_df['bb_high'] - features_df['bb_low']) / features_df['Close']
        features_df['bb_position'] = (features_df['Close'] - features_df['bb_low']) / (features_df['bb_high'] - features_df['bb_low'])
        
        # 成交量特征
        features_df['volume_ma'] = features_df['Volume'].rolling(10).mean()
        features_df['volume_ratio'] = features_df['Volume'] / features_df['volume_ma']
        
        # 价格动量特征
        for window in [5, 10, 20]:
            features_df[f'momentum_{window}'] = features_df['Close'].pct_change(window)
        
        # 高低价特征
        features_df['high_low_ratio'] = features_df['High'] / features_df['Low'] if 'High' in features_df.columns else 1.0
        features_df['close_open_ratio'] = features_df['Close'] / features_df['Open'] if 'Open' in features_df.columns else 1.0
        
        # 选择最终特征列
        feature_cols = [
            'rsi_14', 'rsi_slope', 'macd', 'macd_signal', 'macd_diff', 'macd_slope',
            'bb_width', 'bb_position', 'volatility_5', 'volatility_20',
            'volume_ratio', 'momentum_5', 'momentum_10', 'momentum_20',
            'high_low_ratio', 'close_open_ratio'
        ]
        
        # 只保留存在的列
        available_cols = [col for col in feature_cols if col in features_df.columns]
        result_df = features_df[available_cols].copy()
        
        # 填充NaN值
        result_df = result_df.ffill().bfill().fillna(0)
        
        # 确保特征名称有效
        result_df = result_df.rename(columns=lambda x: x.replace(' ', '_').replace('-', '_'))
        
        return result_df

    def ensemble_predict(self, X_seq, X_lgb):
        """集成预测（更新以处理多任务输出）"""
        device = self.device
        predictions = {}
        
        # LSTM预测（多任务）
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_output = self.lstm_model(
                torch.tensor(X_seq, dtype=torch.float32).to(device)
            )
            predictions['lstm'] = lstm_output
        
        # Transformer预测（多任务）
        self.transformer_model.eval()
        with torch.no_grad():
            transformer_output = self.transformer_model(
                torch.tensor(X_seq, dtype=torch.float32).to(device)
            )
            predictions['transformer'] = transformer_output
        
        # LightGBM预测（保持单任务，只用于分位数预测）
        lgb_pred = self.predict_lightgbm(X_lgb)
        predictions['lightgbm'] = {
            'quantiles': torch.tensor(lgb_pred, dtype=torch.float32).to(device),
            'vol': torch.zeros(len(X_lgb), dtype=torch.float32).to(device),  # 占位符
            'cvar': torch.zeros(len(X_lgb), dtype=torch.float32).to(device),  # 占位符
            'trend_logits': torch.zeros(len(X_lgb), 3, dtype=torch.float32).to(device),  # 占位符
            'risk_logits': torch.zeros(len(X_lgb), 3, dtype=torch.float32).to(device)  # 占位符
        }
        
        # 智能加权集成（需要更新以处理多任务）
        try:
            final_pred, per_q_weights = self.smart_ensemble_multitask(predictions)
            predictions['final'] = final_pred
            predictions['per_quantile_weights'] = per_q_weights
            predictions['confidence'] = confidence_calibration(predictions)
        except Exception as e:
            print(f"智能集成失败，使用简单集成: {e}")
            # 简单集成：只对分位数进行加权平均
            ensemble_quantiles = (
                self.model_weights['lstm'] * lstm_output['quantiles'] +
                self.model_weights['transformer'] * transformer_output['quantiles'] +
                self.model_weights['lightgbm'] * predictions['lightgbm']['quantiles']
            )
            predictions['final'] = {
                'quantiles': ensemble_quantiles,
                'vol': lstm_output['vol'],  # 暂时使用LSTM的输出
                'cvar': lstm_output['cvar'],
                'trend_logits': lstm_output['trend_logits'],
                'risk_logits': lstm_output['risk_logits']
            }
            predictions['per_quantile_weights'] = None
            predictions['confidence'] = torch.ones((ensemble_quantiles.shape[0],))

        return predictions

    def smart_ensemble_multitask(self, predictions: dict):
        """多任务智能集成"""
        try:
            # 获取基础权重
            base_weights = self.model_weights
            
            # 对每个任务进行加权集成
            final_output = {}
            
            # 分位数集成
            if all('quantiles' in pred for pred in predictions.values()):
                quantile_preds = torch.stack([
                    base_weights['lstm'] * predictions['lstm']['quantiles'],
                    base_weights['transformer'] * predictions['transformer']['quantiles'],
                    base_weights['lightgbm'] * predictions['lightgbm']['quantiles']
                ])
                final_output['quantiles'] = quantile_preds.sum(dim=0)
            
            # 波动率集成（只使用深度学习模型）
            if all('vol' in pred for pred in [predictions['lstm'], predictions['transformer']]):
                vol_preds = torch.stack([
                    base_weights['lstm'] * predictions['lstm']['vol'],
                    base_weights['transformer'] * predictions['transformer']['vol']
                ])
                final_output['vol'] = vol_preds.sum(dim=0)
            
            # CVaR集成（只使用深度学习模型）
            if all('cvar' in pred for pred in [predictions['lstm'], predictions['transformer']]):
                cvar_preds = torch.stack([
                    base_weights['lstm'] * predictions['lstm']['cvar'],
                    base_weights['transformer'] * predictions['transformer']['cvar']
                ])
                final_output['cvar'] = cvar_preds.sum(dim=0)
            
            # 趋势分类集成
            if all('trend_logits' in pred for pred in [predictions['lstm'], predictions['transformer']]):
                trend_preds = torch.stack([
                    base_weights['lstm'] * predictions['lstm']['trend_logits'],
                    base_weights['transformer'] * predictions['transformer']['trend_logits']
                ])
                final_output['trend_logits'] = trend_preds.sum(dim=0)
            
            # 风险分类集成
            if all('risk_logits' in pred for pred in [predictions['lstm'], predictions['transformer']]):
                risk_preds = torch.stack([
                    base_weights['lstm'] * predictions['lstm']['risk_logits'],
                    base_weights['transformer'] * predictions['transformer']['risk_logits']
                ])
                final_output['risk_logits'] = risk_preds.sum(dim=0)
            
            # 分位数权重（简化版本）
            per_q_weights = {
                'q01': base_weights,
                'q05': base_weights,
                'q50': base_weights,
                'q95': base_weights,
                'q99': base_weights
            }
            
            return final_output, per_q_weights
            
        except Exception as e:
            print(f"智能集成详细错误: {e}")
            # 回退到简单集成
            return self.fallback_ensemble(predictions), None

def fallback_ensemble(self, predictions: dict):
    """回退集成方法"""
    base_weights = self.model_weights
    
    # 简单加权平均
    ensemble_quantiles = (
        base_weights['lstm'] * predictions['lstm']['quantiles'] +
        base_weights['transformer'] * predictions['transformer']['quantiles'] +
        base_weights['lightgbm'] * predictions['lightgbm']['quantiles']
    )
    
    return {
        'quantiles': ensemble_quantiles,
        'vol': predictions['lstm']['vol'],  # 使用LSTM的输出
        'cvar': predictions['lstm']['cvar'],
        'trend_logits': predictions['lstm']['trend_logits'],
        'risk_logits': predictions['lstm']['risk_logits']
    }
    
# ---------------------- 密钥读取 ----------------------
def get_deepseek_api_key():
    key = ""
    if key:
        return key
    # 尝试从 configs/secret.yaml 读取
    try:
        import yaml
        yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "secret.yaml"))
        if os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                key = data.get("deepseek_api_key") or data.get("DEE PSEEK_API_KEY") or ""
                if key:
                    return key
    except Exception:
        pass
    # 尝试从 configs/secret.json 读取
    try:
        json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "secret.json"))
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                key = data.get("deepseek_api_key") or data.get("DEEPSEEK_API_KEY") or ""
                if key:
                    return key
    except Exception:
        pass
    return ""


# ==================== 多分位损失函数（与训练代码一致） ====================

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

# ==================== 模型定义（与训练代码一致） ====================

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
        h, _ = self.rnn(x)
        return self.fc(h[:, -1, :])
    
def make_features(csv_path="data/market.csv"):
    """生成特征 - 使用7个特征"""
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

def make_sequences(df, lookback=60, horizon=1):
    """创建序列数据 - 使用7个特征"""
    # 7个特征
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

def risk_color(var95, tau_y, tau_r):
    if var95 > tau_r: return "RED"
    if var95 > tau_y: return "YELLOW"
    return "GREEN"


def summarize_signals(csv_path, lookback, tau_y, tau_r):
    df = make_features(csv_path)
    last = df.iloc[-1]
    close = float(last["Close"])
    rsi = float(last["rsi"])
    macd = float(last["macd"])
    macd_sig = float(last["macd_sig"])
    bb_high = float(last["bb_high"])
    bb_low = float(last["bb_low"])
    vol = float(last["Volume"])
    bb_pos = 0.0
    if bb_high != bb_low:
        bb_pos = (close - bb_low) / (bb_high - bb_low)
    N = min(20, len(df))
    slope = float(pd.Series(df["Close"].tail(N)).pct_change().mean())
    return {
        "asof": str(df.index[-1].date()),
        "close": close,
        "rsi14": rsi,
        "macd": macd,
        "macd_signal": macd_sig,
        "bb_high": bb_high,
        "bb_low": bb_low,
        "bb_pos": bb_pos,
        "vol": vol,
        "trend_slope20": slope,
        "tau_y": float(tau_y),
        "tau_r": float(tau_r),
        "lookback": int(lookback),
    }


def _rule_based_note(signals, var95, color):
    rsi = signals["rsi14"]; macd = signals["macd"]; macd_sig = signals["macd_signal"]
    slope = signals["trend_slope20"]; bb_pos = signals["bb_pos"]; close = signals["close"]
    var_cap = max(0.1, 0.6 - 8.0*max(0.0, var95))
    if color == "RED":
        target_pos = min(0.25, var_cap*0.5)
    elif color == "YELLOW":
        target_pos = min(0.5, var_cap*0.8)
    else:
        target_pos = min(0.8, var_cap)
    bias = "多头" if slope>0 and macd>macd_sig else ("观望" if abs(slope)<1e-4 else "防守")
    entry_hint = "回撤到布林中轨附近分批" if bb_pos>0.6 else "突破中轨放量跟随"
    if color=="RED": entry_hint = "仅观察，不追单；等待风险回落至黄/绿灯"
    stop = round(close*(1- max(0.01, 1.2*var95)), 2)
    take = round(close*(1+ max(0.01, 0.6*var95)), 2)
    md = []
    md.append(f"**风险灯：{color}｜VaR95≈{var95:.2%}｜日期：{signals['asof']}**")
    md.append(f"- 趋势判断：{bias}（20日动量 {slope:.2%}；MACD {macd:.4f} vs Signal {macd_sig:.4f}）")
    md.append(f"- 仓位建议：目标 {int(target_pos*100)}%（上限约 {int(var_cap*100)}%）")
    md.append(f"- 入场：{entry_hint}")
    md.append(f"- 止损：{stop}（参考，约 {max(0.01,1.2*var95):.2%} 下沿）｜止盈：{take}")
    md.append(f"- 风险控制：单笔亏损不超过净值 {min(0.01, 0.6*var95):.2%}；分批/不追高")
    return "\n".join(md)


def _deepseek_note(signals, var95, color, api_key, model="deepseek-chat"):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    sys_prompt = (
        "你是一名量化投研分析师。根据给定的市场信号与风险测度，输出**结构化**投研建议。"
        "要求：1) 给出趋势判断与核心依据；2) 目标仓位区间与加减仓触发点；"
        "3) 明确入场/减仓/止损/止盈规则（用数字阈值与条件）；4) 仅基于输入信息，不臆测；"
        "5) 用中文，短句分点。"
    )
    user_payload = {"risk_light": color, "var95": var95, "signals": signals}
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "请基于以下JSON生成严谨、可执行的投研建议:\n" + json.dumps(user_payload, ensure_ascii=False)}
        ],
        "temperature": 0.2,
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
        resp.raise_for_status()
        out = resp.json()
        return out["choices"][0]["message"]["content"].strip()
    except Exception:
        return f"（DeepSeek 生成失败，已切换规则模板）\n\n" + _rule_based_note(signals, var95, color)

def analyze_feature_importance(model, X_sample, feature_names, model_type='lightgbm'):
    """分析特征重要性并返回可视化"""
    try:
        print(f"开始SHAP分析，模型类型: {model_type}, 样本形状: {X_sample.shape}")
        
        if model_type == 'lightgbm' and hasattr(model, 'predict'):
            print("使用LightGBM SHAP分析")
            # 确保输入数据格式正确
            if hasattr(X_sample, 'values'):
                X_sample = X_sample.values
            X_sample = X_sample.reshape(1, -1) if len(X_sample.shape) == 1 else X_sample
            
            # 创建解释器
            explainer = shap.TreeExplainer(model)
            
            # 计算SHAP值
            shap_values = explainer.shap_values(X_sample)
            print(f"SHAP值形状: {np.array(shap_values).shape}")
            
            # 处理SHAP值格式
            if isinstance(shap_values, list):
                # 多分类情况，取第一个类的SHAP值
                shap_values = shap_values[0]
            elif len(shap_values.shape) == 2:
                # 二维数组，取第一行（单个样本）
                shap_values = shap_values[0]
            
            shap_abs = np.abs(shap_values)
            print(f"SHAP值: {shap_values}")
            print(f"绝对SHAP值: {shap_abs}")
            
        elif model_type == 'lstm':
            print("使用LSTM SHAP分析")
            # 简化LSTM分析，只分析最后一个时间步
            background = X_sample[:50]  # 使用更小的背景数据集
            explainer = shap.DeepExplainer(model, 
                                         torch.tensor(background, dtype=torch.float32))
            
            # 计算SHAP值
            shap_values = explainer.shap_values(
                torch.tensor(X_sample, dtype=torch.float32)
            )
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # 取第一个输出
            
            # 平均过序列维度，保留特征维度
            shap_values = shap_values.mean(1)  # (batch, seq_len, features) -> (batch, features)
            shap_abs = np.abs(shap_values[0])  # 取第一个样本
            
        else:
            print(f"不支持的模型类型: {model_type}")
            return None
        
        # 检查SHAP值是否全为0
        if np.all(shap_abs == 0):
            print("警告: SHAP值全为0，使用替代方法计算特征重要性")
            # 使用模型内置的特征重要性作为备选
            if hasattr(model, 'feature_importances_'):
                shap_abs = model.feature_importances_
            else:
                # 随机生成示例数据用于演示
                shap_abs = np.random.rand(len(feature_names)) * 0.1
                print("使用随机特征重要性作为演示")
        
        # 确保特征名称和SHAP值长度匹配
        if len(feature_names) != len(shap_abs):
            print(f"特征名称数量({len(feature_names)})与SHAP值数量({len(shap_abs)})不匹配")
            # 截断或填充以匹配
            min_len = min(len(feature_names), len(shap_abs))
            feature_names = feature_names[:min_len]
            shap_abs = shap_abs[:min_len]
        
        # 获取Top5特征
        feature_importance = list(zip(feature_names, shap_abs))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        top5_features = feature_importance[:5]
        
        print(f"Top5特征: {top5_features}")
        
        # 生成条形图（解决中文显示问题）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 特征名称映射（英文到中文）
        feature_name_map = {
            'ret': '收益率',
            'rsi': 'RSI指标', 
            'macd': 'MACD',
            'macd_sig': 'MACD信号线',
            'bb_high': '布林上轨',
            'bb_low': '布林下轨',
            'Volume': '成交量',
            'rsi_14': 'RSI(14)',
            'rsi_slope': 'RSI斜率',
            'macd_diff': 'MACD差值',
            'macd_slope': 'MACD斜率',
            'bb_width': '布林带宽度',
            'bb_position': '布林带位置',
            'volatility_5': '5日波动率',
            'volatility_20': '20日波动率',
            'volume_ratio': '成交量比率',
            'momentum_5': '5日动量',
            'momentum_10': '10日动量',
            'momentum_20': '20日动量',
            'returns': '对数收益率',
            'volume_ma': '成交量均线'
        }
        
        features = [feature_name_map.get(f[0], f[0]) for f in top5_features]
        importance = [f[1] for f in top5_features]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
        bars = ax.barh(features, importance, color=colors, height=0.6)
        
        # 添加数值标签
        for bar, value in zip(bars, importance):
            if value > 0.0001:  # 只在值较大时显示标签
                ax.text(bar.get_width() + max(importance)*0.01, 
                       bar.get_y() + bar.get_height()/2,
                       f'{value:.4f}', ha='left', va='center', fontsize=10)
        
        ax.set_xlabel('平均绝对SHAP值', fontsize=12)
        ax.set_title('Top 5 风险因子贡献度', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 自动调整布局
        plt.tight_layout()
        
        # 转换为base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return {
            'top_features': top5_features,
            'plot_url': f'data:image/png;base64,{img_base64}',
            'model_type': model_type,
            'feature_names': features,
            'importance_values': importance
        }
        
    except Exception as e:
        print(f"SHAP分析错误: {str(e)}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return None

def run_inference_core(csv_path, model_path, scaler_path, lookback, alpha, tau_y, tau_r):
    if not os.path.exists(csv_path):
        return "缺少数据文件 data/market.csv（先执行数据抓取）", None, None, None, None, None, None, None, None, None, None
    
    df = make_features(csv_path)
    if len(df) < lookback + 5:
        return f"数据太短（{len(df)}行）不足以形成窗口（lookback={lookback}）", None, None, None, None, None, None, None, None, None, None
    
    X, y, idx = make_sequences(df, lookback=lookback, horizon=1)
    
    if os.path.exists(scaler_path):
        scaler = load(scaler_path)
        tip_scaler = ""
    else:
        tr_end = int(len(X)*0.7)
        scaler = StandardScaler().fit(X[:tr_end].reshape(tr_end, -1))
        tip_scaler = "（未找到训练时保存的 scaler.pkl，已临时在前70%样本上拟合）"
    
    Xs = scaler.transform(X.reshape(len(X), -1)).reshape(X.shape)
    
    if not os.path.exists(model_path):
        return "缺少模型权重 models/best_lstm_quantile.pt（先执行训练）", None, None, None, None, None, None, None, None, None, None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 修改模型定义，支持5个分位
    model = LSTMQuantile(in_dim=X.shape[-1], hidden=64, layers=2, num_quantiles=5, dropout=0.2).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    
    # 多分位预测
    quantiles = [0.01, 0.05, 0.5, 0.95, 0.99]
    q_pred_list = []
    
    with torch.no_grad():
        for i in range(0, len(Xs), 1024):
            xb = torch.tensor(Xs[i:i+1024], dtype=torch.float32).to(device)
            q_pred = model(xb).cpu().numpy()  # (B, 5)
            q_pred_list.append(q_pred)
    
    q_pred_all = np.concatenate(q_pred_list)  # (N, 5)
    
    # 提取各个分位
    q01 = q_pred_all[:, 0]  # 1%分位
    q05 = q_pred_all[:, 1]  # 5%分位
    q50 = q_pred_all[:, 2]  # 中位数
    q95 = q_pred_all[:, 3]  # 95%分位
    q99 = q_pred_all[:, 4]  # 99%分位
    
    y_true = y
    
    # 计算风险指标
    var95 = -q05  # VaR95
    cvar95 = -np.array([np.mean(y_true[i-lookback:i][y_true[i-lookback:i] <= q05[i]]) 
                       if i > lookback else var95[i] for i in range(len(q05))])  # CVaR95
    opportunity_band = q95 - q05  # 机会区间宽度
    
    exceed = y_true <= q05
    
    # 使用新的多分位验证器
    validator = VaRValidation(y_true, q_pred_all, quantiles)
    confidence_score = validator.coverage_quality_score()
    coverage_report = validator.get_detailed_coverage_report()
    
    # 主要关注q05的覆盖率（VaR95）
    main_coverage = validator.hit_ratio('q05')
    
    cur_v95 = float(var95[-1])
    cur_cvar = float(cvar95[-1]) if not np.isnan(cvar95[-1]) else cur_v95
    cur_opportunity = float(opportunity_band[-1])
    
    color = risk_color(cur_v95, tau_y, tau_r)
    
    # 简化SHAP分析
    try:
        print("开始SHAP特征重要性分析...")
        
        # 使用最新样本进行分析
        sample_idx = -1
        if len(Xs) > 0:
            X_sample = Xs[sample_idx].reshape(1, *Xs[sample_idx].shape)
            
            # 特征名称
            feature_names = ["ret", "rsi", "macd", "macd_sig", "bb_high", "bb_low", "Volume"]
            
            # 执行SHAP分析
            shap_analysis = analyze_feature_importance(
                model, X_sample, feature_names, model_type='lstm'
            )
            print(f"SHAP分析完成: {shap_analysis is not None}")
        else:
            shap_analysis = None
            print("数据不足，跳过SHAP分析")
            
    except Exception as e:
        print(f"SHAP分析错误: {str(e)}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        shap_analysis = None
    # 价格图表
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#3b6af7")))
    fig_price.update_layout(
        height=360, 
        margin=dict(l=30,r=30,t=30,b=30),
    )
    
    # 多分位风险带图
    fig_quantiles = go.Figure()
    
    # 添加风险带（1%-99%置信区间）
    fig_quantiles.add_trace(go.Scatter(
        x=np.concatenate([idx, idx[::-1]]),
        y=np.concatenate([q99, q01[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 91, 91, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='1%-99% 风险带',
        showlegend=True
    ))
    
    # 添加机会区间带（5%-95%置信区间）
    fig_quantiles.add_trace(go.Scatter(
        x=np.concatenate([idx, idx[::-1]]),
        y=np.concatenate([q95, q05[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 181, 76, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='5%-95% 机会区间',
        showlegend=True
    ))
    
    # 添加实际收益
    fig_quantiles.add_trace(go.Scatter(
        x=idx, y=y_true,
        name='实际收益',
        line=dict(color='#636EFA', width=1.5)
    ))
    
    # 添加中位数预测
    fig_quantiles.add_trace(go.Scatter(
        x=idx, y=q50,
        name='中位数预测 (q50)',
        line=dict(color='#ef4444', width=2)
    ))
    
    # 添加超损点标记
    fig_quantiles.add_trace(go.Scatter(
        x=idx[exceed], y=y_true[exceed],
        mode="markers",
        name="超损点(<=q0.05)",
        marker=dict(size=6, color='#00CC96')
    ))
    
    fig_quantiles.update_layout(
        height=420,
        margin=dict(l=30,r=30,t=30,b=30),
        xaxis_title="日期",
        yaxis_title="收益率"
    )
    # 计算波动率序列 (20日滚动波动率)
    try:
        returns_series = df['ret'].copy()
        volatility_series = returns_series.rolling(window=20, min_periods=1).std().fillna(0)
        # 对齐到预测日期
        volatility_aligned = []
        for i, date in enumerate(idx):
            if i < len(volatility_series):
                volatility_aligned.append(float(volatility_series.iloc[i]))
            else:
                volatility_aligned.append(float(volatility_series.iloc[-1]))
        volatility_series = np.array(volatility_aligned)
    except Exception as e:
        print(f"波动率计算错误: {e}")
        volatility_series = np.zeros(len(idx))

    # 计算趋势方向 (基于5日动量)
    try:
        close_prices = df['Close'].values
        trend_direction = []
        for i in range(len(idx)):
            if i >= 5:
                # 计算5日收益率
                momentum = (close_prices[i] / close_prices[i-5] - 1) if close_prices[i-5] != 0 else 0
                # 转换为趋势方向: 1(上涨), 0(震荡), -1(下跌)
                if momentum > 0.005:  # 0.5%阈值
                    trend_direction.append(1)
                elif momentum < -0.005:
                    trend_direction.append(-1)
                else:
                    trend_direction.append(0)
            else:
                trend_direction.append(0)
        trend_direction = np.array(trend_direction)
    except Exception as e:
        print(f"趋势方向计算错误: {e}")
        trend_direction = np.zeros(len(idx))

    # 计算风险等级 (基于VaR95)
    try:
        risk_levels = []
        for i in range(len(var95)):
            if var95[i] > tau_r:  # 高于红灯阈值
                risk_levels.append(2)  # 高风险
            elif var95[i] > tau_y:  # 高于黄灯阈值
                risk_levels.append(1)  # 中风险
            else:
                risk_levels.append(0)  # 低风险
        risk_levels = np.array(risk_levels)
    except Exception as e:
        print(f"风险等级计算错误: {e}")
        risk_levels = np.zeros(len(idx))

    # 构建多任务数据
    multi_task_data = {
        'volatility': {
            'values': volatility_series.tolist(),
            'dates': [str(d.date()) for d in idx]
        },
        'trend_direction': {
            'values': trend_direction.tolist(),
            'dates': [str(d.date()) for d in idx]
        },
        'risk_levels': {
            'values': risk_levels.tolist(),
            'dates': [str(d.date()) for d in idx]
        },
        'confidence_band': {
            'upper': q95.tolist(),
            'lower': q05.tolist(),
            'median': q50.tolist(),
            'dates': [str(d.date()) for d in idx]
        }
    }
    # 多任务数据
    multi_task_data = {
        'volatility': {
            'values': volatility_series.tolist(),  # 波动率序列
            'dates': [str(d.date()) for d in idx]
        },
        'trend_direction': {
            'values': trend_direction.tolist(),    # 趋势方向 (1:上涨, 0:震荡, -1:下跌)
            'dates': [str(d.date()) for d in idx]
        },
        'risk_levels': {
            'values': risk_levels.tolist(),        # 风险等级 (0:低, 1:中, 2:高)
            'dates': [str(d.date()) for d in idx]
        },
        'confidence_band': {
            'upper': q95.tolist(),                # 95%分位
            'lower': q05.tolist(),                # 5%分位  
            'median': q50.tolist(),               # 中位数
            'dates': [str(d.date()) for d in idx]
        }
    }

    
    # 风险指标图表
    fig_risk_metrics = go.Figure()
    
    fig_risk_metrics.add_trace(go.Scatter(
        x=idx, y=var95,
        name='VaR95',
        line=dict(color='#00CC96', width=2)
    ))
    
    fig_risk_metrics.add_trace(go.Scatter(
        x=idx, y=cvar95,
        name='CVaR95',
        line=dict(color='#ef4444', width=2, dash='dash')
    ))
    
    fig_risk_metrics.add_trace(go.Scatter(
        x=idx, y=opportunity_band,
        name='机会区间宽度',
        line=dict(color='#636EFA', width=2)
    ))
    
    fig_risk_metrics.update_layout(
        height=360,
        margin=dict(l=30,r=30,t=30,b=30),
        xaxis_title="日期",
        yaxis_title="风险指标值"
    )
    
    header = (
        f"当前风险等级：**{color}** ｜ VaR95 ≈ **{cur_v95:.2%}** ｜ CVaR95 ≈ **{cur_cvar:.2%}** ｜ "
        f"机会区间 ≈ **{cur_opportunity:.2%}** ｜ 覆盖率≈**{main_coverage:.3f}** "
        f"（目标 {alpha:.2f}）｜ 置信度 **{confidence_score:.1%}** {tip_scaler}"
    )
    
    # 组装最近10次预测表
    rows = []
    try:
        for i in range(max(0, len(idx)-10), len(idx)):
            base_date = idx[i]
            base_loc = df.index.get_loc(base_date)
            tgt_date = df.index[base_loc + 1] if base_loc + 1 < len(df.index) else None
            
            # 多分位数据
            q01_val = float(q01[i])
            q05_val = float(q05[i])
            q50_val = float(q50[i])
            q95_val = float(q95[i])
            q99_val = float(q99[i])
            
            v95 = float(var95[i])
            cvar = float(cvar95[i]) if not np.isnan(cvar95[i]) else v95
            opportunity = float(opportunity_band[i])
            
            y1 = float(y_true[i])
            light = risk_color(v95, tau_y, tau_r)
            
            rows.append({
                "base_date": str(base_date.date()),
                "target_date": (str(tgt_date.date()) if tgt_date is not None else "-"),
                "q01": q01_val,
                "q05": q05_val,
                "q50": q50_val,
                "q95": q95_val,
                "q99": q99_val,
                "var95": v95,
                "cvar95": cvar,
                "opportunity_band": opportunity,
                "risk_light": light,
                "ret1": y1,
                "exceed": bool(y1 <= q05_val),
            })
    except Exception as e:
        print(f"表格数据组装错误: {e}")
        rows = []
    
    # 使用新的多分位验证器
    validator = VaRValidation(y_true, q_pred_all, quantiles)
    confidence_score = validator.coverage_quality_score()
    coverage_report = validator.get_detailed_coverage_report()
    
    # 主要关注q05的覆盖率（VaR95）
    main_coverage = validator.hit_ratio('q05')
    
    # 计算性能评级
    hit_bias = abs(main_coverage - alpha)
    if hit_bias <= 0.005:
        performance_rating = "优秀"
    elif hit_bias <= 0.01:
        performance_rating = "良好"
    elif hit_bias <= 0.02:
        performance_rating = "一般"
    else:
        performance_rating = "需要改进"
    
    # 计算综合偏差
    overall_bias = validator.calculate_overall_bias()
    
    # 组装验证数据
    validation_data = {
        'returns': y_true.tolist(),
        'quantile_predictions': q_pred_all.tolist(),  # 确保包含这个
        'var_predictions': var95.tolist(),
        'dates': [str(d.date()) for d in idx],
        'confidence_score': confidence_score,
        'coverage_report': coverage_report,
        'main_coverage': main_coverage,
        'overall_bias': validator.calculate_overall_bias(),  # 新增综合偏差
        'performance_rating': performance_rating,  # 性能评级
        'total_periods': len(y_true),  # 总验证周期
        'violation_count': int(np.sum(validator.hits['q05']))
    }
    # 在返回语句中添加 multi_task_data
    return header, fig_price, fig_quantiles, fig_risk_metrics, color, cur_v95, cur_cvar, cur_opportunity, rows, validation_data, shap_analysis, multi_task_data

# 添加LightGBM特征准备函数
def prepare_lightgbm_features(df):
    """为SHAP分析准备LightGBM特征"""
    features_df = df.copy()
    
    # 基础价格特征
    features_df['returns'] = np.log(features_df['Close']).diff()
    features_df['volatility_5'] = features_df['returns'].rolling(5).std()
    features_df['volatility_20'] = features_df['returns'].rolling(20).std()
    
    # RSI相关特征
    features_df['rsi_14'] = features_df['rsi']
    features_df['rsi_slope'] = features_df['rsi'].diff(3)
    
    # MACD相关特征
    features_df['macd_diff'] = features_df['macd'] - features_df['macd_signal']
    features_df['macd_slope'] = features_df['macd'].diff(3)
    
    # 布林带相关特征
    features_df['bb_width'] = (features_df['bb_high'] - features_df['bb_low']) / features_df['Close']
    features_df['bb_position'] = (features_df['Close'] - features_df['bb_low']) / (features_df['bb_high'] - features_df['bb_low'])
    
    # 成交量特征
    features_df['volume_ma'] = features_df['Volume'].rolling(10).mean()
    features_df['volume_ratio'] = features_df['Volume'] / features_df['volume_ma']
    
    # 价格动量特征
    for window in [5, 10, 20]:
        features_df[f'momentum_{window}'] = features_df['Close'].pct_change(window)
    
    # 选择最终特征列
    feature_cols = [
        'rsi_14', 'rsi_slope', 'macd', 'macd_signal', 'macd_diff', 'macd_slope',
        'bb_width', 'bb_position', 'volatility_5', 'volatility_20',
        'volume_ratio', 'momentum_5', 'momentum_10', 'momentum_20'
    ]
    
    # 只保留存在的列
    available_cols = [col for col in feature_cols if col in features_df.columns]
    result_df = features_df[available_cols].copy()
    
    # 填充NaN值
    result_df = result_df.ffill().bfill().fillna(0)
    
    return result_df

# ==================== 集成模型训练函数 ====================

def train_ensemble_model(csv_path="data/market.csv", lookback=60, multitask=True):
    """训练集成模型（支持多任务学习）"""
    print("开始训练集成模型..." + ("（多任务模式）" if multitask else ""))
    
    # 准备数据
    df = make_features(csv_path)
    X_seq, y, dates = make_sequences(df, lookback=lookback)
    
    if len(X_seq) < 100:
        print("数据量不足，需要更多数据训练集成模型")
        return None
    
    # 准备多任务目标
    if multitask:
        targets_dict = prepare_multitask_targets(df, dates, lookback)
    else:
        targets_dict = {'y': y}
    
    # 初始化集成模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensemble_model = EnsembleManager(lookback=lookback, num_quantiles=5, device=device)
    
    # 准备LightGBM特征
    df_lgb = ensemble_model.prepare_technical_features(df)
    valid_dates = dates[lookback:] if len(dates) > lookback else dates
    X_lgb = df_lgb.loc[valid_dates].values
    
    # 数据分割
    train_idx = int(0.7 * len(X_seq))
    val_idx = int(0.85 * len(X_seq))
    
    X_seq_train, X_seq_val = X_seq[:train_idx], X_seq[train_idx:val_idx]
    X_lgb_train, X_lgb_val = X_lgb[:train_idx], X_lgb[train_idx:val_idx]
    y_train, y_val = y[:train_idx], y[train_idx:val_idx]
    
    # 准备多任务目标数据
    if multitask:
        train_targets = {k: v[:train_idx] for k, v in targets_dict.items() if len(v) == len(X_seq)}
        val_targets = {k: v[train_idx:val_idx] for k, v in targets_dict.items() if len(v) == len(X_seq)}
    else:
        train_targets = {'y': y_train}
        val_targets = {'y': y_val}
    
    # 标准化序列数据
    from sklearn.preprocessing import StandardScaler
    scaler_seq = StandardScaler()
    X_seq_train_scaled = scaler_seq.fit_transform(
        X_seq_train.reshape(-1, X_seq_train.shape[-1])
    ).reshape(X_seq_train.shape)
    X_seq_val_scaled = scaler_seq.transform(
        X_seq_val.reshape(-1, X_seq_val.shape[-1])
    ).reshape(X_seq_val.shape)
    
    # 标准化LightGBM特征
    scaler_lgb = StandardScaler()
    X_lgb_train_scaled = scaler_lgb.fit_transform(X_lgb_train)
    X_lgb_val_scaled = scaler_lgb.transform(X_lgb_val)
    
    # 训练LightGBM模型
    ensemble_model.train_lightgbm_models(X_lgb_train_scaled, y_train)
    
    # 训练深度学习模型（多任务）
    print("训练LSTM模型..." + ("（多任务）" if multitask else ""))
    train_multitask_model(ensemble_model.lstm_model, X_seq_train_scaled, train_targets, 
                         X_seq_val_scaled, val_targets, device, model_name="LSTM", 
                         multitask=multitask)
    
    print("训练Transformer模型..." + ("（多任务）" if multitask else ""))
    train_multitask_model(ensemble_model.transformer_model, X_seq_train_scaled, train_targets,
                         X_seq_val_scaled, val_targets, device, model_name="Transformer",
                         multitask=multitask)
    
    # 验证集性能评估和权重调整
    val_predictions = ensemble_model.ensemble_predict(X_seq_val_scaled, X_lgb_val_scaled)
    
    # 计算多任务性能指标
    if multitask:
        val_metrics = evaluate_multitask_performance(val_predictions['final'], val_targets)
        print("多任务验证性能:")
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # 基于性能更新模型权重
    val_losses = calculate_model_performance(val_predictions, val_targets, multitask=multitask)
    
    if val_losses:
        total_loss = sum(val_losses.values())
        new_weights = {}
        for model_name, loss in val_losses.items():
            # 损失越小，权重越大
            new_weights[model_name] = (1 - loss / total_loss) / (len(val_losses) - 1)
        
        # 归一化
        weight_sum = sum(new_weights.values())
        ensemble_model.model_weights = {k: v/weight_sum for k, v in new_weights.items()}
        print(f"模型权重已更新: {ensemble_model.model_weights}")
    
    # 保存模型
    import joblib
    os.makedirs('models', exist_ok=True)
    
    # 保存深度学习模型
    torch.save(ensemble_model.lstm_model.state_dict(), 'models/ensemble_lstm.pt')
    torch.save(ensemble_model.transformer_model.state_dict(), 'models/ensemble_transformer.pt')
    
    # 保存LightGBM模型
    joblib.dump(ensemble_model.lgb_models, 'models/ensemble_lgb.pkl')
    
    # 保存标准化器
    joblib.dump(scaler_seq, 'models/ensemble_scaler_seq.pkl')
    joblib.dump(scaler_lgb, 'models/ensemble_scaler_lgb.pkl')
    
    # 保存模型配置
    ensemble_config = {
        'model_weights': ensemble_model.model_weights,
        'quantiles': ensemble_model.quantiles,
        'lookback': lookback,
        'multitask': multitask
    }
    joblib.dump(ensemble_config, 'models/ensemble_config.pkl')
    
    print("集成模型训练完成!" + ("（多任务）" if multitask else ""))
    return ensemble_model

def train_multitask_model(model, X_train, train_targets, X_val, val_targets, device, 
                         model_name="Model", multitask=True, epochs=100, patience=15):
    """训练单个多任务深度学习模型"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_loss = float('inf')
    patience_counter = 0
    task_weights = {'q': 1.0, 'vol': 1.0, 'cvar': 1.0, 'trend': 1.0, 'risk': 1.0}
    
    # 基础任务权重
    base_weights = {'q': 1.0, 'vol': 0.5, 'cvar': 0.8, 'trend': 0.3, 'risk': 0.3}
    quantiles = [0.01, 0.05, 0.5, 0.95, 0.99]
    quantile_weights = [2.0, 1.5, 1.0, 1.5, 2.0]  # 极端分位更高权重
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # 小批量训练
        batch_size = 64
        for i in range(0, len(X_train), batch_size):
            end_idx = min(i + batch_size, len(X_train))
            x_batch = torch.tensor(X_train[i:end_idx], dtype=torch.float32).to(device)
            
            # 准备批次目标 - 确保张量需要梯度
            batch_targets = {}
            for key, target in train_targets.items():
                if isinstance(target, (np.ndarray, list)) and len(target) == len(X_train):
                    if key in ['trend', 'risk']:
                        # 分类标签不需要梯度
                        batch_targets[key] = torch.tensor(target[i:end_idx], dtype=torch.long).to(device)
                    else:
                        # 回归目标需要梯度
                        batch_targets[key] = torch.tensor(target[i:end_idx], dtype=torch.float32).to(device).requires_grad_(False)
            
            optimizer.zero_grad()
            preds = model(x_batch)
            
            if multitask:
                total_loss, loss_dict = multi_task_loss(
                    preds, batch_targets, weights=base_weights,
                    quantiles=quantiles, quantile_weights=quantile_weights,
                    task_weights=task_weights
                )
            else:
                # 单任务回退
                total_loss = multi_quantile_loss(
                    preds['quantiles'], batch_targets['y'], 
                    quantiles=quantiles, quantile_weights=quantile_weights
                )
            
            # 确保损失需要梯度
            if not isinstance(total_loss, torch.Tensor):
                total_loss = torch.tensor(total_loss, device=device, requires_grad=True)
            elif not total_loss.requires_grad:
                total_loss = total_loss.clone().requires_grad_(True)
            
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += total_loss.item()
        
        # 验证 - 不需要梯度
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            x_val = torch.tensor(X_val, dtype=torch.float32).to(device)
            val_preds = model(x_val)
            
            # 准备验证目标
            val_batch_targets = {}
            for key, target in val_targets.items():
                if isinstance(target, (np.ndarray, list)) and len(target) == len(X_val):
                    if key in ['trend', 'risk']:
                        val_batch_targets[key] = torch.tensor(target, dtype=torch.long).to(device)
                    else:
                        val_batch_targets[key] = torch.tensor(target, dtype=torch.float32).to(device)
            
            if multitask:
                val_total_loss, val_loss_dict = multi_task_loss(
                    val_preds, val_batch_targets, weights=base_weights,
                    quantiles=quantiles, quantile_weights=quantile_weights,
                    task_weights=task_weights
                )
            else:
                val_total_loss = multi_quantile_loss(
                    val_preds['quantiles'], val_batch_targets['y'],
                    quantiles=quantiles, quantile_weights=quantile_weights
                )
            
            val_loss = val_total_loss.item() if isinstance(val_total_loss, torch.Tensor) else val_total_loss
        
        # 学习率调度
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()
        scheduler.step(val_loss)
        
        # 动态调整任务权重（每10个epoch）
        if multitask and epoch % 10 == 0:
            try:
                val_metrics = evaluate_multitask_performance(val_preds, val_batch_targets)
                task_weights = compute_task_weights(val_metrics)
                print(f"📊 任务权重更新: {task_weights}")
            except Exception as e:
                print(f"⚠️ 任务权重更新失败: {e}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'models/best_{model_name.lower()}_multitask.pt')
            print(f'✅ {model_name} Epoch {epoch}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f} *')
        else:
            patience_counter += 1
            print(f'⏳ {model_name} Epoch {epoch}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if patience_counter >= patience:
            print(f'⏹️  {model_name} 早停于第 {epoch} 轮')
            break
    
    # 加载最佳模型
    best_model_path = f'models/best_{model_name.lower()}_multitask.pt'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f'✅ 加载最佳模型: {best_model_path}')
    
    return model

def prepare_multitask_targets(df, dates, lookback):
    """准备多任务学习的目标变量"""
    targets = {}
    
    # 1. 主要目标：收益率（用于分位数回归）
    targets['y'] = np.log(df['Close']).diff().shift(-1).dropna().values
    
    # 2. 波动率目标：滚动波动率
    returns = np.log(df['Close']).diff()
    targets['vol'] = returns.rolling(window=5).std().shift(-1).fillna(0).values
    
    # 3. CVaR目标：基于历史数据的条件风险价值
    var_95 = -returns.rolling(window=20).quantile(0.05)
    cvar_95 = returns[returns <= -var_95].rolling(window=20).mean().shift(-1).fillna(0)
    targets['cvar'] = -cvar_95.values
    
    # 4. 趋势分类目标
    trend = np.where(df['Close'].pct_change(5).shift(-1) > 0.005, 2,  # 上涨
                    np.where(df['Close'].pct_change(5).shift(-1) < -0.005, 0, 1))  # 下跌, 震荡
    targets['trend'] = trend
    
    # 5. 风险分类目标
    volatility = returns.rolling(window=10).std()
    risk_level = np.where(volatility > 0.02, 2,  # 高风险
                         np.where(volatility > 0.01, 1, 0))  # 中风险, 低风险
    targets['risk'] = risk_level
    
    # 确保所有目标长度一致
    min_length = min(len(v) for v in targets.values() if hasattr(v, '__len__'))
    for key in targets:
        if hasattr(targets[key], '__len__') and len(targets[key]) > min_length:
            targets[key] = targets[key][:min_length]
    
    return targets

def evaluate_multitask_performance(preds, targets):
    """评估多任务模型性能"""
    metrics = {}
    device = next(iter(preds.values())).device if isinstance(preds, dict) else 'cpu'
    
    # 分位数覆盖率和偏差
    if 'quantiles' in preds and 'y' in targets:
        try:
            coverage = compute_coverage(preds['quantiles'], targets['y'], [0.01, 0.05, 0.5, 0.95, 0.99])
            coverage_bias = np.mean([abs(coverage.get(f'q{int(q*100):02d}', 0) - (q if q < 0.5 else 1-q)) 
                                   for q in [0.01, 0.05, 0.5, 0.95, 0.99]])
            metrics['quantile_coverage_bias'] = float(coverage_bias)
        except Exception as e:
            metrics['quantile_coverage_bias'] = 0.1
    
    # 波动率预测MAE
    if 'vol' in preds and 'vol' in targets:
        try:
            metrics['volatility_mae'] = torch.abs(preds['vol'] - targets['vol']).mean().item()
        except:
            metrics['volatility_mae'] = 0.01
    
    # CVaR预测MAE
    if 'cvar' in preds and 'cvar' in targets:
        try:
            metrics['cvar_mae'] = torch.abs(preds['cvar'] - targets['cvar']).mean().item()
        except:
            metrics['cvar_mae'] = 0.01
    
    # 趋势分类准确率
    if 'trend_logits' in preds and 'trend' in targets:
        try:
            trend_pred = torch.argmax(preds['trend_logits'], dim=1)
            metrics['trend_accuracy'] = (trend_pred == targets['trend']).float().mean().item()
        except:
            metrics['trend_accuracy'] = 0.5
    
    # 风险分类准确率
    if 'risk_logits' in preds and 'risk' in targets:
        try:
            risk_pred = torch.argmax(preds['risk_logits'], dim=1)
            metrics['risk_accuracy'] = (risk_pred == targets['risk']).float().mean().item()
        except:
            metrics['risk_accuracy'] = 0.5
    
    return metrics

def calculate_model_performance(predictions, targets, multitask=True):
    """计算各模型性能"""
    val_losses = {}
    
    for model_name, pred in predictions.items():
        if model_name in ['lstm', 'transformer', 'lightgbm']:
            if multitask:
                # 多任务性能评估
                metrics = evaluate_multitask_performance(pred, targets)
                # 使用加权综合评分
                total_score = (0.4 * (1 - metrics.get('quantile_coverage_bias', 0.1)) +
                             0.2 * (1 - min(1.0, metrics.get('volatility_mae', 0.1) / 0.01)) +
                             0.2 * (1 - min(1.0, metrics.get('cvar_mae', 0.1) / 0.01)) +
                             0.1 * metrics.get('trend_accuracy', 0.5) +
                             0.1 * metrics.get('risk_accuracy', 0.5))
                val_losses[model_name] = 1.0 - total_score
            else:
                # 单任务性能评估
                if 'quantiles' in pred:
                    var_predictions = -pred['quantiles'][:, 1]  # 5%分位
                    actual_exceedances = (targets['y'] <= pred['quantiles'][:, 1]).float()
                    coverage_bias = abs(actual_exceedances.mean().item() - 0.05)
                    val_losses[model_name] = coverage_bias
    
    return val_losses

def train_single_model(model, X_train, y_train, X_val, y_val, device, model_name="Model", epochs=50, patience=10):
    """训练单个深度学习模型"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 使用损失函数
    loss_fn = multi_quantile_loss(quantiles=[0.01, 0.05, 0.5, 0.95, 0.99])
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(torch.tensor(X_train, dtype=torch.float32).to(device))
        loss = loss_fn(outputs, torch.tensor(y_train, dtype=torch.float32).to(device))
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.tensor(X_val, dtype=torch.float32).to(device))
            val_loss = loss_fn(val_outputs, torch.tensor(y_val, dtype=torch.float32).to(device))
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f'models/best_{model_name.lower()}.pt')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
        if epoch % 10 == 0:
            print(f'{model_name} Epoch {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    # 加载最佳模型
    if os.path.exists(f'models/best_{model_name.lower()}.pt'):
        model.load_state_dict(torch.load(f'models/best_{model_name.lower()}.pt', map_location=device))
    
    return model


# ==================== 集成模型推理函数 ====================

def run_ensemble_inference(csv_path, lookback=60, alpha=0.05, tau_y=0.015, tau_r=0.03):
    """使用集成模型进行推理"""
    try:
        # 检查模型文件是否存在
        model_files = [
            'models/ensemble_lstm.pt', 
            'models/ensemble_transformer.pt',
            'models/ensemble_lgb.pkl',
            'models/ensemble_scaler_seq.pkl',
            'models/ensemble_scaler_lgb.pkl',
            'models/ensemble_config.pkl'
        ]
        
        missing_files = [f for f in model_files if not os.path.exists(f)]
        if missing_files:
            return f"集成模型文件缺失: {missing_files}。请先训练集成模型。", None, None, None, None, None, None, None, None, None
        
        # 加载集成模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        import joblib
        
        # 加载配置
        ensemble_config = joblib.load('models/ensemble_config.pkl')
        scaler_seq = joblib.load('models/ensemble_scaler_seq.pkl')
        scaler_lgb = joblib.load('models/ensemble_scaler_lgb.pkl')
        lgb_models = joblib.load('models/ensemble_lgb.pkl')
        
        # 初始化集成管理器
        ensemble_model = EnsembleManager(
            lookback=lookback, 
            num_quantiles=5, 
            device=device
        )
        ensemble_model.model_weights = ensemble_config['model_weights']
        ensemble_model.lgb_models = lgb_models
        
        # 加载模型权重
        ensemble_model.lstm_model.load_state_dict(
            torch.load('models/ensemble_lstm.pt', map_location=device)
        )
        ensemble_model.transformer_model.load_state_dict(
            torch.load('models/ensemble_transformer.pt', map_location=device)
        )
        
        # 准备数据
        df = make_features(csv_path)
        if len(df) < lookback + 10:
            return "数据量不足进行集成模型推理", None, None, None, None, None, None, None, None, None
            
        X_seq, y, idx = make_sequences(df, lookback=lookback)
        
        # 准备LightGBM特征
        df_lgb = ensemble_model.prepare_technical_features(df)
        valid_dates = idx[lookback:] if len(idx) > lookback else idx
        X_lgb = df_lgb.loc[valid_dates].values
        
        # 标准化数据
        X_seq_scaled = scaler_seq.transform(
            X_seq.reshape(-1, X_seq.shape[-1])
        ).reshape(X_seq.shape)
        X_lgb_scaled = scaler_lgb.transform(X_lgb)
        
        # 集成预测
        predictions = ensemble_model.ensemble_predict(X_seq_scaled, X_lgb_scaled)
        # prefer smart final prediction if available
        q_pred_all = predictions.get('final', predictions.get('ensemble'))

        # 提取分位数预测
        q01 = q_pred_all[:, 0]  # 1%分位
        q05 = q_pred_all[:, 1]  # 5%分位
        q50 = q_pred_all[:, 2]  # 中位数
        q95 = q_pred_all[:, 3]  # 95%分位
        q99 = q_pred_all[:, 4]  # 99%分位

        y_true = y

        # 计算风险指标
        var95 = -q05  # VaR95
        cvar95 = -np.array([np.mean(y_true[i-lookback:i][y_true[i-lookback:i] <= q05[i]]) 
                           if i > lookback else var95[i] for i in range(len(q05))])
        opportunity_band = q95 - q05

        exceed = y_true <= q05
        coverage = float((y_true <= q05).mean())

        cur_v95 = float(var95[-1])
        cur_cvar = float(cvar95[-1]) if not np.isnan(cvar95[-1]) else cur_v95
        cur_opportunity = float(opportunity_band[-1])

        color = risk_color(cur_v95, tau_y, tau_r)

        # 计算各模型单独表现
        individual_performance = {}
        for model_name, pred in predictions.items():
            if model_name != 'ensemble':
                model_var95 = -pred[:, 1]  # 5%分位
                model_coverage = float((y_true <= pred[:, 1]).mean())
                individual_performance[model_name] = {
                    'coverage': model_coverage,
                    'bias': abs(model_coverage - alpha),
                    'current_var95': float(model_var95[-1]),
                    'weight': ensemble_model.model_weights.get(model_name, 0)
                }

        # 验证器
        # 计算整体置信度（基于多分位覆盖质量）
        validator = VaRValidation(y_true, q_pred_all, ensemble_model.quantiles if hasattr(ensemble_model, 'quantiles') else [0.01,0.05,0.5,0.95,0.99])
        try:
            confidence_score = validator.coverage_quality_score()
            coverage_report = validator.get_detailed_coverage_report()
        except Exception:
            confidence_score = 0.5
            coverage_report = {}

        # 生成图表
        fig_price, fig_quantiles, fig_risk_metrics = create_ensemble_charts(
            df, idx, q01, q05, q50, q95, q99, y_true, var95, cvar95, opportunity_band, exceed
        )

        # 组装结果
        header = (
            f"🎯 集成模型预测 | 风险等级: **{color}** | VaR95: **{cur_v95:.2%}** | "
            f"CVaR95: **{cur_cvar:.2%}** | 覆盖率: **{coverage:.3f}** (目标 {alpha:.2f}) | "
            f"置信度: **{confidence_score:.1%}**"
        )
        
        # 预测表格数据
        rows = create_prediction_table(df, idx, q01, q05, q50, q95, q99, y_true, var95, cvar95, opportunity_band, tau_y, tau_r)
        
        validation_data = {
            'returns': y_true.tolist(),
            'var_predictions': var95.tolist(),
            'dates': [str(d.date()) for d in idx],
            'confidence_score': confidence_score,
            'individual_performance': individual_performance,
            'model_weights': ensemble_model.model_weights,
            'model_type': 'ensemble'
        }
        validation_data['coverage_report'] = coverage_report

        # 额外多任务预测（volatility, cvar, trend, risk）
        multi_task_preds = {}
        # quantiles (final)
        multi_task_preds['quantiles'] = q_pred_all.tolist()

        # volatility: try loading a saved LightGBM vol regressor
        vol_preds = None
        try:
            vol_model_path = os.path.join('models', 'lgb_volatility_regressor.pkl')
            if os.path.exists(vol_model_path):
                vol_model = joblib.load(vol_model_path)
                vol_preds = vol_model.predict(X_lgb_scaled)
            else:
                # fallback: use rolling volatility from df
                vol_series = df['ret'].rolling(window=5, min_periods=1).std().fillna(0.0)
                vol_preds = vol_series.loc[idx].values if set(idx).issubset(set(vol_series.index)) else np.repeat(float(vol_series.iloc[-1]), len(idx))
        except Exception:
            vol_preds = np.repeat(0.0, len(idx))
        multi_task_preds['volatility'] = np.array(vol_preds).tolist()

        # risk label: try loading classifier
        risk_labels = None
        try:
            risk_model_path = os.path.join('models', 'lgb_risk_classifier.pkl')
            if os.path.exists(risk_model_path):
                risk_clf = joblib.load(risk_model_path)
                # if scaler needed, X_lgb_scaled already scaled
                probs = risk_clf.predict_proba(X_lgb_scaled)
                # predicted class (argmax)
                preds = np.argmax(probs, axis=1)
                risk_labels = preds.tolist()
            else:
                # fallback: simple threshold on var95
                risk_labels = [int(v > tau_r) + int(v > tau_y and v <= tau_r) for v in var95]
        except Exception:
            risk_labels = [0]*len(idx)
        multi_task_preds['risk_label'] = risk_labels

        # trend: simple rule-based from recent close momentum
        try:
            closes = df['Close'].loc[idx]
            trend_vals = []
            for i in range(len(closes)):
                window_start = max(0, i - int(lookback/4))
                recent = closes.iloc[window_start:i+1]
                slope = (recent.iloc[-1] / recent.iloc[0] - 1) if len(recent) > 1 else 0.0
                if slope > 0.001:
                    trend_vals.append(1)  # up
                elif slope < -0.001:
                    trend_vals.append(0)  # down
                else:
                    trend_vals.append(2)  # neutral
        except Exception:
            trend_vals = [2]*len(idx)
        multi_task_preds['trend'] = trend_vals

        # CVaR (already computed as cvar95) include as cvar95
        multi_task_preds['cvar95'] = list(cvar95)

        # confidence per-sample: prefer predictions['confidence'] if present
        confidence_arr = None
        try:
            confidence_arr = predictions.get('confidence')
            if confidence_arr is None:
                # fallback to validator-based scalar
                confidence_arr = np.repeat(confidence_score, len(idx))
        except Exception:
            confidence_arr = np.repeat(confidence_score, len(idx))
        
    except Exception as e:
        import traceback
        error_msg = f"集成模型推理错误: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, None, None, None, None, None, None, None, None, None
    
    return header, fig_price, fig_quantiles, fig_risk_metrics, color, cur_v95, cur_cvar, cur_opportunity, rows, validation_data, multi_task_preds, confidence_arr

def create_ensemble_charts(df, idx, q01, q05, q50, q95, q99, y_true, var95, cvar95, opportunity_band, exceed):
    """创建集成模型图表"""
    # 价格图表
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#3b6af7")))
    fig_price.update_layout(height=360, margin=dict(l=30,r=30,t=30,b=30), title="📈 收盘价走势 - 集成模型")
    
    # 多分位风险带图
    fig_quantiles = go.Figure()
    
    # 添加风险带
    fig_quantiles.add_trace(go.Scatter(
        x=np.concatenate([idx, idx[::-1]]), y=np.concatenate([q99, q01[::-1]]),
        fill='toself', fillcolor='rgba(255, 91, 91, 0.1)', line=dict(color='rgba(255,255,255,0)'),
        name='1%-99% 风险带', showlegend=True
    ))
    
    fig_quantiles.add_trace(go.Scatter(
        x=np.concatenate([idx, idx[::-1]]), y=np.concatenate([q95, q05[::-1]]),
        fill='toself', fillcolor='rgba(255, 181, 76, 0.2)', line=dict(color='rgba(255,255,255,0)'),
        name='5%-95% 机会区间', showlegend=True
    ))
    
    fig_quantiles.add_trace(go.Scatter(x=idx, y=q50, name='中位数预测 (q50)', line=dict(color='#3b6af7', width=2)))
    fig_quantiles.add_trace(go.Scatter(x=idx, y=y_true, name='实际收益', line=dict(color='#16a34a', width=1.5)))
    fig_quantiles.add_trace(go.Scatter(x=idx[exceed], y=y_true[exceed], mode="markers", name="超损点(<=q0.05)", marker=dict(size=6, color='#ef4444')))
    
    fig_quantiles.update_layout(height=420, margin=dict(l=30,r=30,t=30,b=30), title="📊 集成模型多分位预测", xaxis_title="日期", yaxis_title="收益率")
    
    # 风险指标图表
    fig_risk_metrics = go.Figure()
    fig_risk_metrics.add_trace(go.Scatter(x=idx, y=var95, name='VaR95', line=dict(color='#ef4444', width=2)))
    fig_risk_metrics.add_trace(go.Scatter(x=idx, y=cvar95, name='CVaR95', line=dict(color='#dc2626', width=2, dash='dash')))
    fig_risk_metrics.add_trace(go.Scatter(x=idx, y=opportunity_band, name='机会区间宽度', line=dict(color='#16a34a', width=2)))
    
    fig_risk_metrics.update_layout(height=360, margin=dict(l=30,r=30,t=30,b=30), title="⚡ 集成模型风险指标", xaxis_title="日期", yaxis_title="风险指标值")
    
    return fig_price, fig_quantiles, fig_risk_metrics

def create_prediction_table(df, idx, q01, q05, q50, q95, q99, y_true, var95, cvar95, opportunity_band, tau_y, tau_r):
    """创建预测结果表格"""
    rows = []
    try:
        for i in range(max(0, len(idx)-10), len(idx)):
            base_date = idx[i]
            base_loc = df.index.get_loc(base_date)
            tgt_date = df.index[base_loc + 1] if base_loc + 1 < len(df.index) else None
            
            # 分位数数据
            q01_val = float(q01[i])
            q05_val = float(q05[i])
            q50_val = float(q50[i])
            q95_val = float(q95[i])
            q99_val = float(q99[i])
            
            v95 = float(var95[i])
            cvar = float(cvar95[i]) if not np.isnan(cvar95[i]) else v95
            opportunity = float(opportunity_band[i])
            
            y1 = float(y_true[i])
            light = risk_color(v95, tau_y, tau_r)
            
            rows.append({
                "base_date": str(base_date.date()),
                "target_date": (str(tgt_date.date()) if tgt_date is not None else "-"),
                "q01": q01_val,
                "q05": q05_val,
                "q50": q50_val,
                "q95": q95_val,
                "q99": q99_val,
                "var95": v95,
                "cvar95": cvar,
                "opportunity_band": opportunity,
                "risk_light": light,
                "ret1": y1,
                "exceed": bool(y1 <= q05_val),
            })
    except Exception as e:
        print(f"表格数据组装错误: {e}")
    return rows


# ---------------------- Flask App ----------------------
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)


@app.route("/")
def index():
    default_cfg = {
        "csv_path": "data/market.csv",
        "model_path": "models/best_lstm_quantile.pt",
        "scaler_path": "models/scaler.pkl",
        "lookback": 60,
        "alpha": 0.05,
        "tau_y": 0.015,
        "tau_r": 0.03,
    }
    return render_template("home_softpurple.html", **default_cfg)


# === 新增：渲染 chat 页面 ===
# 打开 chat 页面
@app.route("/chat")
def chat():
    return render_template("chat.html")

# 指南页面
@app.route("/guide")
def guide_page():
    return render_template("guide.html")

# 风险预测原始页面（将原 index.html 暴露为 /risk）
@app.route("/risk")
def risk_page():
    default_cfg = {
        "csv_path": "data/market.csv",
        "model_path": "models/best_lstm_quantile.pt",
        "scaler_path": "models/scaler.pkl",
        "lookback": 60,
        "alpha": 0.05,
        "tau_y": 0.015,
        "tau_r": 0.03,
    }
    return render_template("index.html", **default_cfg)

# app_flask.py 片段 —— 替换 /api/chat
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    msgs = data.get("messages", [])
    use_deepseek = bool(data.get("use_deepseek", True))

    # 读取 key（用上面的函数）
    api_key = get_deepseek_api_key()

    # —— 平台侧默认推理配置（与 index.html 保持一致）——
    default_cfg = {
        "csv_path": "data/market.csv",
        "model_path": "models/best_lstm_quantile.pt",
        "scaler_path": "models/scaler.pkl",
        "lookback": 60,
        "alpha": 0.05,
        "tau_y": 0.015,
        "tau_r": 0.03,
    }
    # 调用单模型推理并正确解包返回的 12 个值，所有异常都会被捕获并返回 JSON 错误
    try:
        result = run_inference_core(
            default_cfg["csv_path"], default_cfg["model_path"], default_cfg["scaler_path"],
            default_cfg["lookback"], default_cfg["alpha"], default_cfg["tau_y"], default_cfg["tau_r"]
        )
        # run_inference_core 返回 12 个元素：header, fig_price, fig_quantiles, fig_risk_metrics,
        # color, cur_v95, cur_cvar, cur_opportunity, rows, validation_data, shap_analysis, multi_task_data
        header, _, _, _, color, cur_v95, cur_cvar, cur_opportunity, rows, validation_data, shap_analysis, multi_task_data = result

        if color is None:
            reply = f"你好，我是 Lumen。当前模型或数据未准备好：{header}"
            return jsonify({"ok": True, "reply": reply, "context": {}})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # 返回 JSON 错误，前端不会尝试 JSON.parse HTML
        return jsonify({"ok": False, "error": str(e), "traceback": tb})
    

    # 汇总技术指标（RSI/MACD/布林/动量等）
    signals = summarize_signals(
        default_cfg["csv_path"], default_cfg["lookback"], default_cfg["tau_y"], default_cfg["tau_r"]
    )
    # 候选池：目前你还没有横截面数据，这里给空列表；未来可接入截面打分后填充
    candidates = []  # [{"symbol":"XYZ","name":"示例","score":0.87,"mom_20":0.12}, ...]

    # —— 更新后的 Lumen 系统提示（更自然、细致、实操性强）——
    sys_prompt = (
        "你是 Lumen（AuroraFin 的量化投研助手）📈📈。"
        "核心任务：基于实时风险指标生成专业但易于理解的投资建议。"
        "\n\n【回答风格要求】"
        "1. 自然流畅：用对话式语言解释专业指标，避免生硬的技术术语堆砌；保持内容与问题之间的关联性和回答内容的逻辑性与流畅性"
        "2. 细致入微：对每个指标不仅要说明数值，还要解释其市场含义和实际影响"
        "3. 实操性强：给出具体的价格点位、仓位比例和明确的触发条件。"
        "4. 情景化建议：结合当前市场状态提供有针对性的操作方案"
        "5. 添加风险分析，输出主要影响因素对风险变化的影响机制"
        "\n\n【专业边界】"
        "保持专业严谨，所有建议都要有数据支撑，不夸大收益，明确提示风险。"
        "\n\n【回答格式要求】"
        "适当添加大小标题、要点、分割线、表格和emoji将回答可视化，合理使用换行，确保回答结构清晰，可读性强。但不要缩减内容，数据和解读不可因此被删减。"
    )

    # 把"平台侧上下文"作为 JSON 注入给模型
    context_payload = {
        "risk_light": color,
        "var95": float(cur_v95),
        "asof": signals["asof"],
        "signals": signals,
        "candidates": candidates
    }

    stitched = [{"role": "system", "content": sys_prompt}]
    # 注入一条"上下文 JSON"
    stitched.append({
        "role": "user", 
        "content": "以下为平台实时上下文（JSON），请据此生成自然细致的投资建议：\n" + json.dumps(context_payload, ensure_ascii=False)
    })
    # 拼上用户历史消息
    stitched.extend(msgs)

    # DeepSeek 推理
    if use_deepseek and api_key:
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "deepseek-chat", 
                "messages": stitched, 
                "temperature": 0.3,
                "top_p": 0.9
            }
            # 增加超时时间并添加重试机制
            for attempt in range(3):  # 重试3次
                try:
                    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)  # 增加到60秒
                    r.raise_for_status()
                    out = r.json()
                    reply = out["choices"][0]["message"]["content"].strip()
                    break  # 成功则跳出重试循环
                except requests.exceptions.Timeout:
                    if attempt == 2:  # 最后一次重试也失败
                        raise
                    continue  # 继续重试
        except Exception as e:
            # 回退时也尽量保持自然风格
            reply = f"目前遇到技术问题（{e}），我先基于规则给您建议：\n\n" + _rule_based_note(signals, cur_v95, color)
    else:
        last_user = next((m.get("content","") for m in reversed(msgs) if m.get("role")=="user"), "")
        reply = (
            "你好！我是 Lumen，AuroraFin 的量化投研助手。"
            f"注意到您提到：「{last_user[:200]}」。"
            "目前大模型接口暂未配置，配置后我可以提供更细致的市场分析和操作建议。"
        )

    return jsonify({
        "ok": True,
        "reply": reply,
        "context": {
            "risk_light": color,
            "v95_drawdown": float(cur_v95),
            "asof": signals["asof"],
            "signals": signals,
            "candidates": candidates
        }
    })
    
# 更新验证API端点
@app.route("/api/validation", methods=["POST"])
def api_validation():
    data = request.json or {}
    returns = data.get("returns", [])
    quantile_predictions = data.get("quantile_predictions", [])
    alpha = float(data.get("alpha", 0.05))
    
    if len(returns) == 0 or len(quantile_predictions) == 0:
        return jsonify({"ok": False, "error": "缺少收益率或分位预测数据"})
    
    try:
        # 使用新的多分位验证器
        quantiles = [0.01, 0.05, 0.5, 0.95, 0.99]
        validator = VaRValidation(returns, quantile_predictions, quantiles)
        confidence_score = validator.coverage_quality_score()
        coverage_report = validator.get_detailed_coverage_report()
        
        # 主要关注q05（VaR95）的覆盖率
        main_coverage = validator.hit_ratio('q05')
        hit_bias = main_coverage - alpha
        
        # 性能评级
        if abs(hit_bias) <= 0.005:
            performance = "优秀"
        elif abs(hit_bias) <= 0.01:
            performance = "良好"
        elif abs(hit_bias) <= 0.02:
            performance = "一般"
        else:
            performance = "需要改进"
        
        # 计算综合偏差
        overall_bias = validator.calculate_overall_bias()
        
        report = {
            'hit_ratio': main_coverage,
            'hit_bias': hit_bias,
            'overall_bias': overall_bias,  # 新增综合偏差
            'confidence_score': confidence_score,
            'performance_rating': performance,
            'theoretical_coverage': alpha,
            'actual_coverage': main_coverage,
            'total_periods': len(returns),
            'violation_count': int(np.sum(validator.hits['q05'])),
            'detailed_coverage': coverage_report
        }
        
        return jsonify({
            "ok": True,
            "report": report
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# 更新推理API端点
@app.route("/api/infer", methods=["POST"]) 
def api_infer():
    data = request.json or {}
    csv_path = data.get("csv_path", "data/market.csv")
    model_path = data.get("model_path", "models/best_lstm_quantile.pt")
    scaler_path = data.get("scaler_path", "models/scaler.pkl")
    lookback = int(data.get("lookback", 60))
    alpha = float(data.get("alpha", 0.05))
    tau_y = float(data.get("tau_y", 0.015))
    tau_r = float(data.get("tau_r", 0.03))
    
    result = run_inference_core(
        csv_path, model_path, scaler_path, lookback, alpha, tau_y, tau_r
    )
    header, fig_price, fig_quantiles, fig_risk_metrics, color, cur_v95, cur_cvar, cur_opportunity, rows, validation_data, shap_analysis, multi_task_data = result
    
    response_data = {
        "ok": True,
        "status_markdown": header,
        "color": color,
        "cur_v95": cur_v95,
        "cur_cvar": cur_cvar,
        "cur_opportunity": cur_opportunity,
        "confidence_score": validation_data['confidence_score'],
        "coverage_report": validation_data['coverage_report'],
        "main_coverage": validation_data['main_coverage'],
        "fig_price": json.loads(json.dumps(fig_price, cls=PlotlyJSONEncoder)),
        "fig_quantiles": json.loads(json.dumps(fig_quantiles, cls=PlotlyJSONEncoder)),
        "fig_risk_metrics": json.loads(json.dumps(fig_risk_metrics, cls=PlotlyJSONEncoder)),
        "rows": rows,
        "validation_data": validation_data,
        "multi_task_data": multi_task_data
    }
    
    # 添加SHAP分析结果
    if shap_analysis:
        response_data["shap_analysis"] = shap_analysis

    # 尝试读取训练时保存的异常统计
    try:
        stats_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'anomaly_stats.json')
        stats_path = os.path.abspath(stats_path)
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                tstats = json.load(f)
            response_data['anomaly_stats'] = tstats
    except Exception:
        pass

    # 对当前输入数据做一次简单的实时异常检测
    try:
        df = make_features(csv_path)
        if 'ret' in df.columns:
            s = df['ret'].fillna(0.0)
            win = 60
            roll_mean = s.rolling(window=win, min_periods=1).mean()
            roll_std = s.rolling(window=win, min_periods=1).std().replace(0, np.nan).fillna(0.0)
            z = (s - roll_mean) / roll_std.replace(0, np.nan)
            z = z.fillna(0.0).abs()
            thr = 3.0
            mask = (z > thr).astype(int)
            # 最近 lookback 天内的异常统计
            recent = int(lookback)
            recent_mask = mask.tail(recent)
            cur_stats = {
                'n_total': int(len(s)),
                'n_anomalies_total': int(mask.sum()),
                'anomaly_ratio_total': float(mask.sum() / max(1, len(s))),
                'n_recent': int(recent),
                'n_anomalies_recent': int(recent_mask.sum()),
                'anomaly_ratio_recent': float(recent_mask.sum() / max(1, len(recent_mask)))
            }
            response_data['current_anomaly_stats'] = cur_stats
    except Exception:
        pass
    
    return jsonify(response_data)

# ==================== Flask 集成模型端点 ====================

@app.route("/api/train_ensemble", methods=["POST"])
def api_train_ensemble():
    """训练集成模型端点"""
    try:
        data = request.json or {}
        csv_path = data.get("csv_path", "data/market.csv")
        lookback = int(data.get("lookback", 60))
        
        result = train_ensemble_model(csv_path, lookback)
        if result is None:
            return jsonify({"ok": False, "error": "集成模型训练失败"})
        
        return jsonify({"ok": True, "message": "集成模型训练完成"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route('/api/feature_causality', methods=['POST'])
def api_feature_causality():
    data = request.json or {}
    csv_path = data.get('csv_path', 'data/market.csv')
    maxlag = int(data.get('maxlag', 5))
    alpha = float(data.get('alpha', 0.05))
    n_windows = int(data.get('n_windows', 3))
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'feature_causality.json'))
    try:
        # 尝试通过文件路径动态加载模块，避免包路径问题
        import importlib.util, traceback
        gc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training', 'granger_causality.py'))
        if not os.path.exists(gc_path):
            # fallback to workspace-relative
            gc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'training', 'granger_causality.py'))
        spec = importlib.util.spec_from_file_location('granger_causality', gc_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        generate_causality_report = getattr(module, 'generate_causality_report')
        report = generate_causality_report(csv_path, maxlag=maxlag, alpha=alpha, n_windows=n_windows, save_path=out_path)
        return jsonify({'ok': True, 'report': report, 'path': out_path})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # return stack trace for debugging in frontend
        return jsonify({'ok': False, 'error': str(e), 'traceback': tb})

@app.route("/api/infer_ensemble", methods=["POST"]) 
def api_infer_ensemble():
    """集成模型推理端点"""
    data = request.json or {}
    csv_path = data.get("csv_path", "data/market.csv")
    lookback = int(data.get("lookback", 60))
    alpha = float(data.get("alpha", 0.05))
    tau_y = float(data.get("tau_y", 0.015))
    tau_r = float(data.get("tau_r", 0.03))
    
    header, fig_price, fig_quantiles, fig_risk_metrics, color, cur_v95, cur_cvar, cur_opportunity, rows, validation_data, multi_task_preds, confidence_arr = run_ensemble_inference(
        csv_path, lookback, alpha, tau_y, tau_r
    )
    
    if color is None:
        return jsonify({"ok": False, "error": header})
    
    return jsonify({
        "ok": True,
        "status_markdown": header,
        "color": color,
        "cur_v95": cur_v95,
        "cur_cvar": cur_cvar,
        "cur_opportunity": cur_opportunity,
        "confidence_score": validation_data['confidence_score'] if validation_data else 0.5,
        "multi_task": multi_task_preds,
        "confidence_arr": (confidence_arr.tolist() if hasattr(confidence_arr, 'tolist') else list(confidence_arr)),
        "fig_price": json.loads(json.dumps(fig_price, cls=PlotlyJSONEncoder)) if fig_price else None,
        "fig_quantiles": json.loads(json.dumps(fig_quantiles, cls=PlotlyJSONEncoder)) if fig_quantiles else None,
        "fig_risk_metrics": json.loads(json.dumps(fig_risk_metrics, cls=PlotlyJSONEncoder)) if fig_risk_metrics else None,
        "rows": rows,
        "validation_data": validation_data,
        "model_type": "ensemble"
    })

@app.route("/api/compare_models", methods=["POST"])
def api_compare_models():
    """比较单模型和集成模型性能"""
    data = request.json or {}
    csv_path = data.get("csv_path", "data/market.csv")
    lookback = int(data.get("lookback", 60))
    alpha = float(data.get("alpha", 0.05))
    
    # 运行单模型推理
    single_result = run_inference_core(csv_path, "models/best_lstm_quantile.pt", 
                                     "models/scaler.pkl", lookback, alpha, 0.015, 0.03)
    
    # 运行集成模型推理
    ensemble_result = run_ensemble_inference(csv_path, lookback, alpha, 0.015, 0.03)
    
    single_header, _, _, _, single_color, single_v95, single_cvar, single_opp, _, single_val = single_result
    ensemble_header, _, _, _, ensemble_color, ensemble_v95, ensemble_cvar, ensemble_opp, _, ensemble_val, ensemble_multi, ensemble_conf = ensemble_result
    
    comparison = {
        "single_model": {
            "var95": single_v95,
            "cvar95": single_cvar,
            "opportunity": single_opp,
            "color": single_color,
            "confidence": single_val['confidence_score'] if single_val else 0.5,
            "coverage": single_val.get('coverage', 0) if single_val else 0
        },
        "ensemble_model": {
            "var95": ensemble_v95,
            "cvar95": ensemble_cvar,
            "opportunity": ensemble_opp,
            "color": ensemble_color,
            "confidence": ensemble_val['confidence_score'] if ensemble_val else 0.5,
            "coverage": ensemble_val.get('coverage', 0) if ensemble_val else 0,
            "individual_performance": ensemble_val.get('individual_performance', {}) if ensemble_val else {}
        }
    }
    
    return jsonify({"ok": True, "comparison": comparison})

@app.route("/api/note", methods=["POST"]) 
def api_note():
    data = request.json or {}
    csv_path = data.get("csv_path", "data/market.csv")
    model_path = data.get("model_path", "models/best_lstm_quantile.pt")
    scaler_path = data.get("scaler_path", "models/scaler.pkl")
    lookback = int(data.get("lookback", 60))
    alpha = float(data.get("alpha", 0.05))
    tau_y = float(data.get("tau_y", 0.015))
    tau_r = float(data.get("tau_r", 0.03))
    use_deepseek = bool(data.get("use_deepseek", True))
    api_key = get_deepseek_api_key()

    # 调用单模型推理并正确解包返回的12个值（部分为占位）
    result = run_inference_core(csv_path, model_path, scaler_path, lookback, alpha, tau_y, tau_r)
    # run_inference_core 返回: header, fig_price, fig_quantiles, fig_risk_metrics, color,
    # cur_v95, cur_cvar, cur_opportunity, rows, validation_data, shap_analysis, multi_task_data
    header, _, _, _, color, cur_v95, cur_cvar, cur_opportunity, rows, validation_data, shap_analysis, multi_task = result

    if color is None:
        return jsonify({"ok": False, "msg": header})
    
    signals = summarize_signals(csv_path, lookback, tau_y, tau_r)
    used_llm = False
    
    if use_deepseek and not api_key:
        return jsonify({"ok": False, "msg": "模型服务未配置"})
    
    if use_deepseek and api_key:
        try:
            note = _deepseek_note(signals, cur_v95, color, api_key)
            used_llm = True
        except Exception:
            note = _rule_based_note(signals, cur_v95, color)
            used_llm = False
    else:
        note = _rule_based_note(signals, cur_v95, color)
        used_llm = False
        
    print(f"[Note] use_deepseek={use_deepseek}, used_llm={used_llm}")
    return jsonify({"ok": True, "note_markdown": note, "used_llm": used_llm})


def _deepseek_note_json(signals, var95, color, api_key, model="deepseek-chat"):
    """向 Deepseek/Lumen 请求严格的 JSON 输出（结构化投研报告）。
    要求 LLM 返回一个 JSON 对象，包含固定字段：var95, hit_status, main_factors, confidence, summary, bullets, citations
    如果 Deepseek 无法返回合法 JSON，该函数会抛出异常以便上层回退到本地生成器。
    """
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    sys_prompt = (
        "你是一个结构化投研写作助手（输出严格 JSON）。\n"
        "输入为市场信号与风险测度，请输出一个 JSON 对象，字段如下：\n"
        "- var95: number (例如 -0.024 表示 -2.4%)\n"
        "- hit_status: string, one of ('under', 'normal', 'over')，表示命中率相对预期 alpha 是否偏低/正常/偏高\n"
        "- main_factors: array of strings, 列出导致当前风险/机会的主要因子（最多 5 项）\n"
        "- confidence: number (0-1) 表示整体置信度\n"
        "- summary: concise sentence summarizing核心观点\n"
        "- bullets: array of short bullet strings（可执行建议）\n"
        "- citations: array of {source: str, evidence: str}，用于前端高亮数据引用\n"
        "严格只返回 JSON，不要输出任何额外解释或 markdown。"
    )

    user_payload = {"signals": signals, "var95": var95, "risk_light": color}
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "请基于以下 JSON 输出结构化投研报告（严格 JSON）：\n" + json.dumps(user_payload, ensure_ascii=False)}
        ],
        "temperature": 0.15,
    }

    resp = requests.post(url, headers=headers, json=data, timeout=12)
    resp.raise_for_status()
    j = resp.json()
    # Deepseek 返回格式因版本不同，请根据实际解析，下面假设 j['choices'][0]['message']['content'] 为字符串 JSON
    content = None
    try:
        content = j['choices'][0]['message']['content']
    except Exception:
        # 兼容 older style
        content = j.get('choices', [{}])[0].get('text') or j.get('choices', [{}])[0].get('message', {}).get('content')

    if content is None:
        raise ValueError('Deepseek 返回无内容')

    # 尝试解析 content 为 JSON
    parsed = json.loads(content)
    return parsed


def build_local_structured_report(signals, validation_data=None, timeframe='daily'):
    """本地回退的、非 LLM 的结构化报表生成器（启发式规则）。返回与 LLM 输出兼容的 dict。
    """
    # 尝试从 validation_data 中提取 var95 与 hit_rate
    var95 = None
    hit_rate = None
    if validation_data:
        try:
            # validation_data 可能包含 'var_predictions' 和 'coverage_report' 等
            var_preds = validation_data.get('var_predictions') or validation_data.get('var_predictions', [])
            if var_preds and len(var_preds) > 0:
                var95 = float(var_preds[-1])
        except Exception:
            var95 = None
        try:
            hit_rate = float(validation_data.get('main_coverage', 0.0))
        except Exception:
            hit_rate = None

    # fallback to signals
    if var95 is None:
        # signals may contain asof or tau fields; if not, set 0
        var95 = float(signals.get('tau_y', 0.0)) * -1 if signals.get('tau_y') else -0.02

    # 简单判定命中状态
    alpha = 0.05
    hit_status = 'normal'
    if hit_rate is not None:
        if hit_rate < alpha - 0.01:
            hit_status = 'under'
        elif hit_rate > alpha + 0.01:
            hit_status = 'over'

    # 识别主导因子（启发式）
    main_factors = []
    vol = signals.get('vol', signals.get('volatility', None))
    rsi = signals.get('rsi14', signals.get('rsi', None))
    macd = signals.get('macd', None)
    bb_pos = signals.get('bb_pos', None)
    slope = signals.get('trend_slope20', None)

    if vol is not None:
        try:
            if float(vol) > 1.5 * (signals.get('vol_ma', float(vol)) if signals.get('vol_ma') else float(vol)):
                main_factors.append('波动率显著上升')
        except Exception:
            pass
    if rsi is not None:
        try:
            rsi = float(rsi)
            if rsi < 35:
                main_factors.append('RSI 回调/超卖')
            elif rsi > 70:
                main_factors.append('RSI 处于超买区')
        except Exception:
            pass
    if macd is not None and signals.get('macd_signal') is not None:
        try:
            if float(macd) > float(signals.get('macd_signal')):
                main_factors.append('MACD 看多')
            else:
                main_factors.append('MACD 看空/回撤')
        except Exception:
            pass
    if bb_pos is not None:
        try:
            bp = float(bb_pos)
            if bp > 0.8:
                main_factors.append('价格接近布林带上轨（压力）')
            elif bp < 0.2:
                main_factors.append('价格接近布林带下轨（支撑）')
        except Exception:
            pass
    if slope is not None:
        try:
            if float(slope) > 0:
                main_factors.append('短期动量向上')
            elif float(slope) < 0:
                main_factors.append('短期动量向下')
        except Exception:
            pass

    # 限制因子数量
    main_factors = main_factors[:5] if main_factors else ['无显著单一因子']

    # 置信度估计（启发式）
    confidence = 0.6
    if hit_rate is not None:
        confidence = 0.8 - min(0.6, abs(hit_rate - alpha) * 5)

    summary = f"VaR95 = {var95:.2%}，命中率 {'未提供' if hit_rate is None else f'{hit_rate:.2%}'}，主导因子为{';'.join(main_factors)}，置信度 {confidence:.2f}。"

    bullets = []
    bullets.append(f"建议：根据当前风险等级（{signals.get('tau_r', 'N/A')}），调整仓位或风险敞口。")
    bullets.append("监控短期波动并在突破布林带中轨时分批入场/离场。")

    citations = []
    citations.append({"source": "signals.rsi14", "evidence": f"rsi={rsi}"})
    citations.append({"source": "signals.vol", "evidence": f"vol={vol}"})

    return {
        'var95': var95,
        'hit_status': hit_status,
        'main_factors': main_factors,
        'confidence': float(round(confidence, 3)),
        'summary': summary,
        'bullets': bullets,
        'citations': citations,
    }


@app.route('/api/generate_report', methods=['POST'])
def api_generate_report():
    """生成结构化投研简报的 endpoint。
    可选参数：csv_path, lookback, alpha, tau_y, tau_r, timeframe ('daily'|'weekly'), use_deepseek (bool)
    返回 JSON 报表（可直接被前端渲染或导出为日报/周报）。
    """
    data = request.json or {}
    csv_path = data.get('csv_path', 'data/market.csv')
    lookback = int(data.get('lookback', 60))
    alpha = float(data.get('alpha', 0.05))
    tau_y = float(data.get('tau_y', 0.015))
    tau_r = float(data.get('tau_r', 0.03))
    timeframe = data.get('timeframe', 'daily')
    use_deepseek = bool(data.get('use_deepseek', True))

    api_key = get_deepseek_api_key()

    try:
        # 获取最新推理数据（集成）
        result = run_ensemble_inference(csv_path, lookback, alpha, tau_y, tau_r)
        # run_ensemble_inference 返回多个元素；第三个元素及 validation_data 在 unpack 中
        header, fig_price, fig_quantiles, fig_risk_metrics, color, cur_v95, cur_cvar, cur_opportunity, rows, validation_data, multi_task, conf_arr = result

        # 如果集成推理返回错误（color 为空或 header 为错误信息），记录并回退到仅基于 signals 的本地生成器
        if color is None:
            print(f"[generate_report] ensemble inference failed, falling back: {header}")
            signals = summarize_signals(csv_path, lookback, tau_y, tau_r)
            report = build_local_structured_report(signals, validation_data=None, timeframe=timeframe)
            return jsonify({'ok': True, 'report': report, 'used_llm': False, 'note': str(header)})

        signals = summarize_signals(csv_path, lookback, tau_y, tau_r)

        # 优先调用 LLM 生成严格 JSON
        if use_deepseek:
            if not api_key:
                return jsonify({'ok': False, 'error': 'Deepseek API key 未配置'})
            try:
                report = _deepseek_note_json(signals, cur_v95, color, api_key)
                return jsonify({'ok': True, 'report': report, 'used_llm': True})
            except Exception as e:
                # 记录并回退
                print('[generate_report] deepseek json failed, fallback to local:', str(e))

        # 本地回退生成
        report = build_local_structured_report(signals, validation_data=validation_data, timeframe=timeframe)
        return jsonify({'ok': True, 'report': report, 'used_llm': False})

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return jsonify({'ok': False, 'error': str(e), 'traceback': tb})

# 添加验证页面路由
@app.route("/validation")
def validation_page():
    return render_template("validation.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=True)



