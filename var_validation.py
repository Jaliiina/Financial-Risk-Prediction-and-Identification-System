import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class VaRValidation:
    def __init__(self, returns, var_predictions, alpha=0.05, window=250):
        """
        VaR预测验证类
        
        Parameters:
        -----------
        returns : array-like
            实际收益率序列
        var_predictions : array-like  
            VaR预测序列
        alpha : float
            VaR置信水平 (默认5%)
        window : int
            滚动窗口大小
        """
        self.returns = np.array(returns)
        self.var_predictions = np.array(var_predictions)
        self.alpha = alpha
        self.window = window
        self.hits = None
        self._calculate_hits()
    
    def _calculate_hits(self):
        """计算突破事件（实际收益低于VaR预测）"""
        self.hits = (self.returns < -self.var_predictions).astype(int)
        return self.hits
    
    def hit_ratio(self):
        """计算命中率"""
        if self.hits is None:
            self._calculate_hits()
        return np.mean(self.hits)
    
    def hit_bias(self):
        """计算命中偏差（实际命中率 - 理论命中率）"""
        return self.hit_ratio() - self.alpha
    
    def kupiec_test(self):
        """
        Kupiec回测检验
        原假设：模型预测的VaR是准确的
        """
        if self.hits is None:
            self._calculate_hits()
        
        n = len(self.hits)
        x = np.sum(self.hits)  # 突破次数
        p_expected = self.alpha  # 理论突破概率
        p_actual = x / n  # 实际突破概率
        
        # 似然比检验
        if p_actual == 0:
            LR = -2 * np.log(((1 - p_expected) ** n))
        elif p_actual == 1:
            LR = -2 * np.log((p_expected ** n))
        else:
            LR = -2 * np.log(
                ((1 - p_expected) ** (n - x) * p_expected ** x) / 
                ((1 - p_actual) ** (n - x) * p_actual ** x)
            )
        
        p_value = 1 - stats.chi2.cdf(LR, 1)
        
        return {
            'LR_statistic': LR,
            'p_value': p_value,
            'reject_null': p_value < 0.05,  # 在5%水平下拒绝原假设
            'expected_violations': n * p_expected,
            'actual_violations': x,
            'total_observations': n
        }
    
    def rolling_coverage(self, window=None):
        """滚动覆盖率"""
        if window is None:
            window = self.window
        
        if self.hits is None:
            self._calculate_hits()
        
        coverage = pd.Series(self.hits).rolling(window=window).mean()
        return coverage.dropna()
    
    def confidence_score(self, recent_window=60):
        """
        计算当前预测的置信度分数
        基于近期命中率的稳定性
        """
        if len(self.hits) < recent_window:
            return 0.5  # 数据不足时返回中性分数
        
        recent_hits = self.hits[-recent_window:]
        recent_coverage = np.mean(recent_hits)
        
        # 计算命中率与理论值的偏差
        coverage_bias = abs(recent_coverage - self.alpha)
        
        # 计算命中率的稳定性（波动性）
        if len(recent_hits) >= 20:
            rolling_20 = pd.Series(recent_hits).rolling(20).mean().dropna()
            stability = 1 - min(1, rolling_20.std() / 0.1)  # 标准化稳定性
        else:
            stability = 0.5
        
        # 综合置信度分数
        bias_penalty = max(0, 1 - coverage_bias / self.alpha)
        confidence = 0.7 * bias_penalty + 0.3 * stability
        
        return min(1.0, max(0.0, confidence))
    
    def generate_validation_report(self):
        """生成完整的验证报告"""
        hit_ratio = self.hit_ratio()
        hit_bias_val = self.hit_bias()
        kupiec = self.kupiec_test()
        confidence = self.confidence_score()
        
        # 评估模型表现
        if abs(hit_bias_val) <= 0.005:  # 偏差在0.5%以内
            performance = "优秀"
        elif abs(hit_bias_val) <= 0.01:  # 偏差在1%以内
            performance = "良好"
        elif abs(hit_bias_val) <= 0.02:  # 偏差在2%以内
            performance = "一般"
        else:
            performance = "需要改进"
        
        report = {
            'hit_ratio': hit_ratio,
            'hit_bias': hit_bias_val,
            'kupiec_test': kupiec,
            'confidence_score': confidence,
            'performance_rating': performance,
            'theoretical_coverage': self.alpha,
            'actual_coverage': hit_ratio,
            'total_periods': len(self.returns),
            'violation_count': int(np.sum(self.hits))
        }
        
        return report

def create_coverage_plots(returns, var_predictions, alpha=0.05):
    """创建覆盖率可视化图表"""
    
    validator = VaRValidation(returns, var_predictions, alpha)
    report = validator.generate_validation_report()
    hits = validator.hits
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'VaR突破事件时序图',
            '滚动覆盖率 (250日窗口)',
            '命中率分布直方图',
            '模型表现指标'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "indicator"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 子图1: VaR突破事件
    dates = pd.date_range(start='2020-01-01', periods=len(returns), freq='D')
    fig.add_trace(
        go.Scatter(x=dates, y=returns, name='实际收益', line=dict(color='blue'), opacity=0.6),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=-var_predictions, name='VaR95预测', line=dict(color='red')),
        row=1, col=1
    )
    
    # 标记突破点
    violation_dates = dates[hits.astype(bool)]
    violation_returns = returns[hits.astype(bool)]
    fig.add_trace(
        go.Scatter(x=violation_dates, y=violation_returns, 
                  mode='markers', name='突破事件',
                  marker=dict(color='black', size=6, symbol='x')),
        row=1, col=1
    )
    
    # 子图2: 滚动覆盖率
    rolling_cov = validator.rolling_coverage(250)
    fig.add_trace(
        go.Scatter(x=dates[len(dates)-len(rolling_cov):], y=rolling_cov, 
                  name='滚动覆盖率', line=dict(color='green')),
        row=1, col=2
    )
    # 添加理论水平线
    fig.add_hline(y=alpha, line_dash="dash", line_color="red", 
                  annotation_text=f"理论水平 ({alpha:.1%})", 
                  row=1, col=2)
    
    # 子图3: 命中率分布
    hit_series = pd.Series(hits)
    hit_distribution = [len(hit_series[hit_series==0]), len(hit_series[hit_series==1])]
    fig.add_trace(
        go.Bar(x=['未突破', '突破'], y=hit_distribution, 
               marker_color=['lightblue', 'lightcoral']),
        row=2, col=1
    )
    
    # 子图4: 置信度指标
    fig.add_trace(
        go.Indicator(
            mode = "gauge+number+delta",
            value = report['confidence_score'],
            domain = {'row': 1, 'col': 1},
            title = {'text': "置信度分数"},
            delta = {'reference': 0.7, 'increasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [0, 1], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.6], 'color': "lightgray"},
                    {'range': [0.6, 0.8], 'color': "yellow"},
                    {'range': [0.8, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.7
                }
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, 
                     title_text=f"VaR模型验证报告 | 命中率: {report['hit_ratio']:.2%} | 性能: {report['performance_rating']}")
    
    return fig, report