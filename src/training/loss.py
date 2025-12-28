import torch

def multi_quantile_loss(pred, y, quantiles=[0.01, 0.05, 0.5, 0.95, 0.99]):
    """
    多分位损失函数
    pred: (B, num_quantiles) 每个分位的预测值
    y: (B,) 真实值
    quantiles: 分位列表
    """
    losses = []
    for i, q in enumerate(quantiles):
        error = y - pred[:, i]
        loss = torch.maximum(q * error, (q - 1) * error)
        losses.append(loss)
    
    # 对所有分位损失求平均
    total_loss = torch.stack(losses, dim=1).mean(dim=1).mean()
    return total_loss

def compute_coverage(pred, y, quantiles):
    """
    计算各分位的覆盖率
    """
    coverage = {}
    for i, q in enumerate(quantiles):
        if q < 0.5:
            cov = (y <= pred[:, i]).float().mean().item()
        else:
            cov = (y >= pred[:, i]).float().mean().item()
        coverage[f'q{int(q*100):02d}'] = cov
    return coverage