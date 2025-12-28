#!/usr/bin/env python3
"""
多任务集成模型训练脚本
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径 - 修复路径问题
project_root = Path(__file__).resolve().parent.parent  # 向上两级到项目根目录
sys.path.insert(0, str(project_root))

import torch
import numpy as np

# 直接导入函数，避免src前缀
try:
    from app.app_flask import train_ensemble_model, prepare_multitask_targets, make_features, make_sequences
except ImportError:
    # 备用导入方式
    try:
        from src.app.app_flask import train_ensemble_model, prepare_multitask_targets, make_features, make_sequences
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("📁 当前Python路径:")
        for path in sys.path:
            print(f"  - {path}")
        sys.exit(1)

def main():
    print("🎯 开始多任务集成模型训练")
    
    # 检查数据文件
    csv_path = "data/market.csv"
    if not os.path.exists(csv_path):
        print(f"❌ 数据文件不存在: {csv_path}")
        print("请先运行数据抓取脚本: python src/data/fetch.py")
        return
    
    # 检查序列数据
    seq_path = "data/seq_lbk60_h1.npz"
    if not os.path.exists(seq_path):
        print(f"⚠️ 序列数据不存在，正在生成...")
        try:
            df = make_features(csv_path)
            X, y, dates = make_sequences(df, lookback=60)
            np.savez(seq_path, X=X, y=y, dates=dates)
            print(f"✅ 序列数据已生成: X{X.shape}, y{y.shape}")
        except Exception as e:
            print(f"❌ 生成序列数据失败: {e}")
            return
    
    # 训练参数
    lookback = 60
    multitask = True
    
    print(f"📊 训练配置:")
    print(f"  - 数据路径: {csv_path}")
    print(f"  - 回看窗口: {lookback}")
    print(f"  - 多任务模式: {multitask}")
    print(f"  - 设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # 开始训练
    try:
        ensemble_model = train_ensemble_model(
            csv_path=csv_path,
            lookback=lookback,
            multitask=multitask
        )
        
        if ensemble_model:
            print("🎉 多任务集成模型训练完成！")
            print("📁 模型文件已保存到 models/ 目录")
        else:
            print("❌ 训练失败")
            
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()