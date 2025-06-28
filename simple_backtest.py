#!/usr/bin/env python3
"""
简化回测脚本
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def simple_backtest_15m():
    """简化15m回测"""
    print("📊 简化回测15m模型...")
    
    # 读取数据和模型
    df = pd.read_parquet('/Users/qiutianyu/data/processed/features_15m_2023_2025.parquet')
    model = xgb.XGBClassifier()
    model.load_model('xgb_15m_optimized.bin')
    
    # 准备特征
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'trend_1h', 'trend_4h']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols].fillna(0)
    predictions = model.predict_proba(X)[:, 1]
    
    # 简单回测逻辑
    long_signals = (predictions > 0.8) & (df['trend_1h'] == 1) & (df['trend_4h'] == 1)
    short_signals = (predictions < 0.2) & (df['trend_1h'] == -1) & (df['trend_4h'] == -1)
    
    print(f"做多信号数: {long_signals.sum()}")
    print(f"做空信号数: {short_signals.sum()}")
    print(f"总信号数: {long_signals.sum() + short_signals.sum()}")
    
    # 计算信号质量
    long_accuracy = df[long_signals]['label'].mean() if long_signals.sum() > 0 else 0
    short_accuracy = (1 - df[short_signals]['label']).mean() if short_signals.sum() > 0 else 0
    
    print(f"做多信号准确率: {long_accuracy:.2%}")
    print(f"做空信号准确率: {short_accuracy:.2%}")
    
    return long_signals.sum() + short_signals.sum(), long_accuracy, short_accuracy

def simple_backtest_5m():
    """简化5m回测"""
    print("\n📊 简化回测5m模型...")
    
    # 读取数据和模型
    df = pd.read_parquet('/Users/qiutianyu/data/processed/features_5m_2023_2025.parquet')
    model = xgb.XGBClassifier()
    model.load_model('xgb_5m_optimized.bin')
    
    # 准备特征
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'trend_1h', 'trend_4h']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols].fillna(0)
    predictions = model.predict_proba(X)[:, 1]
    
    # 简单回测逻辑
    long_signals = (predictions > 0.8) & (df['trend_1h'] == 1) & (df['trend_4h'] == 1)
    short_signals = (predictions < 0.2) & (df['trend_1h'] == -1) & (df['trend_4h'] == -1)
    
    print(f"做多信号数: {long_signals.sum()}")
    print(f"做空信号数: {short_signals.sum()}")
    print(f"总信号数: {long_signals.sum() + short_signals.sum()}")
    
    # 计算信号质量
    long_accuracy = df[long_signals]['label'].mean() if long_signals.sum() > 0 else 0
    short_accuracy = (1 - df[short_signals]['label']).mean() if short_signals.sum() > 0 else 0
    
    print(f"做多信号准确率: {long_accuracy:.2%}")
    print(f"做空信号准确率: {short_accuracy:.2%}")
    
    return long_signals.sum() + short_signals.sum(), long_accuracy, short_accuracy

def main():
    print("🎯 简化回测优化后的模型...")
    
    # 回测15m
    signals_15m, long_acc_15m, short_acc_15m = simple_backtest_15m()
    
    # 回测5m
    signals_5m, long_acc_5m, short_acc_5m = simple_backtest_5m()
    
    # 结果总结
    print("\n📊 简化回测结果总结:")
    print(f"15m模型 - 信号数: {signals_15m}, 做多准确率: {long_acc_15m:.2%}, 做空准确率: {short_acc_15m:.2%}")
    print(f"5m模型 - 信号数: {signals_5m}, 做多准确率: {long_acc_5m:.2%}, 做空准确率: {short_acc_5m:.2%}")
    
    print("\n🎉 简化回测完成!")

if __name__ == "__main__":
    main() 