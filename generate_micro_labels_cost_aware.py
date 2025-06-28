#!/usr/bin/env python3
"""
成本感知的微观标签生成
使用更长的horizon和成本阈值，生成{亏/赚/平}三分类标签
"""

import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
from utils.labeling import make_labels, get_label_stats

def load_latest_features():
    """Load the latest micro features file"""
    files = glob.glob("data/micro_features_*.parquet")
    if not files:
        raise FileNotFoundError("No micro features files found")
    
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading: {latest_file}")
    df = pd.read_parquet(latest_file)
    print(f"Loaded {len(df)} rows")
    return df

def generate_cost_aware_labels(df, horizon=2000, cost_multiplier=2.0, fee_rate=0.0005):
    """
    生成成本感知的标签
    horizon: 预测时间窗口（毫秒）
    cost_multiplier: 成本倍数（相对于spread）
    fee_rate: 手续费率
    """
    print(f"Generating cost-aware labels with:")
    print(f"  horizon: {horizon} steps")
    print(f"  cost_multiplier: {cost_multiplier}")
    print(f"  fee_rate: {fee_rate}")
    
    # 计算成本阈值
    df['cost_threshold'] = cost_multiplier * df['rel_spread'] + fee_rate
    
    # 计算未来价格变动
    df['future_price'] = df['mid_price'].shift(-horizon)
    df['price_change'] = (df['future_price'] - df['mid_price']) / df['mid_price']
    
    # 生成标签：1=赚, 0=平, -1=亏
    df['label'] = 0  # 默认平
    
    # 多头：价格涨幅超过成本阈值
    long_mask = df['price_change'] > df['cost_threshold']
    df.loc[long_mask, 'label'] = 1
    
    # 空头：价格跌幅超过成本阈值
    short_mask = df['price_change'] < -df['cost_threshold']
    df.loc[short_mask, 'label'] = -1
    
    # 移除无法计算未来价格的样本
    df = df.dropna(subset=['future_price', 'price_change'])
    
    return df

def main():
    # 加载数据
    df = load_latest_features()
    
    # 测试不同的horizon
    horizons = [1000, 2000, 5000]
    cost_multipliers = [1.5, 2.0, 2.5]
    
    best_config = None
    best_balance = 0
    
    print("\n=== 测试不同参数组合 ===")
    
    for horizon in horizons:
        for cost_mult in cost_multipliers:
            print(f"\n测试: horizon={horizon}, cost_mult={cost_mult}")
            
            # 生成标签
            labeled_df = generate_cost_aware_labels(df.copy(), horizon, cost_mult)
            
            # 统计标签分布
            label_counts = labeled_df['label'].value_counts().sort_index()
            total_samples = len(labeled_df)
            
            print(f"  总样本: {total_samples}")
            print(f"  标签分布: {dict(label_counts)}")
            
            # 计算平衡度（避免极端不平衡）
            balance_score = min(label_counts) / max(label_counts) if len(label_counts) > 1 else 0
            print(f"  平衡度: {balance_score:.3f}")
            
            # 计算有效信号比例
            effective_signals = (label_counts.get(1, 0) + label_counts.get(-1, 0)) / total_samples
            print(f"  有效信号比例: {effective_signals:.3f}")
            
            # 选择最佳配置（平衡度好且有效信号适中）
            if balance_score > 0.1 and 0.1 < effective_signals < 0.8:
                if balance_score > best_balance:
                    best_balance = balance_score
                    best_config = (horizon, cost_mult)
    
    if best_config:
        horizon, cost_mult = best_config
        print(f"\n=== 选择最佳配置: horizon={horizon}, cost_mult={cost_mult} ===")
        
        # 用最佳配置生成最终标签
        final_df = generate_cost_aware_labels(df.copy(), horizon, cost_mult)
        
        # 统计最终结果
        label_counts = final_df['label'].value_counts().sort_index()
        total_samples = len(final_df)
        
        print(f"最终标签分布:")
        print(f"  总样本: {total_samples}")
        print(f"  空头(-1): {label_counts.get(-1, 0)} ({label_counts.get(-1, 0)/total_samples:.1%})")
        print(f"  平仓(0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/total_samples:.1%})")
        print(f"  多头(1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/total_samples:.1%})")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/micro_features_cost_aware_labels_{timestamp}.parquet"
        final_df.to_parquet(output_file, index=False)
        print(f"\n标签数据已保存: {output_file}")
        
        # 显示样本数据
        print(f"\n样本数据:")
        sample_cols = ['timestamp', 'mid_price', 'rel_spread', 'cost_threshold', 'price_change', 'label']
        print(final_df[sample_cols].head(10))
        
    else:
        print("未找到合适的参数配置，请调整参数范围")

if __name__ == "__main__":
    main() 