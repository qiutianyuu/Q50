#!/usr/bin/env python3
"""
合并订单流特征到主特征表
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def merge_orderflow_features():
    """合并订单流特征到主特征表"""
    print("🔄 合并订单流特征...")
    
    # 加载主特征表
    main_file = Path("/Users/qiutianyu/data/processed/features_15m_2023_2025.parquet")
    orderflow_file = Path("data/mid_features_15m_orderflow.parquet")
    
    print(f"📥 加载主特征表: {main_file}")
    main = pd.read_parquet(main_file)
    print(f"主表形状: {main.shape}")
    
    print(f"📥 加载订单流特征: {orderflow_file}")
    orderflow = pd.read_parquet(orderflow_file)
    print(f"订单流表形状: {orderflow.shape}")
    
    # 确保时间戳格式一致
    main['timestamp'] = pd.to_datetime(main['timestamp'], utc=True)
    orderflow['timestamp'] = pd.to_datetime(orderflow['timestamp'], utc=True)
    
    print(f"主表时间范围: {main['timestamp'].min()} 到 {main['timestamp'].max()}")
    print(f"订单流时间范围: {orderflow['timestamp'].min()} 到 {orderflow['timestamp'].max()}")
    
    # 选择高信息密度的订单流特征
    exclude_cols = ['timestamp', 'mid_price', 'bid_price', 'ask_price', 'vwap', 'price_mean', 'price_std', 'price_range']
    orderflow_cols = [col for col in orderflow.columns if col not in exclude_cols]
    
    print(f"📊 选择 {len(orderflow_cols)} 个订单流特征:")
    for col in orderflow_cols:
        print(f"  - {col}")
    
    # 合并特征
    print("🔗 执行特征合并...")
    merged = pd.merge_asof(
        main.sort_values('timestamp'),
        orderflow[['timestamp'] + orderflow_cols].sort_values('timestamp'),
        on='timestamp', direction='backward'
    )
    
    print(f"合并后形状: {merged.shape}")
    
    # 填充NaN值
    print("🔧 填充缺失值...")
    merged[orderflow_cols] = merged[orderflow_cols].fillna(0)
    
    # 检查合并结果
    print("📊 检查合并结果...")
    print(f"主表特征数: {len([col for col in main.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label']])}")
    print(f"新增特征数: {len(orderflow_cols)}")
    print(f"总特征数: {len([col for col in merged.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label']])}")
    
    # 检查时间对齐
    print("⏰ 检查时间对齐...")
    merged_sorted = merged.sort_values('timestamp')
    time_gaps = merged_sorted['timestamp'].diff().dt.total_seconds() / 900  # 15分钟
    print(f"时间间隔统计: 平均={time_gaps.mean():.2f}个15分钟, 最大={time_gaps.max():.2f}")
    
    # 保存合并后的特征表
    output_file = Path("/Users/qiutianyu/data/processed/features_15m_enhanced.parquet")
    merged.to_parquet(output_file, index=False)
    print(f"✅ 合并后的特征表已保存: {output_file}")
    
    # 显示新增特征的前几行
    print("\n📈 新增特征样本:")
    sample_cols = orderflow_cols[:10]  # 显示前10个特征
    print(merged[['timestamp'] + sample_cols].head())
    
    return merged, orderflow_cols

def main():
    print("=== 订单流特征合并 ===")
    
    merged_df, orderflow_features = merge_orderflow_features()
    
    print(f"\n🎉 合并完成!")
    print(f"最终特征表: {merged_df.shape[0]} 行, {merged_df.shape[1]} 列")
    print(f"新增订单流特征: {len(orderflow_features)} 个")
    
    return merged_df, orderflow_features

if __name__ == "__main__":
    main() 