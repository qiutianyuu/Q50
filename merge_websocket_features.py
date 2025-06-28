#!/usr/bin/env python3
"""
将websocket特征与现有的15分钟特征合并
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def main():
    """主函数"""
    print("开始合并websocket特征...")
    
    # 加载现有特征
    try:
        existing_features = pd.read_parquet('data/features_15m_2023_2025.parquet')
        print(f"现有特征: {existing_features.shape}")
    except FileNotFoundError:
        print("未找到现有特征文件，将只使用websocket特征")
        existing_features = pd.DataFrame()
    
    # 加载websocket特征
    try:
        websocket_features = pd.read_parquet('data/websocket_features_15m.parquet')
        print(f"Websocket特征: {websocket_features.shape}")
    except FileNotFoundError:
        print("未找到websocket特征文件")
        return
    
    # 清理websocket特征
    websocket_features = websocket_features[websocket_features['mid_price'] > 0].copy()
    websocket_features = websocket_features.reset_index(drop=True)
    
    # 重命名websocket特征以避免冲突
    websocket_columns = [col for col in websocket_features.columns if col != 'timestamp']
    websocket_features = websocket_features.rename(columns={
        col: f'ws_{col}' for col in websocket_columns
    })
    
    print(f"清理后websocket特征: {websocket_features.shape}")
    
    if len(existing_features) > 0:
        # 合并特征
        print("正在合并特征...")
        
        # 确保时间戳格式一致
        existing_features['timestamp'] = pd.to_datetime(existing_features['timestamp'])
        websocket_features['timestamp'] = pd.to_datetime(websocket_features['timestamp'])
        
        # 合并
        merged_features = pd.merge(
            existing_features, 
            websocket_features, 
            on='timestamp', 
            how='left'
        )
        
        print(f"合并后特征: {merged_features.shape}")
        
        # 填充缺失的websocket特征
        ws_columns = [col for col in merged_features.columns if col.startswith('ws_')]
        merged_features[ws_columns] = merged_features[ws_columns].fillna(0)
        
        # 保存合并后的特征
        output_file = 'data/features_15m_with_websocket.parquet'
        merged_features.to_parquet(output_file, compression='zstd')
        
        print(f"合并特征已保存到: {output_file}")
        print(f"特征维度: {merged_features.shape}")
        
        # 显示特征统计
        print(f"\nWebsocket特征覆盖范围:")
        ws_coverage = merged_features[ws_columns].notna().sum()
        print(f"有websocket数据的时间窗口: {ws_coverage.iloc[0]} / {len(merged_features)}")
        
        # 显示样本数据
        print("\n样本数据:")
        print(merged_features[['timestamp'] + ws_columns[:5]].head())
        
    else:
        # 只有websocket特征
        print("只有websocket特征，直接保存...")
        output_file = 'data/features_15m_websocket_only.parquet'
        websocket_features.to_parquet(output_file, compression='zstd')
        print(f"Websocket特征已保存到: {output_file}")

if __name__ == "__main__":
    main() 