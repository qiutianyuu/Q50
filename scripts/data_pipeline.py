#!/usr/bin/env python3
"""
数据管道脚本
用于自动化工作流中的数据更新
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import glob
import os

def check_latest_data():
    """检查最新数据文件"""
    # 检查特征文件
    feature_files = glob.glob("data/realtime_features_*.parquet")
    if not feature_files:
        print("❌ 没有找到特征文件")
        return False
    
    latest_feature = max(feature_files, key=os.path.getctime)
    print(f"✅ 最新特征文件: {latest_feature}")
    
    # 检查数据时间范围
    df = pd.read_parquet(latest_feature)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        latest_time = df['timestamp'].max()
        earliest_time = df['timestamp'].min()
        print(f"📊 数据时间范围: {earliest_time} 到 {latest_time}")
        print(f"📊 总记录数: {len(df)}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Data Pipeline')
    parser.add_argument('--days', type=int, default=1, help='Number of days to check')
    parser.add_argument('--check-only', action='store_true', help='Only check data, do not update')
    
    args = parser.parse_args()
    
    print(f"🔍 检查数据管道 (最近 {args.days} 天)")
    
    # 检查现有数据
    if check_latest_data():
        print("✅ 数据管道检查完成")
    else:
        print("❌ 数据管道检查失败")
        return False
    
    # 这里可以添加数据更新逻辑
    # 例如：从WebSocket获取新数据、合并历史数据等
    
    return True

if __name__ == "__main__":
    main() 