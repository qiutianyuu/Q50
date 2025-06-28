#!/usr/bin/env python3
"""
整合历史数据 - 合并多个月份的K线数据
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import argparse

def merge_1h_data(start_year=2023, start_month=1, end_year=2025, end_month=5):
    """合并1小时数据"""
    print(f"🔄 合并1小时数据: {start_year}-{start_month:02d} 到 {end_year}-{end_month:02d}")
    
    all_data = []
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # 跳过超出范围的月份
            if year == start_year and month < start_month:
                continue
            if year == end_year and month > end_month:
                continue
                
            file_path = f"/Users/qiutianyu/ETHUSDT-1h/ETHUSDT-1h-{year}-{month:02d}/ETHUSDT-1h-{year}-{month:02d}.csv"
            
            if os.path.exists(file_path):
                try:
                    print(f"📁 加载: {file_path}")
                    # 读取没有列名的CSV文件
                    df = pd.read_csv(file_path, header=None)
                    
                    # 设置列名（Binance K线格式）
                    df.columns = [
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                        'taker_buy_quote_volume', 'ignore'
                    ]
                    
                    # 转换时间戳（毫秒转秒）
                    df['timestamp'] = pd.to_datetime(df['open_time'] / 1000, unit='s', utc=True)
                    
                    # 只保留需要的列
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    
                    all_data.append(df)
                except Exception as e:
                    print(f"⚠️ 加载失败 {file_path}: {e}")
            else:
                print(f"⚠️ 文件不存在: {file_path}")
    
    if not all_data:
        print("❌ 没有找到任何数据文件")
        return None
    
    # 合并所有数据
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # 按时间排序并去重
    merged_df = merged_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
    
    print(f"✅ 合并完成")
    print(f"📊 总样本数: {len(merged_df)}")
    print(f"📅 时间范围: {merged_df['timestamp'].min()} 到 {merged_df['timestamp'].max()}")
    
    return merged_df

def merge_5m_data_v2(root_dir="/Users/qiutianyu/ETHUSDT-5m"):
    """递归遍历ETHUSDT-5m目录，合并所有5分钟数据"""
    print(f"🔄 遍历并合并5分钟数据: {root_dir}")
    all_data = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.startswith("ETHUSDT-5m-") and file.endswith(".csv"):
                file_path = os.path.join(subdir, file)
                try:
                    print(f"📁 加载: {file_path}")
                    df = pd.read_csv(file_path, header=None)
                    df.columns = [
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                        'taker_buy_quote_volume', 'ignore'
                    ]
                    df['timestamp'] = pd.to_datetime(df['open_time'] / 1000, unit='s', utc=True)
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    all_data.append(df)
                except Exception as e:
                    print(f"⚠️ 加载失败 {file_path}: {e}")
    if not all_data:
        print("❌ 没有找到任何5分钟数据文件")
        return None
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df = merged_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
    print(f"✅ 合并完成")
    print(f"📊 总样本数: {len(merged_df)}")
    print(f"📅 时间范围: {merged_df['timestamp'].min()} 到 {merged_df['timestamp'].max()}")
    return merged_df

def main():
    parser = argparse.ArgumentParser(description='整合历史数据')
    parser.add_argument('--start_year', type=int, default=2023, help='开始年份')
    parser.add_argument('--start_month', type=int, default=1, help='开始月份')
    parser.add_argument('--end_year', type=int, default=2025, help='结束年份')
    parser.add_argument('--end_month', type=int, default=5, help='结束月份')
    parser.add_argument('--output_dir', type=str, default='data', help='输出目录')
    
    args = parser.parse_args()
    
    print("🚀 整合历史数据")
    print(f"📅 时间范围: {args.start_year}-{args.start_month:02d} 到 {args.end_year}-{args.end_month:02d}")
    print(f"📁 输出目录: {args.output_dir}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 合并1小时数据
    print("\n" + "="*50)
    print("📊 合并1小时数据")
    print("="*50)
    df_1h = merge_1h_data(args.start_year, args.start_month, args.end_year, args.end_month)
    
    if df_1h is not None:
        output_path = os.path.join(args.output_dir, f"ETHUSDT-1h-{args.start_year}-{args.start_month:02d}-to-{args.end_year}-{args.end_month:02d}.csv")
        df_1h.to_csv(output_path, index=False)
        print(f"✅ 1小时数据已保存: {output_path}")
    
    # 合并5分钟数据
    print("\n" + "="*50)
    print("📊 合并5分钟数据")
    print("="*50)
    df_5m = merge_5m_data_v2("/Users/qiutianyu/ETHUSDT-5m")
    if df_5m is not None:
        output_path = os.path.join(args.output_dir, f"ETHUSDT-5m-full.csv")
        df_5m.to_csv(output_path, index=False)
        print(f"✅ 5分钟数据已保存: {output_path}")
    
    print("\n✅ 数据整合完成！")

if __name__ == "__main__":
    main() 