#!/usr/bin/env python3
"""
合并所有2.5年的15分钟K线数据
从 /Users/qiutianyu/ETHUSDT-15m/ 目录读取所有月份数据
"""

import pandas as pd
import glob
import os
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_timestamp(timestamp_str):
    """智能转换时间戳，支持毫秒和微秒"""
    try:
        timestamp = int(timestamp_str)
        # 如果时间戳大于1e12，认为是毫秒，否则是微秒
        if timestamp > 1e12:
            return pd.to_datetime(timestamp, unit='ms')
        else:
            return pd.to_datetime(timestamp, unit='us')
    except:
        return pd.NaT

def merge_15m_data():
    """合并所有15分钟数据"""
    
    # 数据目录
    data_dir = Path("/Users/qiutianyu/ETHUSDT-15m/")
    output_file = "data/ETHUSDT-15m-2023-01-to-2025-05.csv"
    
    # 获取所有月份目录
    month_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("ETHUSDT-15m-")])
    
    logger.info(f"找到 {len(month_dirs)} 个月份目录")
    
    all_data = []
    
    for month_dir in month_dirs:
        csv_file = month_dir / f"{month_dir.name}.csv"
        if csv_file.exists():
            logger.info(f"处理 {csv_file}")
            
            try:
                # 读取CSV文件
                df = pd.read_csv(csv_file, header=None)
                
                # 设置列名
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                             'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                             'taker_buy_quote', 'ignore']
                
                # 智能转换时间戳
                df['timestamp'] = df['timestamp'].apply(convert_timestamp)
                df['close_time'] = df['close_time'].apply(convert_timestamp)
                
                # 过滤无效时间戳
                df = df.dropna(subset=['timestamp'])
                
                # 只保留需要的列
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']]
                
                all_data.append(df)
                logger.info(f"  - 添加 {len(df)} 行数据")
                
            except Exception as e:
                logger.error(f"处理 {csv_file} 时出错: {e}")
                continue
    
    if not all_data:
        logger.error("没有找到任何数据文件")
        return
    
    # 合并所有数据
    logger.info("合并所有数据...")
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # 按时间排序
    merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
    
    # 去重
    merged_df = merged_df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    
    logger.info(f"合并完成: {len(merged_df)} 行数据")
    logger.info(f"时间范围: {merged_df['timestamp'].min()} 到 {merged_df['timestamp'].max()}")
    
    # 保存到文件
    os.makedirs("data", exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    logger.info(f"数据已保存到: {output_file}")
    
    # 显示统计信息
    logger.info("\n=== 数据统计 ===")
    logger.info(f"总行数: {len(merged_df):,}")
    logger.info(f"时间跨度: {(merged_df['timestamp'].max() - merged_df['timestamp'].min()).days} 天")
    logger.info(f"平均每日K线数: {len(merged_df) / ((merged_df['timestamp'].max() - merged_df['timestamp'].min()).days):.1f}")
    
    return merged_df

if __name__ == "__main__":
    merge_15m_data() 