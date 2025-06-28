#!/usr/bin/env python3
"""
合并2021-2022年15分钟K线数据
从 /Users/qiutianyu/ETHUSDT-15m-2021-2022/ 目录递归读取所有CSV文件
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

def merge_15m_2021_2022_data():
    """合并2021-2022年15分钟数据"""
    
    # 数据目录
    data_dir = Path("/Users/qiutianyu/ETHUSDT-15m-2021-2022/")
    output_file = "data/merged_15m_2021_2022.parquet"
    
    # 递归获取所有CSV文件
    csv_pattern = str(data_dir / "**/*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    logger.info(f"找到 {len(csv_files)} 个CSV文件")
    
    all_data = []
    
    for csv_file in sorted(csv_files):
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
    
    # 保存到Parquet文件（zstd压缩）
    os.makedirs("data", exist_ok=True)
    merged_df.to_parquet(output_file, compression='zstd', index=False)
    logger.info(f"数据已保存到: {output_file}")
    
    # 显示统计信息
    logger.info("\n=== 数据统计 ===")
    logger.info(f"总行数: {len(merged_df):,}")
    logger.info(f"时间跨度: {(merged_df['timestamp'].max() - merged_df['timestamp'].min()).days} 天")
    logger.info(f"平均每日K线数: {len(merged_df) / ((merged_df['timestamp'].max() - merged_df['timestamp'].min()).days):.1f}")
    
    return merged_df

if __name__ == "__main__":
    merge_15m_2021_2022_data() 