#!/usr/bin/env python3
"""
将1小时K线数据合并为4小时数据
用于生成4小时趋势过滤信号
"""
import pandas as pd
from pathlib import Path

# 路径配置
DATA_DIR = Path("/Users/qiutianyu/data/processed")
INPUT_FILE = DATA_DIR / "merged_1h_2023_2025.parquet"
OUTPUT_FILE = DATA_DIR / "merged_4h_2023_2025.parquet"

def merge_1h_to_4h():
    print("📥 读取1小时K线数据...")
    df = pd.read_parquet(INPUT_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 按4小时分组（每4根1小时K线为一组）
    df['group'] = df.index // 4
    
    # 聚合为4小时数据
    agg_dict = {
        'timestamp': 'first',  # 取每组第一个时间戳
        'open': 'first',       # 开盘价
        'high': 'max',         # 最高价
        'low': 'min',          # 最低价
        'close': 'last',       # 收盘价
        'volume': 'sum'        # 成交量累加
    }
    
    print("🔄 合并为4小时数据...")
    df_4h = df.groupby('group').agg(agg_dict).reset_index(drop=True)
    
    # 保存4小时数据
    df_4h.to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ 4小时数据已保存: {OUTPUT_FILE}")
    print(f"数据行数: {len(df_4h)}")
    print("前5行示例:")
    print(df_4h.head())

if __name__ == "__main__":
    merge_1h_to_4h() 