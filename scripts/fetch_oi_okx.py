#!/usr/bin/env python3
"""
从OKX API拉取ETH-USDT的OI数据
"""
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path

# OKX API配置
BASE_URL = "https://www.okx.com"
INSTRUMENT_ID = "ETH-USDT-SWAP"  # ETH永续合约
OUTPUT_FILE = Path("data/funding_oi_1h_2021_2022.parquet")

def fetch_oi_data(start_time, end_time):
    """拉取指定时间范围的OI数据"""
    url = f"{BASE_URL}/api/v5/public/open-interest"
    
    params = {
        'instId': INSTRUMENT_ID,
        'period': '1H',  # 1小时数据
        'start': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end': end_time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print(f"API响应: {data}")  # 调试信息
        
        if data['code'] == '0':
            return data['data']
        else:
            print(f"API错误: {data}")
            return []
            
    except Exception as e:
        print(f"请求失败: {e}")
        return []

def main():
    print("⏬ 从OKX拉取OI数据...")
    
    # 先测试一个小的日期范围
    test_start = datetime(2021, 1, 1)
    test_end = datetime(2021, 1, 2)
    
    print(f"测试拉取 {test_start.strftime('%Y-%m-%d')} 到 {test_end.strftime('%Y-%m-%d')}")
    test_data = fetch_oi_data(test_start, test_end)
    
    if test_data:
        print(f"测试数据样本: {test_data[0] if test_data else 'None'}")
        print(f"数据字段: {list(test_data[0].keys()) if test_data else 'None'}")
    else:
        print("❌ 测试失败，无法获取数据")
        return
    
    # 如果测试成功，继续完整拉取
    print("\n开始完整拉取...")
    
    # 时间范围：2021-01-01 到 2022-12-31
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2022, 12, 31, 23, 59, 59)
    
    all_data = []
    current_date = start_date
    
    # 分批拉取，每次30天
    batch_days = 30
    
    while current_date < end_date:
        batch_end = min(current_date + timedelta(days=batch_days), end_date)
        
        print(f"拉取 {current_date.strftime('%Y-%m-%d')} 到 {batch_end.strftime('%Y-%m-%d')}")
        
        data = fetch_oi_data(current_date, batch_end)
        
        if data:
            all_data.extend(data)
            print(f"获取 {len(data)} 条记录")
        else:
            print("该时间段无数据")
        
        current_date = batch_end + timedelta(days=1)
        time.sleep(0.5)  # 避免请求过快
    
    if not all_data:
        print("❌ 未获取到任何数据")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    
    # 检查数据字段
    print(f"数据字段: {list(df.columns)}")
    print(f"数据样本:\n{df.head()}")
    
    # 根据实际字段名处理
    if 'ts' in df.columns:
        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # 查找OI字段
    oi_column = None
    for col in ['oi', 'openInterest', 'open_interest']:
        if col in df.columns:
            oi_column = col
            break
    
    if oi_column is None:
        print(f"❌ 未找到OI字段，可用字段: {list(df.columns)}")
        return
    
    df['oi'] = df[oi_column].astype(float)
    
    # 只保留需要的列
    df = df[['timestamp', 'oi']].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ 总共获取 {len(df)} 条OI记录")
    print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    
    # 读取现有的funding数据
    if OUTPUT_FILE.exists():
        print("📖 读取现有funding数据...")
        existing_df = pd.read_parquet(OUTPUT_FILE)
        print(f"现有funding记录: {len(existing_df)}")
        
        # 合并funding和OI数据
        merged_df = existing_df.merge(df, on='timestamp', how='left')
        merged_df['oi'] = merged_df['oi'].fillna(0)
        
        print(f"合并后记录: {len(merged_df)}")
    else:
        print("⚠️ 未找到现有funding数据，只保存OI")
        merged_df = df.copy()
        merged_df['funding'] = 0  # 默认值
    
    # 保存
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(OUTPUT_FILE, index=False, compression='zstd')
    print(f"✅ 已保存到 {OUTPUT_FILE}")
    
    # 显示样本数据
    print("\n📊 样本数据:")
    print(merged_df.head())
    print(f"\nOI统计:")
    print(f"  平均值: {merged_df['oi'].mean():,.0f}")
    print(f"  最大值: {merged_df['oi'].max():,.0f}")
    print(f"  最小值: {merged_df['oi'].min():,.0f}")

if __name__ == "__main__":
    main() 