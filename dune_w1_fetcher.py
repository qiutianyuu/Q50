import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json

# Dune Analytics配置
DUNE_API_KEY = "YOUR_DUNE_API_KEY"  # 需要用户提供
OUTPUT_FILE = '/Users/qiutianyu/ETHUSDT-w1/dune_w1_2023_2025.csv'

def fetch_dune_data(start_date, end_date):
    """获取Dune Analytics大额转账数据"""
    # Dune Analytics查询ID (需要先创建查询)
    query_id = "YOUR_QUERY_ID"  # 需要用户提供
    
    url = f"https://api.dune.com/api/v1/query/{query_id}/results"
    
    headers = {
        'X-Dune-API-Key': DUNE_API_KEY,
        'Content-Type': 'application/json'
    }
    
    params = {
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'min_amount': 1000  # 最小1000 ETH
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        print(f"API响应状态: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"获取到 {len(data.get('result', {}).get('rows', []))} 条记录")
            return data.get('result', {}).get('rows', [])
        else:
            print(f"API错误: {response.text}")
            return []
    except Exception as e:
        print(f"请求错误: {e}")
        return []

def create_dune_query():
    """创建Dune Analytics查询的SQL示例"""
    sql_query = """
    -- ETH大额转账查询 (2023-2025)
    SELECT 
        DATE_TRUNC('hour', block_time) as timestamp,
        SUM(value) as total_value,
        COUNT(*) as transfer_count,
        AVG(value) as avg_value
    FROM ethereum.transactions 
    WHERE 
        block_time >= '2023-01-01' 
        AND block_time <= '2025-12-31'
        AND value >= 1000000000000000000000  -- 1000 ETH in wei
        AND success = true
    GROUP BY DATE_TRUNC('hour', block_time)
    ORDER BY timestamp
    """
    
    print("Dune Analytics查询SQL:")
    print(sql_query)
    print("\n使用步骤:")
    print("1. 访问 https://dune.com/")
    print("2. 创建新查询")
    print("3. 粘贴上述SQL")
    print("4. 运行查询并获取查询ID")
    print("5. 更新脚本中的query_id变量")

def process_transfers(transfers):
    """处理转账数据"""
    processed_data = []
    
    for transfer in transfers:
        try:
            timestamp = datetime.fromisoformat(transfer['timestamp'].replace('Z', '+00:00'))
            value = float(transfer['total_value']) / 1e18  # 转换为ETH
            
            processed_data.append({
                'timestamp': timestamp,
                'value': value,
                'count': transfer.get('transfer_count', 1)
            })
        except Exception as e:
            print(f"处理记录错误: {e}")
            continue
    
    return processed_data

def aggregate_to_4h(transfers):
    """按4H聚合数据"""
    if not transfers:
        return pd.DataFrame()
    
    df = pd.DataFrame(transfers)
    df = df.sort_values('timestamp')
    
    # 按4H聚合
    df['4h_time'] = df['timestamp'].dt.floor('4H')
    df_agg = df.groupby('4h_time').agg({
        'value': 'sum',
        'count': 'sum'
    }).reset_index()
    df_agg.columns = ['timestamp', 'value', 'count']
    
    # 计算zscore
    df_agg['w1_zscore'] = (df_agg['value'] - df_agg['value'].mean()) / df_agg['value'].std()
    df_agg['w1_signal'] = (df_agg['value'] > 1000) & (df_agg['w1_zscore'] > 0.5)
    
    return df_agg

def main():
    print("开始获取Dune Analytics W1数据...")
    print("注意：需要提供有效的Dune Analytics API密钥和查询ID")
    
    # 检查API密钥
    if DUNE_API_KEY == "YOUR_DUNE_API_KEY":
        print("请先设置Dune Analytics API密钥")
        print("1. 访问 https://dune.com/")
        print("2. 注册并获取API密钥")
        print("3. 创建ETH大额转账查询")
        print("4. 更新脚本中的API_KEY和query_id变量")
        create_dune_query()
        return
    
    # 时间范围：2023-2025
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    # 获取数据
    transfers = fetch_dune_data(start_date, end_date)
    
    if transfers:
        print(f"总共获取到 {len(transfers)} 条转账记录")
        
        # 处理数据
        processed_data = process_transfers(transfers)
        print(f"处理后的记录: {len(processed_data)} 条")
        
        # 聚合到4H
        df_agg = aggregate_to_4h(processed_data)
        
        if not df_agg.empty:
            # 保存到CSV
            df_agg.to_csv(OUTPUT_FILE, index=False)
            print(f"数据已保存到: {OUTPUT_FILE}")
            print(f"聚合后数据点: {len(df_agg)}")
            print(f"W1信号数: {df_agg['w1_signal'].sum()}")
            print(f"数据范围: {df_agg['timestamp'].min()} 到 {df_agg['timestamp'].max()}")
        else:
            print("没有有效的聚合数据")
    else:
        print("没有获取到数据")

if __name__ == '__main__':
    main() 