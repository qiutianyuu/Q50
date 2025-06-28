import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json

# IntoTheBlock API配置
API_KEY = "BS3utDEquhRHvnbv0Kuvjcz6EiUYAovcSxJtNTTRRkFcr8MhTOqTfFWMV0CEDKLQ"
SECRET_KEY = "sjqTZkT56nCEKardas6pIypF2dgPapYogiQ0e3pdR70NaJlfkhXJucrI5oMzunag"
OUTPUT_FILE = '/Users/qiutianyu/ETHUSDT-w1/intotheblock_w1_2023_2025.csv'

def fetch_intotheblock_data(start_date, end_date):
    """获取IntoTheBlock大额转账数据"""
    url = "https://api.intotheblock.com/analytics/whales/transfers"
    
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    params = {
        'symbol': 'ETH',
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'min_amount': 3000,  # 最小3000 ETH
        'type': 'all'  # 包括CEX和DEX
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        print(f"API响应状态: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"获取到 {len(data.get('data', []))} 条记录")
            return data.get('data', [])
        else:
            print(f"API错误: {response.text}")
            return []
    except Exception as e:
        print(f"请求错误: {e}")
        return []

def process_transfers(transfers):
    """处理转账数据"""
    processed_data = []
    
    for transfer in transfers:
        try:
            timestamp = datetime.fromisoformat(transfer['timestamp'].replace('Z', '+00:00'))
            value = float(transfer['amount'])
            
            processed_data.append({
                'timestamp': timestamp,
                'value': value,
                'from_address': transfer.get('from_address', ''),
                'to_address': transfer.get('to_address', ''),
                'type': transfer.get('type', 'unknown')
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
        'timestamp': 'count'
    }).reset_index()
    df_agg.columns = ['timestamp', 'value', 'count']
    
    # 计算zscore
    df_agg['w1_zscore'] = (df_agg['value'] - df_agg['value'].mean()) / df_agg['value'].std()
    df_agg['w1_signal'] = (df_agg['value'] > 3000) & (df_agg['w1_zscore'] > 1.5)
    
    return df_agg

def main():
    print("开始获取IntoTheBlock W1数据...")
    
    # 时间范围：2023-2025
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    # 按月获取数据（避免API限制）
    all_transfers = []
    current_date = start_date
    
    while current_date < end_date:
        month_end = min(current_date + timedelta(days=30), end_date)
        print(f"获取 {current_date.strftime('%Y-%m')} 的数据...")
        
        transfers = fetch_intotheblock_data(current_date, month_end)
        all_transfers.extend(transfers)
        
        current_date = month_end
        time.sleep(1)  # 避免API限制
    
    if all_transfers:
        print(f"总共获取到 {len(all_transfers)} 条转账记录")
        
        # 处理数据
        processed_data = process_transfers(all_transfers)
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
        print("尝试使用免费API...")
        
        # 尝试免费API
        try:
            url = "https://app.intotheblock.com/api/v1/whales/transfers"
            response = requests.get(url, params={'symbol': 'ETH', 'limit': 1000})
            print(f"免费API响应: {response.status_code}")
            if response.status_code == 200:
                print("免费API可用，但需要手动处理数据")
        except Exception as e:
            print(f"免费API错误: {e}")

if __name__ == '__main__':
    main() 