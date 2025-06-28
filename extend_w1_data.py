import pandas as pd
import requests
import time
from datetime import datetime, timedelta

# 扩展的W1地址
W1_ADDRESSES = [
    "0x6f88ff33196dbd1b99020d9913efde5c695c02d5",  # 原有地址
    "0x4b2b1dc2e7b9c5c5392e54b6a45fd1b769d2e7e7",  # Huobi
    "0x1c4b70a3968436b9a0a9cf5205c787eb81bb558c"   # Bybit
]

API_KEY = "CM56ZD9J9KTV8K93U8EXP8P4E1CBIEJ1P5"  # Etherscan API
OUTPUT_FILE = '/Users/qiutianyu/ETHUSDT-w1/etherscan_w1_2023_2025.csv'

def fetch_etherscan_data(address, start_date, end_date):
    """获取Etherscan数据"""
    url = "https://api.etherscan.io/api"
    
    # 转换日期为时间戳
    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())
    
    params = {
        'module': 'account',
        'action': 'txlist',
        'address': address,
        'startblock': 0,
        'endblock': 99999999,
        'page': 1,
        'offset': 10000,
        'sort': 'asc',
        'apikey': API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['status'] == '1':
            transactions = data['result']
            # 过滤时间范围
            filtered_txs = []
            for tx in transactions:
                tx_time = int(tx['timeStamp'])
                if start_ts <= tx_time <= end_ts:
                    filtered_txs.append({
                        'timestamp': datetime.fromtimestamp(tx_time),
                        'value': float(tx['value']) / 1e18,  # 转换为ETH
                        'address': address
                    })
            return filtered_txs
        else:
            print(f"API错误: {data['message']}")
            return []
    except Exception as e:
        print(f"请求错误: {e}")
        return []

def main():
    print("开始扩展W1数据...")
    
    # 时间范围：2023-2025
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    all_transactions = []
    
    for address in W1_ADDRESSES:
        print(f"获取地址 {address} 的数据...")
        txs = fetch_etherscan_data(address, start_date, end_date)
        all_transactions.extend(txs)
        print(f"获取到 {len(txs)} 条交易")
        time.sleep(1)  # 避免API限制
    
    if all_transactions:
        # 转换为DataFrame
        df = pd.DataFrame(all_transactions)
        df = df.sort_values('timestamp')
        
        # 按4H聚合
        df['4h_time'] = df['timestamp'].dt.floor('4H')
        df_agg = df.groupby('4h_time').agg({
            'value': 'sum',
            'address': 'count'
        }).reset_index()
        df_agg.columns = ['timestamp', 'value', 'count']
        
        # 计算zscore
        df_agg['w1_zscore'] = (df_agg['value'] - df_agg['value'].mean()) / df_agg['value'].std()
        df_agg['w1_signal'] = (df_agg['value'] > 5000) & (df_agg['w1_zscore'] > 1.0)
        
        # 保存到CSV
        df_agg.to_csv(OUTPUT_FILE, index=False)
        print(f"数据已保存到: {OUTPUT_FILE}")
        print(f"总交易数: {len(df)}")
        print(f"聚合后数据点: {len(df_agg)}")
        print(f"W1信号数: {df_agg['w1_signal'].sum()}")
    else:
        print("没有获取到数据")

if __name__ == '__main__':
    main() 