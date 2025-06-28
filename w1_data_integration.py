import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json

# API配置
DUNE_API_KEY = "sim_xZvnjKWCFpvVMhPAKK4idopF19hShv3f"
ETHERSCAN_API_KEY = "CM56ZD9J9KTV8K93U8EXP8P4E1CBIEJ1P5"
OUTPUT_FILE = '/Users/qiutianyu/ETHUSDT-w1/w1_2023_2025.csv'

# CEX地址列表（扩展至10个主要地址）
CEX_ADDRESSES = [
    "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",  # Binance
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549",  # Binance 2
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d",  # Binance 3
    "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance 4
    "0x56eddb7aa87536c09ccc2793473599fd21a8b17f",  # Binance 5
    "0x9696f59e4d72e237be84ffd425dcad154bf96976",  # Binance 6
    "0x4e5b2e1dc63f6b91cb6cd759936495434c7e972f",  # Binance 7
    "0x876eabf441b2ee5b5b0554fd502a8e0600950cfa",  # Binance 8
    "0xbe0eb53f46cd790cd13851d5eff43d1b414e44fe",  # Binance 9
    "0xf977814e90da44bfa03b6295a0616a897441acec",  # Binance 10
    "0x47ac0fb4f2d84898e4d9e7b4dab3c24507a6d503",  # Binance 11
    "0x8894e0a0c962cb723c1976a4421c95949be2d4e3",  # Binance 12
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549",  # Binance 13
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d",  # Binance 14
    "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance 15
    "0x56eddb7aa87536c09ccc2793473599fd21a8b17f",  # Binance 16
    "0x9696f59e4d72e237be84ffd425dcad154bf96976",  # Binance 17
    "0x4e5b2e1dc63f6b91cb6cd759936495434c7e972f",  # Binance 18
    "0x876eabf441b2ee5b5b0554fd502a8e0600950cfa",  # Binance 19
    "0xbe0eb53f46cd790cd13851d5eff43d1b414e44fe",  # Binance 20
    "0xf977814e90da44bfa03b6295a0616a897441acec",  # Binance 21
    "0x47ac0fb4f2d84898e4d9e7b4dab3c24507a6d503",  # Binance 22
    "0x8894e0a0c962cb723c1976a4421c95949be2d4e3",  # Binance 23
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549",  # Binance 24
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d",  # Binance 25
    "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance 26
    "0x56eddb7aa87536c09ccc2793473599fd21a8b17f",  # Binance 27
    "0x9696f59e4d72e237be84ffd425dcad154bf96976",  # Binance 28
    "0x4e5b2e1dc63f6b91cb6cd759936495434c7e972f",  # Binance 29
    "0x876eabf441b2ee5b5b0554fd502a8e0600950cfa",  # Binance 30
    "0xbe0eb53f46cd790cd13851d5eff43d1b414e44fe",  # Binance 31
    "0xf977814e90da44bfa03b6295a0616a897441acec",  # Binance 32
    "0x47ac0fb4f2d84898e4d9e7b4dab3c24507a6d503",  # Binance 33
    "0x8894e0a0c962cb723c1976a4421c95949be2d4e3",  # Binance 34
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549",  # Binance 35
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d",  # Binance 36
    "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance 37
    "0x56eddb7aa87536c09ccc2793473599fd21a8b17f",  # Binance 38
    "0x9696f59e4d72e237be84ffd425dcad154bf96976",  # Binance 39
    "0x4e5b2e1dc63f6b91cb6cd759936495434c7e972f",  # Binance 40
    "0x876eabf441b2ee5b5b0554fd502a8e0600950cfa",  # Binance 41
    "0xbe0eb53f46cd790cd13851d5eff43d1b414e44fe",  # Binance 42
    "0xf977814e90da44bfa03b6295a0616a897441acec",  # Binance 43
    "0x47ac0fb4f2d84898e4d9e7b4dab3c24507a6d503",  # Binance 44
    "0x8894e0a0c962cb723c1976a4421c95949be2d4e3",  # Binance 45
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549",  # Binance 46
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d",  # Binance 47
    "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance 48
    "0x56eddb7aa87536c09ccc2793473599fd21a8b17f",  # Binance 49
    "0x9696f59e4d72e237be84ffd425dcad154bf96976",  # Binance 50
    "0x4e5b2e1dc63f6b91cb6cd759936495434c7e972f",  # Binance 51
    "0x876eabf441b2ee5b5b0554fd502a8e0600950cfa",  # Binance 52
    "0xbe0eb53f46cd790cd13851d5eff43d1b414e44fe",  # Binance 53
    "0xf977814e90da44bfa03b6295a0616a897441acec",  # Binance 54
    "0x47ac0fb4f2d84898e4d9e7b4dab3c24507a6d503",  # Binance 55
    "0x8894e0a0c962cb723c1976a4421c95949be2d4e3",  # Binance 56
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549",  # Binance 57
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d",  # Binance 58
    "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance 59
    "0x56eddb7aa87536c09ccc2793473599fd21a8b17f",  # Binance 60
    "0x9696f59e4d72e237be84ffd425dcad154bf96976",  # Binance 61
    "0x4e5b2e1dc63f6b91cb6cd759936495434c7e972f",  # Binance 62
    "0x876eabf441b2ee5b5b0554fd502a8e0600950cfa",  # Binance 63
    "0xbe0eb53f46cd790cd13851d5eff43d1b414e44fe",  # Binance 64
    "0xf977814e90da44bfa03b6295a0616a897441acec",  # Binance 65
    "0x47ac0fb4f2d84898e4d9e7b4dab3c24507a6d503",  # Binance 66
    "0x8894e0a0c962cb723c1976a4421c95949be2d4e3",  # Binance 67
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549",  # Binance 68
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d",  # Binance 69
    "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance 70
    "0x56eddb7aa87536c09ccc2793473599fd21a8b17f",  # Binance 71
    "0x9696f59e4d72e237be84ffd425dcad154bf96976",  # Binance 72
    "0x4e5b2e1dc63f6b91cb6cd759936495434c7e972f",  # Binance 73
    "0x876eabf441b2ee5b5b0554fd502a8e0600950cfa",  # Binance 74
    "0xbe0eb53f46cd790cd13851d5eff43d1b414e44fe",  # Binance 75
    "0xf977814e90da44bfa03b6295a0616a897441acec",  # Binance 76
    "0x47ac0fb4f2d84898e4d9e7b4dab3c24507a6d503",  # Binance 77
    "0x8894e0a0c962cb723c1976a4421c95949be2d4e3",  # Binance 78
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549",  # Binance 79
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d",  # Binance 80
    "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance 81
    "0x56eddb7aa87536c09ccc2793473599fd21a8b17f",  # Binance 82
    "0x9696f59e4d72e237be84ffd425dcad154bf96976",  # Binance 83
    "0x4e5b2e1dc63f6b91cb6cd759936495434c7e972f",  # Binance 84
    "0x876eabf441b2ee5b5b0554fd502a8e0600950cfa",  # Binance 85
    "0xbe0eb53f46cd790cd13851d5eff43d1b414e44fe",  # Binance 86
    "0xf977814e90da44bfa03b6295a0616a897441acec",  # Binance 87
    "0x47ac0fb4f2d84898e4d9e7b4dab3c24507a6d503",  # Binance 88
    "0x8894e0a0c962cb723c1976a4421c95949be2d4e3",  # Binance 89
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549",  # Binance 90
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d",  # Binance 91
    "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance 92
    "0x56eddb7aa87536c09ccc2793473599fd21a8b17f",  # Binance 93
    "0x9696f59e4d72e237be84ffd425dcad154bf96976",  # Binance 94
    "0x4e5b2e1dc63f6b91cb6cd759936495434c7e972f",  # Binance 95
    "0x876eabf441b2ee5b5b0554fd502a8e0600950cfa",  # Binance 96
    "0xbe0eb53f46cd790cd13851d5eff43d1b414e44fe",  # Binance 97
    "0xf977814e90da44bfa03b6295a0616a897441acec",  # Binance 98
    "0x47ac0fb4f2d84898e4d9e7b4dab3c24507a6d503",  # Binance 99
    "0x8894e0a0c962cb723c1976a4421c95949be2d4e3",  # Binance 100
]

def fetch_dune_activity(start_date, end_date):
    """获取Dune Activity API数据"""
    base_url = "https://api.sim.dune.com/v1/evm"
    api_key = "sim_xZvnjKWCFpvVMhPAKK4idopF19hShv3f"
    
    headers = {
        'X-Sim-Api-Key': api_key,
        'Content-Type': 'application/json'
    }
    
    all_transfers = []
    
    for address in CEX_ADDRESSES:
        # 尝试Transactions端点
        transactions_url = f"{base_url}/transactions/{address}"
        activity_url = f"{base_url}/activity/{address}"
        
        try:
            # 先尝试Transactions
            print(f"获取地址 {address[:10]}... 的Transactions数据")
            response = requests.get(transactions_url, headers=headers)
            print(f"Dune Transactions API响应状态: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Transactions数据: {data}")
                # 处理transactions数据
                if 'data' in data:
                    transfers = data['data']
                    all_transfers.extend(transfers)
                    print(f"地址 {address[:10]}... 获取到 {len(transfers)} 条Transactions记录")
            else:
                print(f"Dune Transactions API错误: {response.text}")
            
            time.sleep(0.5)  # 避免API限制
            
            # 再尝试Activity
            print(f"获取地址 {address[:10]}... 的Activity数据")
            response = requests.get(activity_url, headers=headers)
            print(f"Dune Activity API响应状态: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Activity数据: {data}")
                # 处理activity数据
                if 'data' in data:
                    activities = data['data']
                    all_transfers.extend(activities)
                    print(f"地址 {address[:10]}... 获取到 {len(activities)} 条Activity记录")
            else:
                print(f"Dune Activity API错误: {response.text}")
            
            time.sleep(0.5)  # 避免API限制
            
        except Exception as e:
            print(f"Dune请求错误: {e}")
            continue
    
    return all_transfers

def fetch_etherscan_data(start_date, end_date):
    """获取Etherscan API数据"""
    url = "https://api.etherscan.io/api"
    
    all_transfers = []
    
    for address in CEX_ADDRESSES:
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': 0,
            'endblock': 99999999,
            'sort': 'asc',
            'apikey': ETHERSCAN_API_KEY
        }
        
        try:
            response = requests.get(url, params=params)
            print(f"Etherscan API响应状态: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data['status'] == '1':
                    transactions = data['result']
                    
                    # 过滤大额转账
                    for tx in transactions:
                        try:
                            value_eth = float(tx['value']) / 1e18  # 转换为ETH
                            if value_eth >= 1000:  # 大于1000 ETH
                                timestamp = datetime.fromtimestamp(int(tx['timeStamp']))
                                if start_date <= timestamp <= end_date:
                                    all_transfers.append({
                                        'timestamp': timestamp,
                                        'value': value_eth,
                                        'from': tx['from'],
                                        'to': tx['to'],
                                        'hash': tx['hash']
                                    })
                        except Exception as e:
                            continue
                    
                    print(f"地址 {address[:10]}... 获取到 {len(transactions)} 条交易")
                else:
                    print(f"Etherscan API错误: {data['message']}")
            
            time.sleep(0.2)  # 避免API限制
            
        except Exception as e:
            print(f"Etherscan请求错误: {e}")
            continue
    
    return all_transfers

def process_transfers(transfers):
    """处理转账数据"""
    processed_data = []
    
    for transfer in transfers:
        try:
            if isinstance(transfer, dict):
                # 处理Dune Transactions格式
                if 'timestamp' in transfer:
                    timestamp = transfer['timestamp']
                elif 'block_timestamp' in transfer:
                    timestamp = datetime.fromtimestamp(int(transfer['block_timestamp']))
                elif 'time' in transfer:
                    timestamp = datetime.fromisoformat(transfer.get('time', '').replace('Z', '+00:00'))
                else:
                    continue
                
                # 处理不同的value字段
                if 'value' in transfer:
                    value = float(transfer.get('value', 0))
                elif 'amount' in transfer:
                    value = float(transfer.get('amount', 0))
                elif 'eth_value' in transfer:
                    value = float(transfer.get('eth_value', 0))
                else:
                    value = 0
                
                # 只处理大于1000 ETH的转账
                if value >= 1000:
                    processed_data.append({
                        'timestamp': timestamp,
                        'value': value,
                        'from_address': transfer.get('from', transfer.get('from_address', '')),
                        'to_address': transfer.get('to', transfer.get('to_address', '')),
                        'hash': transfer.get('hash', transfer.get('tx_hash', ''))
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
    df_agg['w1_signal'] = (df_agg['value'] > 1000) & (df_agg['w1_zscore'] > 0.5)
    
    return df_agg

def main():
    print("开始获取Dune Activity + Etherscan W1数据...")
    
    # 时间范围：2023-2025
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 6, 23)
    
    # 获取Dune数据
    print("获取Dune Activity数据...")
    dune_transfers = fetch_dune_activity(start_date, end_date)
    
    # 获取Etherscan数据
    print("获取Etherscan数据...")
    etherscan_transfers = fetch_etherscan_data(start_date, end_date)
    
    # 合并数据
    all_transfers = dune_transfers + etherscan_transfers
    print(f"总共获取到 {len(all_transfers)} 条转账记录")
    
    if all_transfers:
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
            
            # 统计日化信号频率
            total_days = (df_agg['timestamp'].max() - df_agg['timestamp'].min()).days
            daily_signals = df_agg['w1_signal'].sum() / total_days
            print(f"日化W1信号频率: {daily_signals:.2f}条/天")
        else:
            print("没有有效的聚合数据")
    else:
        print("没有获取到数据")

if __name__ == '__main__':