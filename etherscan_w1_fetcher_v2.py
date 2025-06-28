#!/usr/bin/env python3
"""
Etherscan W1 Data Fetcher v2
获取4个CEX热钱包地址的ETH转账数据 (>10000 ETH)
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import sqlite3
import os

class EtherscanW1Fetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.etherscan.io/api"
        
        # CEX热钱包地址
        self.cex_addresses = {
            'binance_1': '0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be',
            'binance_2': '0x28c6c06298d514db089934071355e5743bf21d60',
            'okx': '0x5e5f48a684a8d426a7d6d9d1d9464c1a68e119c2',
            'coinbase': '0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43'
        }
        
        # 创建缓存数据库
        self.cache_db = "etherscan_cache_v2.db"
        self.init_cache()
    
    def init_cache(self):
        """初始化缓存数据库"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        # 创建转账记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transfers (
                hash TEXT PRIMARY KEY,
                from_address TEXT,
                to_address TEXT,
                value REAL,
                timestamp INTEGER,
                block_number INTEGER,
                address_type TEXT
            )
        ''')
        
        # 创建聚合数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS w1_aggregated (
                timestamp TEXT PRIMARY KEY,
                value REAL,
                tx_count INTEGER,
                w1_zscore REAL,
                w1_signal INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def fetch_transfers(self, address, start_block=0, end_block=99999999):
        """获取地址的转账记录"""
        url = f"{self.base_url}"
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': start_block,
            'endblock': end_block,
            'sort': 'asc',
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == '1':
                return data['result']
            else:
                print(f"API Error: {data['message']}")
                return []
                
        except Exception as e:
            print(f"Error fetching transfers for {address}: {e}")
            return []
    
    def process_transfers(self, transfers, address_type):
        """处理转账记录"""
        processed = []
        
        for tx in transfers:
            # 只处理ETH转账 (>10000 ETH)
            value_eth = float(tx['value']) / 1e18  # 转换为ETH
            
            if value_eth >= 10000:
                processed.append({
                    'hash': tx['hash'],
                    'from_address': tx['from'],
                    'to_address': tx['to'],
                    'value': value_eth,
                    'timestamp': int(tx['timeStamp']),
                    'block_number': int(tx['blockNumber']),
                    'address_type': address_type
                })
        
        return processed
    
    def save_to_cache(self, transfers):
        """保存转账记录到缓存"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        for tx in transfers:
            cursor.execute('''
                INSERT OR REPLACE INTO transfers 
                (hash, from_address, to_address, value, timestamp, block_number, address_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (tx['hash'], tx['from_address'], tx['to_address'], 
                  tx['value'], tx['timestamp'], tx['block_number'], tx['address_type']))
        
        conn.commit()
        conn.close()
    
    def aggregate_w1_data(self, start_date='2023-01-01', end_date='2025-12-31'):
        """聚合W1数据到10分钟间隔"""
        conn = sqlite3.connect(self.cache_db)
        
        # 读取所有转账记录
        query = '''
            SELECT timestamp, value FROM transfers 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        '''
        
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        df = pd.read_sql_query(query, conn, params=(start_ts, end_ts))
        conn.close()
        
        if df.empty:
            print("No transfer data found")
            return pd.DataFrame()
        
        # 转换时间戳
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
        
        # 10分钟聚合
        df_10min = df.resample('10T').agg({
            'value': 'sum',
            'timestamp': 'count'
        }).rename(columns={'timestamp': 'tx_count'})
        
        # 计算z-score (基于过去24小时)
        df_10min['w1_zscore'] = df_10min['value'].rolling(window=144).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
        )
        
        # 生成信号 (z-score > 2.5)
        df_10min['w1_signal'] = (df_10min['w1_zscore'] > 2.5).astype(int)
        
        # 重置索引
        df_10min.reset_index(inplace=True)
        df_10min['timestamp'] = df_10min['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return df_10min[['timestamp', 'value', 'tx_count', 'w1_zscore', 'w1_signal']]
    
    def fetch_all_data(self):
        """获取所有CEX地址的转账数据"""
        print("开始获取Etherscan W1数据...")
        
        # 定义时间范围 (2023-2025)
        start_blocks = {
            '2023': 16308193,  # 2023年1月1日左右
            '2024': 19000000,  # 2024年1月1日左右
            '2025': 21000000   # 2025年1月1日左右
        }
        
        end_blocks = {
            '2023': 18999999,
            '2024': 20999999,
            '2025': 99999999
        }
        
        total_transfers = []
        
        for address_name, address in self.cex_addresses.items():
            print(f"获取 {address_name} ({address}) 的转账数据...")
            
            for year in ['2023', '2024', '2025']:
                print(f"  处理 {year} 年数据...")
                
                transfers = self.fetch_transfers(
                    address, 
                    start_blocks[year], 
                    end_blocks[year]
                )
                
                processed = self.process_transfers(transfers, address_name)
                total_transfers.extend(processed)
                
                print(f"    {year}年: 找到 {len(processed)} 笔大额转账")
                
                # API限速
                time.sleep(0.2)
        
        print(f"总共找到 {len(total_transfers)} 笔大额转账")
        
        # 保存到缓存
        self.save_to_cache(total_transfers)
        
        # 聚合数据
        print("聚合W1数据...")
        w1_data = self.aggregate_w1_data()
        
        return w1_data
    
    def save_w1_data(self, w1_data, output_path):
        """保存W1数据到CSV"""
        w1_data.to_csv(output_path, index=False)
        print(f"W1数据已保存到: {output_path}")
        print(f"数据范围: {w1_data['timestamp'].min()} 到 {w1_data['timestamp'].max()}")
        print(f"信号数量: {w1_data['w1_signal'].sum()}")
        print(f"总记录数: {len(w1_data)}")

def main():
    # 使用提供的API key
    api_key = "CM56ZD9J9KTV8K93U8EXP8P4E1CBIEJ1P5"
    
    # 创建输出目录
    output_dir = "/Users/qiutianyu/ETHUSDT-w1"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化fetcher
    fetcher = EtherscanW1Fetcher(api_key)
    
    # 获取数据
    w1_data = fetcher.fetch_all_data()
    
    if not w1_data.empty:
        # 保存数据
        output_path = os.path.join(output_dir, "etherscan_w1_2023_2025.csv")
        fetcher.save_w1_data(w1_data, output_path)
    else:
        print("没有获取到数据")

if __name__ == "__main__":
    main() 