import requests
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import time
import json

class EtherscanDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.etherscan.io/api"
        self.db_path = "etherscan_cache.db"
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS whale_transfers (
                timestamp INTEGER,
                from_address TEXT,
                to_address TEXT,
                value REAL,
                gas_used INTEGER,
                is_cex INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON whale_transfers(timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def get_whale_transfers(self, start_time, end_time, min_value=5000):
        """获取大额转账数据"""
        print(f"Fetching whale transfers from {start_time} to {end_time}")
        
        transfers = []
        current_time = start_time
        
        while current_time < end_time:
            # 获取区块范围
            start_block = self.get_block_by_timestamp(int(current_time.timestamp()))
            end_block = self.get_block_by_timestamp(int((current_time + timedelta(hours=1)).timestamp()))
            
            # 获取转账
            batch_transfers = self.get_transfers_in_range(start_block, end_block, min_value)
            transfers.extend(batch_transfers)
            
            current_time += timedelta(hours=1)
            time.sleep(0.2)  # 避免API限制
        
        # 保存到数据库
        self.save_transfers(transfers)
        
        return transfers
    
    def get_block_by_timestamp(self, timestamp):
        """根据时间戳获取区块号"""
        url = f"{self.base_url}?module=block&action=getblocknobytime&timestamp={timestamp}&closest=before&apikey={self.api_key}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if data['status'] == '1':
                return int(data['result'])
            else:
                print(f"Error getting block: {data['message']}")
                return 0
        except Exception as e:
            print(f"Error: {e}")
            return 0
    
    def get_transfers_in_range(self, start_block, end_block, min_value):
        """获取指定区块范围内的转账"""
        transfers = []
        
        # 获取大额转账事件
        url = f"{self.base_url}?module=account&action=txlist&address=0x0000000000000000000000000000000000000000&startblock={start_block}&endblock={end_block}&sort=desc&apikey={self.api_key}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if data['status'] == '1':
                for tx in data['result']:
                    # 过滤大额转账
                    value_eth = float(tx['value']) / 1e18
                    if value_eth >= min_value:
                        # 判断是否为CEX
                        is_cex = self.is_cex_address(tx['to'])
                        
                        transfers.append({
                            'timestamp': int(tx['timeStamp']),
                            'from_address': tx['from'],
                            'to_address': tx['to'],
                            'value': value_eth,
                            'gas_used': int(tx['gasUsed']),
                            'is_cex': is_cex
                        })
            
        except Exception as e:
            print(f"Error fetching transfers: {e}")
        
        return transfers
    
    def is_cex_address(self, address):
        """判断是否为交易所地址"""
        cex_addresses = {
            '0x21a31ee1afc51d94c2efccaa2092ad1028285549',  # Binance
            '0x28c6c06298d514db089934071355e5743bf21d60',  # Binance
            '0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be',  # Binance
            '0xdac17f958d2ee523a2206206994597c13d831ec7',  # Tether
            '0xa0b86a33e6441b8c4c8c8c8c8c8c8c8c8c8c8c8c',  # Coinbase
        }
        
        return address.lower() in cex_addresses
    
    def save_transfers(self, transfers):
        """保存转账数据到数据库"""
        if not transfers:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for transfer in transfers:
            cursor.execute('''
                INSERT OR REPLACE INTO whale_transfers 
                (timestamp, from_address, to_address, value, gas_used, is_cex)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                transfer['timestamp'],
                transfer['from_address'],
                transfer['to_address'],
                transfer['value'],
                transfer['gas_used'],
                transfer['is_cex']
            ))
        
        conn.commit()
        conn.close()
        print(f"Saved {len(transfers)} transfers to database")
    
    def calculate_netflow_metrics(self, start_time, end_time, window_minutes=10):
        """计算NetFlow指标"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, from_address, to_address, value, is_cex
            FROM whale_transfers
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(query, conn, params=(
            int(start_time.timestamp()), 
            int(end_time.timestamp())
        ))
        conn.close()
        
        if df.empty:
            return pd.DataFrame()
        
        # 计算净流入
        df['netflow'] = df.apply(lambda row: 
            row['value'] if row['is_cex'] == 0 else -row['value'], axis=1)
        
        # 按时间窗口聚合
        df['time_window'] = pd.to_datetime(df['timestamp'], unit='s').dt.floor(f'{window_minutes}min')
        netflow_agg = df.groupby('time_window').agg({
            'netflow': 'sum',
            'value': 'sum'
        }).reset_index()
        
        # 计算Z-score
        netflow_agg['w1_netflow'] = netflow_agg['netflow']
        netflow_agg['w1_zscore'] = (netflow_agg['netflow'] - netflow_agg['netflow'].rolling(100).mean()) / netflow_agg['netflow'].rolling(100).std()
        
        return netflow_agg

def main():
    # 初始化
    api_key = "CM56ZD9J9KTV8K93U8EXP8P4E1CBIEJ1P5"
    fetcher = EtherscanDataFetcher(api_key)
    
    # 获取4月数据
    start_time = datetime(2025, 4, 1)
    end_time = datetime(2025, 4, 30)
    
    print("Starting Etherscan data collection...")
    transfers = fetcher.get_whale_transfers(start_time, end_time, min_value=5000)
    
    # 计算指标
    netflow_df = fetcher.calculate_netflow_metrics(start_time, end_time)
    
    if not netflow_df.empty:
        netflow_df.to_csv('etherscan_netflow_april_2025.csv', index=False)
        print(f"Netflow data saved to etherscan_netflow_april_2025.csv")
        print(f"Total records: {len(netflow_df)}")
        print(f"Average netflow: {netflow_df['w1_netflow'].mean():.2f}")
        print(f"Max netflow: {netflow_df['w1_netflow'].max():.2f}")
        print(f"Min netflow: {netflow_df['w1_netflow'].min():.2f}")

if __name__ == "__main__":
    main() 