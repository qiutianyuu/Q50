import pandas as pd
import numpy as np
import os
import requests
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataPipeline:
    def __init__(self):
        self.base_path = "/Users/qiutianyu"
        self.processed_path = f"{self.base_path}/data/processed"
        self.w1_path = f"{self.base_path}/ETHUSDT-w1"
        
        # API配置
        self.dune_api_key = "sim_xZvnjKWCFpvVMhPAKK4idopF19hShv3f"
        self.etherscan_api_key = "CM56ZD9KTV8K93U8EXP8P4E1CBIEJ1P5"
        
        # 创建必要的目录
        os.makedirs(self.processed_path, exist_ok=True)
        os.makedirs(self.w1_path, exist_ok=True)
        
    def merge_kline_data(self, timeframe):
        """合并指定时间周期的K线数据"""
        print(f"🔄 合并 {timeframe} 数据...")
        import re
        if timeframe == '5m':
            base_path = f"{self.base_path}/ETHUSDT-5m"
        else:
            base_path = f"{self.base_path}/ETHUSDT-{timeframe}"
        
        if not os.path.exists(base_path):
            print(f"❌ 路径不存在: {base_path}")
            return None
            
        all_data = []
        
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and item.startswith(f'ETHUSDT-{timeframe}-'):
                csv_files = [f for f in os.listdir(item_path) if f.endswith('.csv')]
                for csv_file in csv_files:
                    file_path = os.path.join(item_path, csv_file)
                    try:
                        # 所有周期都用正则提取13位毫秒时间戳
                        df = pd.read_csv(file_path, header=None, dtype={0: str}, usecols=range(12))
                        df[0] = df[0].str.extract(r'(\d{13})')[0]
                        df = df.dropna(subset=[0])
                        df[0] = df[0].astype('int64')
                        df = df.iloc[:, :6]
                        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        all_data.append(df)
                        print(f"  ✅ 加载: {csv_file} ({len(df)} 行)")
                    except Exception as e:
                        print(f"  ❌ 错误: {csv_file} - {e}")
        
        if not all_data:
            print(f"❌ 没有找到 {timeframe} 数据")
            return None
            
        merged_df = pd.concat(all_data, ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=['timestamp'])
        merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
        print(f"✅ {timeframe} 数据合并完成: {len(merged_df)} 行")
        return merged_df
    
    def process_kline_data(self, df, timeframe):
        """处理K线数据：UTC对齐、添加时间标签"""
        if df is None or len(df) == 0:
            return None
            
        print(f"🔄 处理 {timeframe} 数据...")
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        print(f"  - 原始时间戳类型: {df['timestamp'].dtype}")
        print(f"  - 时间戳示例: {df['timestamp'].iloc[0]}")
        if np.issubdtype(df['timestamp'].dtype, np.number):
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms', utc=True)
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        # 断言时间戳范围
        min_ts = df['timestamp'].min()
        max_ts = df['timestamp'].max()
        print(f"  - 处理后时间范围: {min_ts} ~ {max_ts}")
        assert pd.Timestamp('2017-01-01', tz='UTC') <= min_ts <= pd.Timestamp('2030-01-01', tz='UTC'), '时间戳异常: min'
        assert pd.Timestamp('2017-01-01', tz='UTC') <= max_ts <= pd.Timestamp('2030-01-01', tz='UTC'), '时间戳异常: max'
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.weekday
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['timeframe'] = timeframe
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_pct'] = df['volume'] / df['volume_ma_20']
        print(f"✅ {timeframe} 数据处理完成")
        return df
    
    def fetch_dune_activity(self, address, limit=100):
        """获取Dune Activity数据"""
        url = f"https://api.sim.dune.com/v1/evm/activity/{address}"
        headers = {"X-Sim-Api-Key": self.dune_api_key}
        
        try:
            response = requests.get(url, headers=headers, params={"limit": limit})
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Dune API错误: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Dune API请求失败: {e}")
            return None
    
    def fetch_dune_transactions(self, address, limit=100):
        """获取Dune Transactions数据"""
        url = f"https://api.sim.dune.com/v1/evm/transactions/{address}"
        headers = {"X-Sim-Api-Key": self.dune_api_key}
        
        try:
            response = requests.get(url, headers=headers, params={"limit": limit})
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Dune Transactions API错误: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Dune Transactions API请求失败: {e}")
            return None
    
    def fetch_etherscan_balance(self, address, chain_id):
        """获取Etherscan余额数据"""
        url = "https://api.etherscan.io/v2/api"
        params = {
            "chainid": chain_id,
            "module": "account",
            "action": "balance",
            "address": address,
            "tag": "latest",
            "apikey": self.etherscan_api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Etherscan API错误: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Etherscan API请求失败: {e}")
            return None
    
    def fetch_dune_data(self):
        """获取Dune大额稳定币转账数据"""
        print("🔄 获取Dune大额稳定币数据...")
        
        # 使用正确的Dune API调用方式
        whale_addresses = [
            "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",  # Binance Hot Wallet
            "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance Cold Wallet
        ]
        
        all_activities = []
        
        for address in whale_addresses:
            print(f"  🔍 获取地址活动: {address[:10]}...")
            
            # 获取Activity数据
            activity_data = self.fetch_dune_activity(address, limit=100)
            if activity_data and 'activity' in activity_data:
                for activity in activity_data['activity']:
                    # 过滤大额转账（>1M USD）
                    if activity.get('value_usd', 0) > 1000000:
                        all_activities.append({
                            'timestamp': activity.get('block_time'),
                            'token': activity.get('token_metadata', {}).get('symbol', 'Unknown'),
                            'amount_usd': activity.get('value_usd', 0),
                            'from_address': activity.get('from'),
                            'to_address': address,
                            'tx_hash': activity.get('tx_hash'),
                            'type': activity.get('type'),
                            'asset_type': activity.get('asset_type'),
                            'chain_id': activity.get('chain_id')
                        })
            
            # API限速
            time.sleep(1)
        
        if all_activities:
            df_dune = pd.DataFrame(all_activities)
            df_dune['timestamp'] = pd.to_datetime(df_dune['timestamp'], utc=True)
            df_dune['date'] = df_dune['timestamp'].dt.date
            df_dune['hour'] = df_dune['timestamp'].dt.hour
            df_dune['weekday'] = df_dune['timestamp'].dt.weekday
            df_dune['month'] = df_dune['timestamp'].dt.month
            df_dune['year'] = df_dune['timestamp'].dt.year
            
            print(f"✅ Dune数据获取完成: {len(df_dune)} 条记录")
        else:
            df_dune = pd.DataFrame()
            print("⚠️  Dune数据为空")
            
        return df_dune
    
    def fetch_etherscan_data(self):
        """获取Etherscan大额ETH转账数据"""
        print("🔄 获取Etherscan大额ETH数据...")
        
        # 定义要监控的链
        chains = [1]  # 只查主网，L2跳过
        
        # 定义要监控的大额地址
        whale_addresses = [
            "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
            "0x28C6c06298d514Db089934071355E5743bf21d60",
            "0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549"
        ]
        
        all_balances = []
        
        for address in whale_addresses:
            print(f"  🔍 获取地址余额: {address[:10]}...")
            
            for chain_id in chains:
                balance_data = self.fetch_etherscan_balance(address, chain_id)
                if balance_data and 'result' in balance_data:
                    try:
                        balance_eth = float(balance_data['result']) / 1e18
                        if balance_eth > 500:  # 过滤大额余额
                            all_balances.append({
                                'timestamp': datetime.now().isoformat(),
                                'amount_eth': balance_eth,
                                'address': address,
                                'chain_id': chain_id,
                                'amount_usd': balance_eth * 2000  # 假设ETH价格
                            })
                    except Exception as e:
                        print(f"    ⚠️  跳过无效余额: {balance_data['result']}")
                        continue
                else:
                    print(f"    ⚠️  API返回异常: {balance_data}")
                # API限速
                time.sleep(0.2)
        
        if all_balances:
            df_etherscan = pd.DataFrame(all_balances)
            df_etherscan['timestamp'] = pd.to_datetime(df_etherscan['timestamp'], utc=True)
            df_etherscan['date'] = df_etherscan['timestamp'].dt.date
            df_etherscan['hour'] = df_etherscan['timestamp'].dt.hour
            df_etherscan['weekday'] = df_etherscan['timestamp'].dt.weekday
            df_etherscan['month'] = df_etherscan['timestamp'].dt.month
            df_etherscan['year'] = df_etherscan['timestamp'].dt.year
            
            print(f"✅ Etherscan数据获取完成: {len(df_etherscan)} 条记录")
        else:
            df_etherscan = pd.DataFrame()
            print("⚠️  Etherscan数据为空")
            
        return df_etherscan
    
    def integrate_whale_signals(self, df_dune, df_etherscan):
        """整合大额转账信号"""
        print("🔄 整合大额转账信号...")
        
        whale_signals = []
        
        # 处理Dune数据
        if len(df_dune) > 0:
            for _, row in df_dune.iterrows():
                whale_signals.append({
                    'timestamp': row['timestamp'],
                    'signal_type': 'stablecoin_whale' if row['asset_type'] == 'erc20' else 'eth_whale',
                    'token': row['token'],
                    'amount': row['amount_usd'],
                    'amount_usd': row['amount_usd'],
                    'source': 'dune',
                    'chain_id': row.get('chain_id', 1),
                    'tx_hash': row.get('tx_hash'),
                    'date': row['date'],
                    'hour': row['hour'],
                    'weekday': row['weekday'],
                    'month': row['month'],
                    'year': row['year']
                })
        
        # 处理Etherscan数据
        if len(df_etherscan) > 0:
            for _, row in df_etherscan.iterrows():
                whale_signals.append({
                    'timestamp': row['timestamp'],
                    'signal_type': 'eth_whale',
                    'token': 'ETH',
                    'amount': row['amount_eth'],
                    'amount_usd': row['amount_usd'],
                    'source': 'etherscan',
                    'chain_id': row['chain_id'],
                    'tx_hash': None,
                    'date': row['date'],
                    'hour': row['hour'],
                    'weekday': row['weekday'],
                    'month': row['month'],
                    'year': row['year']
                })
        
        if whale_signals:
            df_whale = pd.DataFrame(whale_signals)
            df_whale = df_whale.sort_values('timestamp').reset_index(drop=True)
            print(f"✅ 大额信号整合完成: {len(df_whale)} 条记录")
        else:
            df_whale = pd.DataFrame()
            print("⚠️  没有大额信号数据")
            
        return df_whale
    
    def save_processed_data(self, df_5m, df_15m, df_1h, df_4h, df_whale):
        """保存处理后的数据"""
        print("🔄 保存处理后的数据...")
        
        # 保存K线数据
        if df_5m is not None:
            df_5m.to_parquet(f"{self.processed_path}/merged_5m_2023_2025.parquet", index=False)
            print(f"✅ 5m数据已保存: {len(df_5m)} 行")
            
        if df_15m is not None:
            df_15m.to_parquet(f"{self.processed_path}/merged_15m_2023_2025.parquet", index=False)
            print(f"✅ 15m数据已保存: {len(df_15m)} 行")
            
        if df_1h is not None:
            df_1h.to_parquet(f"{self.processed_path}/merged_1h_2023_2025.parquet", index=False)
            print(f"✅ 1h数据已保存: {len(df_1h)} 行")
            
        if df_4h is not None:
            df_4h.to_parquet(f"{self.processed_path}/merged_4h_2023_2025.parquet", index=False)
            print(f"✅ 4h数据已保存: {len(df_4h)} 行")
        
        # 保存大额信号数据
        if len(df_whale) > 0:
            df_whale.to_parquet(f"{self.w1_path}/w1_2023_2025.parquet", index=False)
            print(f"✅ 大额信号数据已保存: {len(df_whale)} 行")
        else:
            print("⚠️  没有大额信号数据可保存")
    
    def run_pipeline(self):
        """运行完整的数据管道"""
        print("🚀 开始数据管道处理...")
        print("=" * 60)
        
        # 1. 合并K线数据
        df_5m = self.merge_kline_data("5m")
        df_15m = self.merge_kline_data("15m")
        df_1h = self.merge_kline_data("1h")
        df_4h = self.merge_kline_data("4h")
        
        # 2. 处理K线数据
        df_5m = self.process_kline_data(df_5m, "5m")
        df_15m = self.process_kline_data(df_15m, "15m")
        df_1h = self.process_kline_data(df_1h, "1h")
        df_4h = self.process_kline_data(df_4h, "4h")
        
        # 3. 获取大额转账数据
        df_dune = self.fetch_dune_data()
        df_etherscan = self.fetch_etherscan_data()
        
        # 4. 整合大额信号
        df_whale = self.integrate_whale_signals(df_dune, df_etherscan)
        
        # 5. 保存处理后的数据
        self.save_processed_data(df_5m, df_15m, df_1h, df_4h, df_whale)
        
        print("=" * 60)
        print("✅ 数据管道处理完成！")
        
        # 返回数据统计
        stats = {
            '5m_rows': len(df_5m) if df_5m is not None else 0,
            '15m_rows': len(df_15m) if df_15m is not None else 0,
            '1h_rows': len(df_1h) if df_1h is not None else 0,
            '4h_rows': len(df_4h) if df_4h is not None else 0,
            'whale_signals': len(df_whale)
        }
        
        print("\n📊 数据统计:")
        for key, value in stats.items():
            print(f"  {key}: {value:,}")
            
        return stats

def main():
    """主函数"""
    pipeline = DataPipeline()
    stats = pipeline.run_pipeline()
    
    print(f"\n💾 数据已保存到:")
    print(f"  K线数据: {pipeline.processed_path}")
    print(f"  大额信号: {pipeline.w1_path}")

if __name__ == "__main__":
    main() 