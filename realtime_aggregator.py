#!/usr/bin/env python3
"""
实时聚合脚本
监控WebSocket数据并实时生成1分钟特征
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging
import glob
import os
import time
from typing import Dict, List, Optional
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('realtime_aggregator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealtimeAggregator:
    """实时聚合器"""
    
    def __init__(self, symbol: str = "ETH-USDT", output_dir: str = "data/realtime_features"):
        self.symbol = symbol
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据缓冲区
        self.orderbook_buffer = []
        self.current_minute = None
        self.last_aggregation_time = None
        
        # 统计信息
        self.stats = {
            "files_processed": 0,
            "rows_processed": 0,
            "features_generated": 0,
            "start_time": None
        }
    
    def load_latest_orderbook_files(self) -> List[str]:
        """加载最新的OrderBook文件"""
        # 查找今天的OrderBook文件
        today = datetime.utcnow().strftime("%Y%m%d")
        pattern = f"data/websocket/{today}/orderbook_{self.symbol}_*.parquet"
        files = glob.glob(pattern)
        
        if not files:
            # 如果没有按日期分区的文件，查找根目录
            pattern = f"data/websocket/orderbook_{self.symbol}_*.parquet"
            files = glob.glob(pattern)
        
        # 按修改时间排序，只处理新文件
        files.sort(key=os.path.getmtime)
        
        if self.last_aggregation_time:
            # 只处理上次聚合后的新文件
            new_files = []
            for file in files:
                if os.path.getmtime(file) > self.last_aggregation_time:
                    new_files.append(file)
            files = new_files
        
        return files
    
    def process_orderbook_file(self, filepath: str) -> pd.DataFrame:
        """处理单个OrderBook文件"""
        try:
            df = pd.read_parquet(filepath)
            logger.info(f"📖 处理文件: {filepath} ({len(df)} 行)")
            return df
        except Exception as e:
            logger.error(f"❌ 处理文件失败 {filepath}: {e}")
            return pd.DataFrame()
    
    def calculate_micro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算微观特征"""
        if df.empty:
            return df
        
        features = []
        
        for _, row in df.iterrows():
            # 提取价格和数量
            bid_prices = []
            bid_sizes = []
            ask_prices = []
            ask_sizes = []
            
            for i in range(1, 6):
                if f'bid{i}_price' in row and f'bid{i}_size' in row:
                    bid_prices.append(row[f'bid{i}_price'])
                    bid_sizes.append(row[f'bid{i}_size'])
                if f'ask{i}_price' in row and f'ask{i}_size' in row:
                    ask_prices.append(row[f'ask{i}_price'])
                    ask_sizes.append(row[f'ask{i}_size'])
            
            if not bid_prices or not ask_prices:
                continue
            
            # 计算基础特征
            best_bid = bid_prices[0]
            best_ask = ask_prices[0]
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            spread_bps = spread / best_bid * 10000
            
            # 计算VWAP
            total_bid_volume = sum(bid_sizes)
            total_ask_volume = sum(ask_sizes)
            
            if total_bid_volume > 0 and total_ask_volume > 0:
                bid_vwap = sum(p * s for p, s in zip(bid_prices, bid_sizes)) / total_bid_volume
                ask_vwap = sum(p * s for p, s in zip(ask_prices, ask_sizes)) / total_ask_volume
                vwap = (bid_vwap + ask_vwap) / 2
            else:
                vwap = mid_price
            
            # 计算订单流特征
            volume_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            price_pressure = (ask_vwap - bid_vwap) / mid_price
            
            # 计算流动性特征
            liquidity_score = (total_bid_volume + total_ask_volume) / spread_bps if spread_bps > 0 else 0
            
            feature_row = {
                'timestamp': row['timestamp'],
                'mid_price': mid_price,
                'bid_price': best_bid,
                'ask_price': best_ask,
                'spread': spread,
                'spread_bps': spread_bps,
                'vwap': vwap,
                'bid_vwap': bid_vwap,
                'ask_vwap': ask_vwap,
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume,
                'volume_imbalance': volume_imbalance,
                'price_pressure': price_pressure,
                'liquidity_score': liquidity_score,
                'price_mean': np.mean(bid_prices + ask_prices),
                'price_std': np.std(bid_prices + ask_prices),
                'price_range': max(ask_prices) - min(bid_prices)
            }
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def aggregate_to_minute(self, df: pd.DataFrame) -> pd.DataFrame:
        """聚合到1分钟"""
        if df.empty:
            return df
        
        # 确保时间戳格式
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 设置时间索引
        df = df.set_index('timestamp').sort_index()
        
        # 1分钟聚合规则
        agg_rules = {
            'mid_price': 'last',
            'bid_price': 'last',
            'ask_price': 'last',
            'spread': 'mean',
            'spread_bps': 'mean',
            'vwap': 'mean',
            'bid_vwap': 'mean',
            'ask_vwap': 'mean',
            'total_bid_volume': 'sum',
            'total_ask_volume': 'sum',
            'volume_imbalance': 'mean',
            'price_pressure': 'mean',
            'liquidity_score': 'mean',
            'price_mean': 'mean',
            'price_std': 'mean',
            'price_range': 'max'
        }
        
        # 执行聚合
        resampled = df.resample('1T').agg(agg_rules)
        
        # 计算额外特征
        resampled = self.calculate_minute_features(resampled)
        
        return resampled.reset_index()
    
    def calculate_minute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算1分钟额外特征"""
        # 价格动量
        df['price_momentum_1m'] = df['mid_price'].pct_change(1)
        df['price_momentum_3m'] = df['mid_price'].pct_change(3)
        df['price_momentum_5m'] = df['mid_price'].pct_change(5)
        
        # 成交量特征
        df['total_volume'] = df['total_bid_volume'] + df['total_ask_volume']
        df['volume_ma_3m'] = df['total_volume'].rolling(3).mean()
        df['volume_ratio'] = df['total_volume'] / df['volume_ma_3m']
        
        # 价差特征
        df['spread_ma'] = df['spread_bps'].rolling(5).mean()
        df['spread_ratio'] = df['spread_bps'] / df['spread_ma']
        
        # 订单流特征
        df['imbalance_ma'] = df['volume_imbalance'].rolling(5).mean()
        df['imbalance_trend'] = df['imbalance_ma'].pct_change()
        
        # 流动性特征
        df['liquidity_ma'] = df['liquidity_score'].rolling(5).mean()
        df['liquidity_trend'] = df['liquidity_ma'].pct_change()
        
        # 波动率特征
        df['price_volatility'] = df['price_momentum_1m'].rolling(5).std()
        df['volatility_ma'] = df['price_volatility'].rolling(5).mean()
        df['volatility_ratio'] = df['price_volatility'] / df['volatility_ma']
        
        # 异常检测
        df['price_jump'] = (abs(df['price_momentum_1m']) > df['price_momentum_1m'].rolling(20).quantile(0.95)).astype(int)
        df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)
        df['spread_widening'] = (df['spread_ratio'] > 1.5).astype(int)
        
        return df
    
    def save_features(self, df: pd.DataFrame):
        """保存特征数据"""
        if df.empty:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"realtime_features_{self.symbol}_{timestamp}.parquet"
            filepath = self.output_dir / filename
            
            # 使用zstd压缩
            df.to_parquet(filepath, index=False, compression='zstd')
            
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            self.stats['features_generated'] += 1
            
            logger.info(f"💾 保存特征: {len(df)} 行 -> {filepath} ({file_size_mb:.2f}MB)")
            
        except Exception as e:
            logger.error(f"❌ 保存特征失败: {e}")
    
    async def run_aggregation(self):
        """运行聚合任务"""
        logger.info("🚀 启动实时聚合器...")
        self.stats['start_time'] = datetime.now()
        
        while True:
            try:
                # 加载新文件
                files = self.load_latest_orderbook_files()
                
                if files:
                    logger.info(f"📁 发现 {len(files)} 个新文件")
                    
                    all_data = []
                    for file in files:
                        df = self.process_orderbook_file(file)
                        if not df.empty:
                            # 计算微观特征
                            micro_features = self.calculate_micro_features(df)
                            all_data.append(micro_features)
                            self.stats['rows_processed'] += len(df)
                    
                    if all_data:
                        # 合并所有数据
                        combined_df = pd.concat(all_data, ignore_index=True)
                        
                        # 聚合到1分钟
                        minute_features = self.aggregate_to_minute(combined_df)
                        
                        if not minute_features.empty:
                            # 保存特征
                            self.save_features(minute_features)
                            self.stats['files_processed'] += len(files)
                
                # 更新最后处理时间
                self.last_aggregation_time = time.time()
                
                # 打印统计
                elapsed = datetime.now() - self.stats['start_time']
                logger.info(f"📊 统计: 处理{self.stats['files_processed']}文件, "
                          f"{self.stats['rows_processed']}行, "
                          f"生成{self.stats['features_generated']}特征文件 "
                          f"({elapsed.total_seconds():.0f}s)")
                
                # 等待30秒后再次检查
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"❌ 聚合任务异常: {e}")
                await asyncio.sleep(30)

async def main():
    """主函数"""
    logger.info("🚀 启动实时聚合器")
    
    aggregator = RealtimeAggregator(symbol="ETH-USDT")
    await aggregator.run_aggregation()

if __name__ == "__main__":
    asyncio.run(main()) 