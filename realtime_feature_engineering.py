#!/usr/bin/env python3
"""
实时特征工程脚本
处理WebSocket收集的OrderBook和Trades数据，计算微价格特征和订单流特征
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealtimeFeatureEngineering:
    def __init__(self, symbol: str = "ETH-USDT", window_size: int = 100):
        self.symbol = symbol
        self.window_size = window_size
        self.websocket_dir = "data/websocket"
        
    def load_latest_data(self, data_type: str = "trades") -> pd.DataFrame:
        """加载最新的数据文件"""
        pattern = f"{self.websocket_dir}/{data_type}_{self.symbol}_*.parquet"
        files = glob.glob(pattern)
        
        if not files:
            logger.warning(f"没有找到{data_type}数据文件")
            return pd.DataFrame()
        
        # 按时间排序，取最新的文件
        files.sort()
        latest_file = files[-1]
        
        logger.info(f"加载最新{data_type}数据: {latest_file}")
        df = pd.read_parquet(latest_file)
        
        # 确保时间戳列存在
        if 'ts' in df.columns:
            df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
        elif 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime('now')
            
        return df
    
    def calculate_micro_price_features(self, orderbook_df: pd.DataFrame) -> pd.DataFrame:
        """计算微价格特征"""
        if orderbook_df.empty:
            return pd.DataFrame()
        
        features = []
        
        for _, row in orderbook_df.iterrows():
            # 直接使用解析好的OrderBook数据
            bid_prices = []
            bid_sizes = []
            ask_prices = []
            ask_sizes = []
            
            # 提取5档买卖盘数据
            for i in range(1, 6):
                bid_price_col = f'bid{i}_price'
                bid_size_col = f'bid{i}_size'
                ask_price_col = f'ask{i}_price'
                ask_size_col = f'ask{i}_size'
                
                if bid_price_col in row and bid_size_col in row:
                    bid_prices.append(row[bid_price_col])
                    bid_sizes.append(row[bid_size_col])
                
                if ask_price_col in row and ask_size_col in row:
                    ask_prices.append(row[ask_price_col])
                    ask_sizes.append(row[ask_size_col])
            
            if not bid_prices or not ask_prices:
                continue
            
            # 加权平均价格 (VWAP)
            bid_vwap = np.average(bid_prices, weights=bid_sizes) if bid_sizes else 0
            ask_vwap = np.average(ask_prices, weights=ask_sizes) if ask_sizes else 0
            
            # 中间价
            mid_price = (bid_prices[0] + ask_prices[0]) / 2 if bid_prices and ask_prices else 0
            
            # 买卖价差
            spread = ask_prices[0] - bid_prices[0] if bid_prices and ask_prices else 0
            spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0
            rel_spread = (spread / mid_price) if mid_price > 0 else 0
            
            # 订单簿不平衡
            bid_volume = sum(bid_sizes)
            ask_volume = sum(ask_sizes)
            volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
            
            # 价格压力
            price_pressure = (ask_vwap - mid_price) / mid_price if mid_price > 0 else 0
            
            features.append({
                'timestamp': row.get('timestamp', pd.Timestamp.now()),
                'mid_price': mid_price,
                'bid_vwap': bid_vwap,
                'ask_vwap': ask_vwap,
                'spread': spread,
                'spread_bps': spread_bps,
                'rel_spread': rel_spread,
                'volume_imbalance': volume_imbalance,
                'price_pressure': price_pressure,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'total_volume': bid_volume + ask_volume
            })
        
        return pd.DataFrame(features)
    
    def calculate_order_flow_features(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """计算订单流特征"""
        if trades_df.empty:
            return pd.DataFrame()
        
        features = []
        
        # 按时间窗口分组计算特征
        trades_df = trades_df.sort_values('timestamp')
        
        for i in range(0, len(trades_df), self.window_size):
            window_trades = trades_df.iloc[i:i+self.window_size]
            
            if len(window_trades) == 0:
                continue
            
            # 计算基础统计
            prices = window_trades['price'].astype(float)
            sizes = window_trades['size'].astype(float)
            sides = window_trades['side'].astype(str)
            
            # 价格特征
            price_mean = prices.mean()
            price_std = prices.std()
            price_range = prices.max() - prices.min()
            
            # 成交量特征
            total_volume = sizes.sum()
            avg_trade_size = sizes.mean()
            large_trades = (sizes > sizes.quantile(0.9)).sum()
            
            # 买卖压力
            buy_volume = sizes[sides == 'buy'].sum()
            sell_volume = sizes[sides == 'sell'].sum()
            buy_ratio = buy_volume / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0.5
            
            # 价格动量
            if len(prices) > 1:
                price_momentum = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
            else:
                price_momentum = 0
            
            # 成交量加权平均价格
            vwap = np.average(prices, weights=sizes)
            
            # 时间特征
            time_span = (window_trades['timestamp'].max() - window_trades['timestamp'].min()).total_seconds()
            trade_frequency = len(window_trades) / time_span if time_span > 0 else 0
            
            features.append({
                'timestamp': window_trades['timestamp'].iloc[-1],
                'price_mean': price_mean,
                'price_std': price_std,
                'price_range': price_range,
                'total_volume': total_volume,
                'avg_trade_size': avg_trade_size,
                'large_trades': large_trades,
                'buy_ratio': buy_ratio,
                'price_momentum': price_momentum,
                'vwap': vwap,
                'trade_frequency': trade_frequency,
                'trade_count': len(window_trades)
            })
        
        return pd.DataFrame(features)
    
    def calculate_combined_features(self, micro_price_df: pd.DataFrame, order_flow_df: pd.DataFrame) -> pd.DataFrame:
        """合并微价格和订单流特征"""
        if micro_price_df.empty and order_flow_df.empty:
            return pd.DataFrame()
        
        # 合并数据框
        combined_df = pd.DataFrame()
        
        if not micro_price_df.empty:
            combined_df = micro_price_df.copy()
        
        if not order_flow_df.empty:
            if combined_df.empty:
                combined_df = order_flow_df.copy()
            else:
                # 基于时间戳合并
                combined_df = pd.merge_asof(
                    combined_df.sort_values('timestamp'),
                    order_flow_df.sort_values('timestamp'),
                    on='timestamp',
                    direction='nearest',
                    suffixes=('', '_flow')
                )
        
        # 计算衍生特征
        if not combined_df.empty:
            # 价格变化率
            combined_df['price_change'] = combined_df['mid_price'].pct_change()
            combined_df['price_change_abs'] = combined_df['price_change'].abs()
            
            # 成交量变化率
            if 'total_volume' in combined_df.columns:
                combined_df['volume_change'] = combined_df['total_volume'].pct_change()
            
            # 买卖压力变化
            if 'buy_ratio' in combined_df.columns:
                combined_df['buy_pressure_change'] = combined_df['buy_ratio'].diff()
            
            # 价差变化
            combined_df['spread_change'] = combined_df['spread'].pct_change()
            
            # 波动率特征
            combined_df['price_volatility'] = combined_df['price_change'].rolling(5).std()
            
            # 趋势特征
            combined_df['price_trend'] = combined_df['mid_price'].rolling(10).mean()
            combined_df['trend_deviation'] = (combined_df['mid_price'] - combined_df['price_trend']) / combined_df['price_trend']
        
        return combined_df
    
    def save_features(self, features_df: pd.DataFrame, output_file: str = None):
        """保存特征数据"""
        if features_df.empty:
            logger.warning("没有特征数据可保存")
            return
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"data/realtime_features_{self.symbol}_{timestamp}.parquet"
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        features_df.to_parquet(output_file, index=False)
        logger.info(f"💾 保存实时特征数据: {len(features_df)} 条 -> {output_file}")
        
        return output_file
    
    def run_feature_engineering(self, save_output: bool = True) -> pd.DataFrame:
        """运行完整的特征工程流程"""
        logger.info("🚀 开始实时特征工程...")
        
        # 1. 加载最新数据
        orderbook_df = self.load_latest_data("orderbook")
        trades_df = self.load_latest_data("trades")
        
        logger.info(f"📊 加载数据: OrderBook {len(orderbook_df)} 条, Trades {len(trades_df)} 条")
        
        # 2. 计算微价格特征
        micro_price_features = self.calculate_micro_price_features(orderbook_df)
        logger.info(f"📈 计算微价格特征: {len(micro_price_features)} 条")
        
        # 3. 计算订单流特征
        order_flow_features = self.calculate_order_flow_features(trades_df)
        logger.info(f"📊 计算订单流特征: {len(order_flow_features)} 条")
        
        # 4. 合并特征
        combined_features = self.calculate_combined_features(micro_price_features, order_flow_features)
        logger.info(f"🔗 合并特征: {len(combined_features)} 条")
        
        # 5. 保存结果
        if save_output and not combined_features.empty:
            output_file = self.save_features(combined_features)
            logger.info(f"✅ 特征工程完成，保存到: {output_file}")
        
        return combined_features

def main():
    parser = argparse.ArgumentParser(description="实时特征工程")
    parser.add_argument("--symbol", default="ETH-USDT", help="交易对")
    parser.add_argument("--window-size", type=int, default=100, help="滑动窗口大小")
    parser.add_argument("--no-save", action="store_true", help="不保存输出文件")
    
    args = parser.parse_args()
    
    # 创建特征工程实例
    fe = RealtimeFeatureEngineering(
        symbol=args.symbol,
        window_size=args.window_size
    )
    
    # 运行特征工程
    features = fe.run_feature_engineering(save_output=not args.no_save)
    
    if not features.empty:
        print("\n📊 特征统计:")
        print(features.describe())
        
        print("\n🔍 特征列:")
        print(features.columns.tolist())

if __name__ == "__main__":
    main() 