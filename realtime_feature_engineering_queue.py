#!/usr/bin/env python3
"""
集成队列模拟器的实时特征工程脚本
处理WebSocket收集的OrderBook和Trades数据，计算微价格特征、订单流特征和队列特征
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import argparse
from utils.queue_simulator import calculate_queue_features

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealtimeFeatureEngineeringQueue:
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
    
    def calculate_queue_features(self, orderbook_df: pd.DataFrame) -> pd.DataFrame:
        """计算队列相关特征"""
        if orderbook_df.empty:
            return pd.DataFrame()
        
        logger.info("计算队列特征...")
        queue_features = calculate_queue_features(orderbook_df)
        logger.info(f"队列特征计算完成: {len(queue_features)} 行")
        
        return queue_features
    
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
    
    def calculate_combined_features(self, micro_price_df: pd.DataFrame, order_flow_df: pd.DataFrame, 
                                  queue_df: pd.DataFrame) -> pd.DataFrame:
        """合并微价格、订单流和队列特征"""
        if micro_price_df.empty and order_flow_df.empty and queue_df.empty:
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
                    tolerance=pd.Timedelta(seconds=30)
                )
        
        if not queue_df.empty:
            if combined_df.empty:
                combined_df = queue_df.copy()
            else:
                # 基于时间戳合并队列特征
                combined_df = pd.merge_asof(
                    combined_df.sort_values('timestamp'),
                    queue_df.sort_values('timestamp'),
                    on='timestamp',
                    direction='nearest',
                    tolerance=pd.Timedelta(seconds=30)
                )
        
        # 计算衍生特征
        if not combined_df.empty:
            combined_df = self.calculate_derived_features(combined_df)
        
        return combined_df
    
    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算衍生特征"""
        # 价格变化
        df['price_change'] = df['mid_price'].diff()
        df['price_change_abs'] = df['price_change'].abs()
        
        # 成交量变化
        if 'total_volume' in df.columns:
            df['volume_change'] = df['total_volume'].diff()
        
        # 买卖压力变化
        if 'buy_ratio' in df.columns:
            df['buy_pressure_change'] = df['buy_ratio'].diff()
        
        # 价差变化
        df['spread_change'] = df['spread'].diff()
        
        # 价格波动率
        df['price_volatility'] = df['price_change'].rolling(window=20).std()
        
        # 价格趋势
        df['price_trend'] = df['mid_price'].rolling(window=20).mean()
        df['trend_deviation'] = (df['mid_price'] - df['price_trend']) / df['price_trend']
        
        # 队列特征变化
        if 'bid_fill_prob' in df.columns:
            df['bid_fill_prob_change'] = df['bid_fill_prob'].diff()
            df['ask_fill_prob_change'] = df['ask_fill_prob'].diff()
            df['fill_prob_imbalance'] = df['bid_fill_prob'] - df['ask_fill_prob']
        
        if 'bid_price_impact' in df.columns:
            df['price_impact_imbalance'] = df['bid_price_impact'] - df['ask_price_impact']
        
        return df
    
    def save_features(self, features_df: pd.DataFrame, output_file: str = None):
        """保存特征数据"""
        if features_df.empty:
            logger.warning("没有特征数据可保存")
            return
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"data/realtime_features_queue_{self.symbol}_{timestamp}.parquet"
        
        features_df.to_parquet(output_file, index=False)
        logger.info(f"💾 保存实时特征数据: {len(features_df)} 条 -> {output_file}")
        
        # 显示特征统计
        print(f"\n📊 特征统计:")
        print(features_df.describe())
        
        print(f"\n🔍 特征列:")
        print(list(features_df.columns))
        
        return output_file
    
    def run_feature_engineering(self, save_output: bool = True) -> pd.DataFrame:
        """运行完整的特征工程流程"""
        logger.info("🚀 开始实时特征工程（集成队列模拟器）...")
        
        # 加载数据
        orderbook_df = self.load_latest_data("orderbook")
        trades_df = self.load_latest_data("trades")
        
        logger.info(f"📊 加载数据: OrderBook {len(orderbook_df)} 条, Trades {len(trades_df)} 条")
        
        # 计算各类特征
        micro_price_df = self.calculate_micro_price_features(orderbook_df)
        logger.info(f"📈 计算微价格特征: {len(micro_price_df)} 条")
        
        order_flow_df = self.calculate_order_flow_features(trades_df)
        logger.info(f"📊 计算订单流特征: {len(order_flow_df)} 条")
        
        queue_df = self.calculate_queue_features(orderbook_df)
        logger.info(f"🎯 计算队列特征: {len(queue_df)} 条")
        
        # 合并特征
        combined_df = self.calculate_combined_features(micro_price_df, order_flow_df, queue_df)
        logger.info(f"🔗 合并特征: {len(combined_df)} 条")
        
        # 保存结果
        output_file = None
        if save_output:
            output_file = self.save_features(combined_df)
            logger.info(f"✅ 特征工程完成，保存到: {output_file}")
        
        return combined_df

def main():
    parser = argparse.ArgumentParser(description='实时特征工程（集成队列模拟器）')
    parser.add_argument('--symbol', default='ETH-USDT', help='交易对')
    parser.add_argument('--window-size', type=int, default=100, help='窗口大小')
    parser.add_argument('--no-save', action='store_true', help='不保存结果')
    
    args = parser.parse_args()
    
    # 创建特征工程实例
    fe = RealtimeFeatureEngineeringQueue(
        symbol=args.symbol,
        window_size=args.window_size
    )
    
    # 运行特征工程
    features_df = fe.run_feature_engineering(save_output=not args.no_save)
    
    return features_df

if __name__ == "__main__":
    main() 