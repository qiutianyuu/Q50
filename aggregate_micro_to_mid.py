#!/usr/bin/env python3
"""
微观数据聚合脚本
将秒级orderbook/trades数据聚合成1m/5m中频特征
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
import os
from typing import Dict, List, Tuple

def load_micro_features():
    """加载微观特征数据"""
    files = glob.glob("data/micro_features_*.parquet")
    if not files:
        raise FileNotFoundError("No micro features files found")
    
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading: {latest_file}")
    df = pd.read_parquet(latest_file)
    print(f"Loaded {len(df)} rows")
    return df

def calculate_order_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算订单流特征"""
    # 基础价格特征
    df['price_change'] = df['mid_price'].pct_change()
    df['price_volatility'] = df['price_change'].rolling(20).std()
    
    # 买卖盘失衡
    df['bid_ask_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
    df['volume_imbalance_ma'] = df['volume_imbalance'].rolling(20).mean()
    
    # 价差特征
    df['spread_ma'] = df['spread_bps'].rolling(20).mean()
    df['spread_std'] = df['spread_bps'].rolling(20).std()
    
    # 流动性特征
    df['liquidity_score'] = (df['bid_volume'] + df['ask_volume']) / df['spread_bps']
    df['liquidity_ma'] = df['liquidity_score'].rolling(20).mean()
    
    # 价格压力
    df['price_pressure_ma'] = df['price_pressure'].rolling(20).mean()
    df['price_pressure_std'] = df['price_pressure'].rolling(20).std()
    
    # VWAP偏离
    df['vwap_deviation'] = (df['mid_price'] - df['vwap']) / df['vwap']
    df['vwap_deviation_ma'] = df['vwap_deviation'].rolling(20).mean()
    
    return df

def aggregate_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """聚合到指定时间框架"""
    # 确保timestamp是datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 设置时间索引
    df = df.set_index('timestamp').sort_index()
    
    # 定义聚合规则
    agg_rules = {
        # 价格特征
        'mid_price': 'last',
        'bid_price': 'last', 
        'ask_price': 'last',
        'vwap': 'mean',
        'price_mean': 'mean',
        'price_std': 'mean',
        'price_range': 'max',
        
        # 价差特征
        'spread': 'mean',
        'spread_bps': 'mean',
        'rel_spread': 'mean',
        'spread_ma': 'mean',
        'spread_std': 'mean',
        
        # 成交量特征
        'total_volume_x': 'sum',
        'total_volume_y': 'sum',
        'bid_volume': 'sum',
        'ask_volume': 'sum',
        'avg_trade_size': 'mean',
        'large_trades': 'sum',
        'trade_frequency': 'mean',
        'trade_count': 'sum',
        
        # 订单流特征
        'volume_imbalance': 'mean',
        'volume_imbalance_ma': 'mean',
        'bid_ask_imbalance': 'mean',
        'price_pressure': 'mean',
        'price_pressure_ma': 'mean',
        'price_pressure_std': 'mean',
        'buy_ratio': 'mean',
        'price_momentum': 'mean',
        
        # 流动性特征
        'liquidity_score': 'mean',
        'liquidity_ma': 'mean',
        'bid_fill_prob': 'mean',
        'ask_fill_prob': 'mean',
        'bid_price_impact': 'mean',
        'ask_price_impact': 'mean',
        'optimal_bid_size': 'mean',
        'optimal_ask_size': 'mean',
        'bid_queue_depth': 'mean',
        'ask_queue_depth': 'mean',
        'bid_levels_available': 'mean',
        'ask_levels_available': 'mean',
        
        # 价格变动特征
        'price_change': 'sum',  # 累计价格变动
        'price_change_abs': 'sum',
        'price_volatility': 'mean',
        'vwap_deviation': 'mean',
        'vwap_deviation_ma': 'mean',
        
        # 趋势特征
        'price_trend': 'mean',
        'trend_deviation': 'mean',
        'price_impact_imbalance': 'mean',
        'fill_prob_imbalance': 'mean',
        
        # 变化率特征
        'buy_pressure_change': 'sum',
        'spread_change': 'sum',
        'price_volatility': 'mean',
        'bid_fill_prob_change': 'sum',
        'ask_fill_prob_change': 'sum'
    }
    
    # 执行聚合
    if timeframe == '1m':
        resampled = df.resample('1T').agg(agg_rules)
    elif timeframe == '5m':
        resampled = df.resample('5T').agg(agg_rules)
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    # 计算额外特征
    resampled = calculate_aggregated_features(resampled)
    
    return resampled.reset_index()

def calculate_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算聚合后的额外特征"""
    # 价格动量
    df['price_momentum_1m'] = df['mid_price'].pct_change(1)
    df['price_momentum_3m'] = df['mid_price'].pct_change(3)
    df['price_momentum_5m'] = df['mid_price'].pct_change(5)
    
    # 成交量特征
    df['volume_ma_3m'] = df['total_volume_x'].rolling(3).mean()
    df['volume_ma_5m'] = df['total_volume_x'].rolling(5).mean()
    df['volume_ratio'] = df['total_volume_x'] / df['volume_ma_3m']
    
    # 价差特征
    df['spread_ratio'] = df['spread_bps'] / df['spread_ma']
    df['spread_trend'] = df['spread_bps'].rolling(3).mean().pct_change()
    
    # 订单流特征
    df['imbalance_trend'] = df['volume_imbalance'].rolling(3).mean().pct_change()
    df['pressure_trend'] = df['price_pressure'].rolling(3).mean().pct_change()
    
    # 流动性特征
    df['liquidity_trend'] = df['liquidity_score'].rolling(3).mean().pct_change()
    df['fill_prob_trend'] = (df['bid_fill_prob'] - df['ask_fill_prob']).rolling(3).mean()
    
    # 波动率特征
    df['volatility_ma'] = df['price_volatility'].rolling(5).mean()
    df['volatility_ratio'] = df['price_volatility'] / df['volatility_ma']
    
    # 异常检测
    df['price_jump'] = (df['price_change_abs'] > df['price_change_abs'].rolling(20).quantile(0.95)).astype(int)
    df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)
    df['spread_widening'] = (df['spread_ratio'] > 1.5).astype(int)
    
    return df

def main():
    print("=== 微观数据聚合脚本 ===")
    
    # 加载微观特征数据
    df = load_micro_features()
    
    # 计算订单流特征
    print("计算订单流特征...")
    df = calculate_order_flow_features(df)
    
    # 聚合到1分钟
    print("聚合到1分钟...")
    df_1m = aggregate_to_timeframe(df, '1m')
    print(f"1分钟数据: {len(df_1m)} 行")
    
    # 聚合到5分钟
    print("聚合到5分钟...")
    df_5m = aggregate_to_timeframe(df, '5m')
    print(f"5分钟数据: {len(df_5m)} 行")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存1分钟数据
    output_1m = f"data/mid_features_1m_{timestamp}.parquet"
    df_1m.to_parquet(output_1m, index=False)
    print(f"1分钟特征已保存: {output_1m}")
    
    # 保存5分钟数据
    output_5m = f"data/mid_features_5m_{timestamp}.parquet"
    df_5m.to_parquet(output_5m, index=False)
    print(f"5分钟特征已保存: {output_5m}")
    
    # 显示特征统计
    print(f"\n=== 特征统计 ===")
    print(f"1分钟特征数量: {len(df_1m.columns)}")
    print(f"5分钟特征数量: {len(df_5m.columns)}")
    
    # 显示样本数据
    print(f"\n=== 1分钟样本数据 ===")
    sample_cols = ['timestamp', 'mid_price', 'price_momentum_1m', 'volume_ratio', 
                  'spread_ratio', 'imbalance_trend', 'price_jump', 'volume_spike']
    print(df_1m[sample_cols].head(10))
    
    print(f"\n=== 5分钟样本数据 ===")
    print(df_5m[sample_cols].head(10))
    
    # 显示数据时间范围
    print(f"\n=== 数据时间范围 ===")
    print(f"1分钟: {df_1m['timestamp'].min()} 到 {df_1m['timestamp'].max()}")
    print(f"5分钟: {df_5m['timestamp'].min()} 到 {df_5m['timestamp'].max()}")

if __name__ == "__main__":
    main() 