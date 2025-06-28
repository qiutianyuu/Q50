#!/usr/bin/env python3
"""
微观数据聚合到15分钟脚本
将秒级orderbook/trades数据聚合成15m高频特征，增强订单流指标
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

def calculate_enhanced_order_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算增强的订单流特征"""
    # 基础价格特征
    df['price_change'] = df['mid_price'].pct_change()
    df['price_volatility'] = df['price_change'].rolling(20).std()
    
    # 买卖盘失衡（核心特征）
    df['bid_ask_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
    df['volume_imbalance_ma'] = df['volume_imbalance'].rolling(20).mean()
    
    # 流动性压强指标
    df['liquidity_pressure'] = df['bid_ask_imbalance'] * df['total_volume_x']
    df['liquidity_pressure_ma'] = df['liquidity_pressure'].rolling(20).mean()
    
    # 成交量不均衡（Taker方向）
    df['taker_imbalance'] = (df['buy_ratio'] - 0.5) * 2  # 归一化到[-1, 1]
    df['taker_imbalance_ma'] = df['taker_imbalance'].rolling(20).mean()
    
    # 价差特征
    df['spread_ma'] = df['spread_bps'].rolling(20).mean()
    df['spread_std'] = df['spread_bps'].rolling(20).std()
    df['spread_compression'] = df['spread_bps'] / df['spread_ma']  # 价差收缩率
    
    # 流动性特征
    df['liquidity_score'] = (df['bid_volume'] + df['ask_volume']) / df['spread_bps']
    df['liquidity_ma'] = df['liquidity_score'].rolling(20).mean()
    
    # 价格压力
    df['price_pressure_ma'] = df['price_pressure'].rolling(20).mean()
    df['price_pressure_std'] = df['price_pressure'].rolling(20).std()
    
    # VWAP偏离
    df['vwap_deviation'] = (df['mid_price'] - df['vwap']) / df['vwap']
    df['vwap_deviation_ma'] = df['vwap_deviation'].rolling(20).mean()
    
    # 新增：订单流强度
    df['order_flow_intensity'] = df['price_pressure'] * df['volume_imbalance']
    df['order_flow_intensity_ma'] = df['order_flow_intensity'].rolling(20).mean()
    
    # 新增：流动性冲击
    df['liquidity_impact'] = df['price_change'] / (df['total_volume_x'] + 1e-8)
    df['liquidity_impact_ma'] = df['liquidity_impact'].rolling(20).mean()
    
    # 新增：买卖压力比率
    df['buy_pressure_ratio'] = df['bid_volume'] / (df['ask_volume'] + 1e-8)
    df['sell_pressure_ratio'] = df['ask_volume'] / (df['bid_volume'] + 1e-8)
    
    return df

def aggregate_to_15m(df: pd.DataFrame) -> pd.DataFrame:
    """聚合到15分钟时间框架"""
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
        'spread_compression': 'mean',
        
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
        'taker_imbalance': 'mean',
        'taker_imbalance_ma': 'mean',
        'order_flow_intensity': 'mean',
        'order_flow_intensity_ma': 'mean',
        
        # 流动性特征
        'liquidity_score': 'mean',
        'liquidity_ma': 'mean',
        'liquidity_pressure': 'mean',
        'liquidity_pressure_ma': 'mean',
        'liquidity_impact': 'mean',
        'liquidity_impact_ma': 'mean',
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
        'buy_pressure_ratio': 'mean',
        'sell_pressure_ratio': 'mean',
        
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
        'bid_fill_prob_change': 'sum',
        'ask_fill_prob_change': 'sum'
    }
    
    # 执行15分钟聚合
    resampled = df.resample('15T').agg(agg_rules)
    
    # 计算聚合后的额外特征
    resampled = calculate_15m_features(resampled)
    
    return resampled.reset_index()

def calculate_15m_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算15分钟聚合后的额外特征"""
    # 价格动量
    df['price_momentum_1'] = df['mid_price'].pct_change(1)
    df['price_momentum_2'] = df['mid_price'].pct_change(2)
    df['price_momentum_4'] = df['mid_price'].pct_change(4)  # 1小时
    
    # 成交量特征
    df['volume_ma_4'] = df['total_volume_x'].rolling(4).mean()  # 1小时
    df['volume_ma_8'] = df['total_volume_x'].rolling(8).mean()  # 2小时
    df['volume_ratio'] = df['total_volume_x'] / df['volume_ma_4']
    
    # 价差特征
    df['spread_ratio'] = df['spread_bps'] / df['spread_ma']
    df['spread_trend'] = df['spread_bps'].rolling(4).mean().pct_change()
    
    # 订单流特征
    df['imbalance_trend'] = df['volume_imbalance'].rolling(4).mean().pct_change()
    df['pressure_trend'] = df['price_pressure'].rolling(4).mean().pct_change()
    df['taker_imbalance_trend'] = df['taker_imbalance'].rolling(4).mean().pct_change()
    
    # 流动性特征
    df['liquidity_trend'] = df['liquidity_score'].rolling(4).mean().pct_change()
    df['liquidity_pressure_trend'] = df['liquidity_pressure'].rolling(4).mean().pct_change()
    df['fill_prob_trend'] = (df['bid_fill_prob'] - df['ask_fill_prob']).rolling(4).mean()
    
    # 波动率特征
    df['volatility_ma'] = df['price_volatility'].rolling(8).mean()  # 2小时
    df['volatility_ratio'] = df['price_volatility'] / df['volatility_ma']
    
    # 异常检测
    df['price_jump'] = (df['price_change_abs'] > df['price_change_abs'].rolling(20).quantile(0.95)).astype(int)
    df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)
    df['spread_widening'] = (df['spread_ratio'] > 1.5).astype(int)
    df['liquidity_crisis'] = (df['liquidity_score'] < df['liquidity_score'].rolling(20).quantile(0.1)).astype(int)
    
    # 新增：订单流强度指标
    df['order_flow_strength'] = df['order_flow_intensity'] * df['volume_ratio']
    df['order_flow_strength_ma'] = df['order_flow_strength'].rolling(8).mean()
    
    # 新增：流动性压力指标
    df['liquidity_stress'] = df['liquidity_pressure'] * df['spread_compression']
    df['liquidity_stress_ma'] = df['liquidity_stress'].rolling(8).mean()
    
    return df

def main():
    print("=== 微观数据聚合到15分钟脚本 ===")
    
    # 加载微观特征数据
    df = load_micro_features()
    
    # 计算增强的订单流特征
    print("计算增强的订单流特征...")
    df = calculate_enhanced_order_flow_features(df)
    
    # 聚合到15分钟
    print("聚合到15分钟...")
    df_15m = aggregate_to_15m(df)
    print(f"15分钟数据: {len(df_15m)} 行")
    
    # 保存结果
    output_file = "data/mid_features_15m_orderflow.parquet"
    df_15m.to_parquet(output_file, index=False)
    print(f"✅ 15分钟订单流特征已保存到: {output_file}")
    
    # 显示特征统计
    print(f"\n📊 特征统计:")
    print(f"总行数: {len(df_15m)}")
    print(f"总列数: {len(df_15m.columns)}")
    print(f"时间范围: {df_15m['timestamp'].min()} 到 {df_15m['timestamp'].max()}")
    
    # 显示新增的高信息特征
    new_features = ['liquidity_pressure', 'taker_imbalance', 'order_flow_intensity', 
                   'liquidity_impact', 'buy_pressure_ratio', 'sell_pressure_ratio',
                   'order_flow_strength', 'liquidity_stress']
    
    print(f"\n🆕 新增高信息特征:")
    for feature in new_features:
        if feature in df_15m.columns:
            print(f"  - {feature}: {df_15m[feature].mean():.6f} (mean)")

if __name__ == "__main__":
    main() 