#!/usr/bin/env python3
"""
处理websocket订单流数据，聚合到15分钟级别
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_websocket_data():
    """加载所有websocket数据"""
    print("正在加载websocket数据...")
    
    # 加载orderbook数据
    orderbook_files = glob.glob('data/websocket/**/orderbook_*.parquet', recursive=True)
    orderbook_files.extend(glob.glob('data/websocket/orderbook_*.parquet'))
    
    print(f"找到 {len(orderbook_files)} 个orderbook文件")
    
    orderbook_dfs = []
    for file in orderbook_files:
        try:
            df = pd.read_parquet(file)
            if len(df) > 0 and 'timestamp' in df.columns:
                orderbook_dfs.append(df)
        except Exception as e:
            print(f"跳过文件 {file}: {e}")
    
    if orderbook_dfs:
        orderbook_df = pd.concat(orderbook_dfs, ignore_index=True)
        orderbook_df = orderbook_df.drop_duplicates(subset=['timestamp', 'ts'])
        orderbook_df = orderbook_df.sort_values('timestamp')
        print(f"Orderbook数据: {len(orderbook_df)} 行")
    else:
        orderbook_df = pd.DataFrame()
    
    # 加载trades数据
    trades_files = glob.glob('data/websocket/**/trades_*.parquet', recursive=True)
    trades_files.extend(glob.glob('data/websocket/trades_*.parquet'))
    
    print(f"找到 {len(trades_files)} 个trades文件")
    
    trades_dfs = []
    for file in trades_files:
        try:
            df = pd.read_parquet(file)
            if len(df) > 0 and 'timestamp' in df.columns:
                trades_dfs.append(df)
        except Exception as e:
            print(f"跳过文件 {file}: {e}")
    
    if trades_dfs:
        trades_df = pd.concat(trades_dfs, ignore_index=True)
        trades_df = trades_df.drop_duplicates(subset=['timestamp', 'trade_id'])
        trades_df = trades_df.sort_values('timestamp')
        print(f"Trades数据: {len(trades_df)} 行")
    else:
        trades_df = pd.DataFrame()
    
    return orderbook_df, trades_df

def aggregate_orderbook_features(orderbook_df):
    """聚合orderbook特征到15分钟"""
    if len(orderbook_df) == 0:
        return pd.DataFrame()
    
    print("正在聚合orderbook特征...")
    
    # 确保timestamp是datetime
    orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'])
    
    # 创建15分钟时间桶
    orderbook_df['time_bucket'] = orderbook_df['timestamp'].dt.floor('15T')
    
    # 计算orderbook特征
    features = []
    
    for bucket, group in orderbook_df.groupby('time_bucket'):
        if len(group) == 0:
            continue
            
        # 使用最新的orderbook数据
        latest = group.iloc[-1]
        
        # 基础价格特征
        mid_price = (latest['bid1_price'] + latest['ask1_price']) / 2
        
        # 流动性特征
        bid_liquidity = sum([latest[f'bid{i}_size'] for i in range(1, 6)])
        ask_liquidity = sum([latest[f'ask{i}_size'] for i in range(1, 6)])
        total_liquidity = bid_liquidity + ask_liquidity
        
        # 流动性不平衡
        liquidity_imbalance = (bid_liquidity - ask_liquidity) / total_liquidity if total_liquidity > 0 else 0
        
        # 价格压力特征
        bid_pressure = sum([latest[f'bid{i}_size'] * latest[f'bid{i}_price'] for i in range(1, 6)])
        ask_pressure = sum([latest[f'ask{i}_size'] * latest[f'ask{i}_price'] for i in range(1, 6)])
        
        # VWAP计算
        bid_vwap = bid_pressure / bid_liquidity if bid_liquidity > 0 else mid_price
        ask_vwap = ask_pressure / ask_liquidity if ask_liquidity > 0 else mid_price
        
        # 价格不平衡
        price_imbalance = (bid_vwap - ask_vwap) / mid_price
        
        # 深度特征
        depth_1 = latest['bid1_size'] + latest['ask1_size']
        depth_5 = total_liquidity
        
        # 价差特征
        spread = latest['spread']
        spread_bps = latest['spread_bps']
        
        # 订单簿倾斜度
        bid_slope = (latest['bid1_price'] - latest['bid5_price']) / latest['bid5_price']
        ask_slope = (latest['ask5_price'] - latest['ask1_price']) / latest['ask1_price']
        
        features.append({
            'timestamp': bucket,
            'mid_price': mid_price,
            'bid_liquidity': bid_liquidity,
            'ask_liquidity': ask_liquidity,
            'total_liquidity': total_liquidity,
            'liquidity_imbalance': liquidity_imbalance,
            'bid_vwap': bid_vwap,
            'ask_vwap': ask_vwap,
            'price_imbalance': price_imbalance,
            'depth_1': depth_1,
            'depth_5': depth_5,
            'spread': spread,
            'spread_bps': spread_bps,
            'bid_slope': bid_slope,
            'ask_slope': ask_slope,
            'bid1_price': latest['bid1_price'],
            'ask1_price': latest['ask1_price'],
        })
    
    return pd.DataFrame(features)

def aggregate_trades_features(trades_df):
    """聚合trades特征到15分钟"""
    if len(trades_df) == 0:
        return pd.DataFrame()
    
    print("正在聚合trades特征...")
    
    # 确保timestamp是datetime
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    # 创建15分钟时间桶
    trades_df['time_bucket'] = trades_df['timestamp'].dt.floor('15T')
    
    # 计算trades特征
    features = []
    
    for bucket, group in trades_df.groupby('time_bucket'):
        if len(group) == 0:
            continue
        
        # 基础统计
        total_volume = group['size'].sum()
        trade_count = len(group)
        avg_trade_size = total_volume / trade_count if trade_count > 0 else 0
        
        # 买卖不平衡
        buy_volume = group[group['side'] == 'buy']['size'].sum()
        sell_volume = group[group['side'] == 'sell']['size'].sum()
        volume_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
        
        # 价格统计
        vwap = (group['price'] * group['size']).sum() / total_volume if total_volume > 0 else 0
        price_std = group['price'].std()
        price_range = group['price'].max() - group['price'].min()
        
        # 大单统计 (假设>0.1 ETH为大单)
        large_trades = group[group['size'] > 0.1]
        large_trade_volume = large_trades['size'].sum()
        large_trade_ratio = large_trade_volume / total_volume if total_volume > 0 else 0
        
        # 时间分布
        time_span = (group['timestamp'].max() - group['timestamp'].min()).total_seconds()
        trades_per_second = trade_count / time_span if time_span > 0 else 0
        
        features.append({
            'timestamp': bucket,
            'total_volume': total_volume,
            'trade_count': trade_count,
            'avg_trade_size': avg_trade_size,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'volume_imbalance': volume_imbalance,
            'vwap': vwap,
            'price_std': price_std,
            'price_range': price_range,
            'large_trade_volume': large_trade_volume,
            'large_trade_ratio': large_trade_ratio,
            'trades_per_second': trades_per_second,
        })
    
    return pd.DataFrame(features)

def merge_features(orderbook_features, trades_features):
    """合并orderbook和trades特征"""
    print("正在合并特征...")
    
    # 合并特征
    if len(orderbook_features) > 0 and len(trades_features) > 0:
        merged = pd.merge(orderbook_features, trades_features, on='timestamp', how='outer')
    elif len(orderbook_features) > 0:
        merged = orderbook_features
    elif len(trades_features) > 0:
        merged = trades_features
    else:
        return pd.DataFrame()
    
    # 按时间排序
    merged = merged.sort_values('timestamp')
    
    # 填充缺失值
    merged = merged.fillna(method='ffill').fillna(0)
    
    print(f"合并后特征: {len(merged)} 行, {len(merged.columns)} 列")
    return merged

def add_rolling_features(df):
    """添加滚动特征"""
    print("正在添加滚动特征...")
    
    # 价格变化
    df['price_change'] = df['mid_price'].pct_change()
    df['price_change_5'] = df['mid_price'].pct_change(5)
    df['price_change_15'] = df['mid_price'].pct_change(15)
    
    # 流动性变化
    df['liquidity_change'] = df['total_liquidity'].pct_change()
    df['liquidity_imbalance_change'] = df['liquidity_imbalance'].diff()
    
    # 成交量变化
    df['volume_change'] = df['total_volume'].pct_change()
    df['volume_imbalance_change'] = df['volume_imbalance'].diff()
    
    # 滚动统计
    for window in [5, 15, 30]:
        df[f'price_volatility_{window}'] = df['price_change'].rolling(window).std()
        df[f'volume_ma_{window}'] = df['total_volume'].rolling(window).mean()
        df[f'liquidity_ma_{window}'] = df['total_liquidity'].rolling(window).mean()
        df[f'vwap_ma_{window}'] = df['vwap'].rolling(window).mean()
    
    # 价差变化
    df['spread_change'] = df['spread'].pct_change()
    df['spread_bps_change'] = df['spread_bps'].diff()
    
    return df

def main():
    """主函数"""
    print("开始处理websocket订单流数据...")
    
    # 加载数据
    orderbook_df, trades_df = load_websocket_data()
    
    if len(orderbook_df) == 0 and len(trades_df) == 0:
        print("没有找到有效的websocket数据")
        return
    
    # 聚合特征
    orderbook_features = aggregate_orderbook_features(orderbook_df)
    trades_features = aggregate_trades_features(trades_df)
    
    # 合并特征
    merged_features = merge_features(orderbook_features, trades_features)
    
    if len(merged_features) == 0:
        print("没有生成有效特征")
        return
    
    # 添加滚动特征
    final_features = add_rolling_features(merged_features)
    
    # 保存结果
    output_file = 'data/websocket_features_15m.parquet'
    final_features.to_parquet(output_file, compression='zstd')
    
    print(f"特征已保存到: {output_file}")
    print(f"特征维度: {final_features.shape}")
    print(f"时间范围: {final_features['timestamp'].min()} 到 {final_features['timestamp'].max()}")
    print(f"特征列: {list(final_features.columns)}")
    
    # 显示样本数据
    print("\n样本数据:")
    print(final_features.head())

if __name__ == "__main__":
    main() 