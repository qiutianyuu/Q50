#!/usr/bin/env python3
"""
测试订单流特征集成
只使用有订单流数据的时间段进行测试
"""

import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os

def load_order_flow_data():
    """加载订单流数据"""
    files = glob.glob("data/mid_features_5m_*.parquet")
    if not files:
        raise FileNotFoundError("No order flow files found")
    
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading order flow: {latest_file}")
    df = pd.read_parquet(latest_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Order flow data: {len(df)} rows, {len(df.columns)} columns")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df

def create_mock_kline_data(order_flow_df):
    """创建模拟K线数据，与订单流数据时间对齐"""
    print("Creating mock K-line data...")
    
    # 使用订单流数据的时间戳作为基础
    timestamps = order_flow_df['timestamp'].copy()
    
    # 创建模拟K线数据
    mock_data = []
    base_price = 2400.0
    
    for i, ts in enumerate(timestamps):
        # 模拟价格变动
        price_change = np.random.normal(0, 0.002)  # 0.2%标准差
        current_price = base_price * (1 + price_change)
        
        # 模拟OHLC
        high = current_price * (1 + abs(np.random.normal(0, 0.001)))
        low = current_price * (1 - abs(np.random.normal(0, 0.001)))
        open_price = current_price * (1 + np.random.normal(0, 0.0005))
        close_price = current_price
        
        # 模拟成交量
        volume = np.random.exponential(1000) + 500
        
        mock_data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
        
        base_price = close_price
    
    df = pd.DataFrame(mock_data)
    print(f"Mock K-line data: {len(df)} rows")
    return df

def add_technical_features(df):
    """添加技术指标"""
    print("Adding technical features...")
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # 基础动量指标
    df['ret_1'] = close.pct_change()
    df['ret_3'] = close.pct_change(3)
    df['ret_6'] = close.pct_change(6)
    df['ret_12'] = close.pct_change(12)
    
    # 波动率
    df['volatility_10'] = close.pct_change().rolling(10).std()
    df['volatility_24'] = close.pct_change().rolling(24).std()
    
    # 移动平均
    df['ma_10'] = close.rolling(10).mean()
    df['ma_20'] = close.rolling(20).mean()
    df['ma_50'] = close.rolling(50).mean()
    
    # 成交量指标
    df['vol_ma20'] = volume.rolling(20).mean()
    df['vol_ratio'] = volume / df['vol_ma20']
    
    return df

def integrate_order_flow_features(kline_df, order_flow_df):
    """集成订单流特征"""
    print("Integrating order flow features...")
    
    # 确保时间戳格式一致
    kline_df['timestamp'] = pd.to_datetime(kline_df['timestamp'])
    order_flow_df['timestamp'] = pd.to_datetime(order_flow_df['timestamp'])
    
    # 合并数据
    merged_df = pd.merge(
        kline_df,
        order_flow_df,
        on='timestamp',
        how='inner'
    )
    
    print(f"Merged data: {len(merged_df)} rows")
    return merged_df

def add_labels(df):
    """添加标签"""
    print("Adding labels...")
    
    # 简单的方向标签：下一根K线涨跌
    df['label'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # 移除最后一行（无法计算标签）
    df = df.dropna(subset=['label']).reset_index(drop=True)
    
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df

def main():
    print("=== 订单流特征集成测试 ===")
    
    # 加载订单流数据
    order_flow_df = load_order_flow_data()
    
    # 创建模拟K线数据
    kline_df = create_mock_kline_data(order_flow_df)
    
    # 添加技术指标
    kline_df = add_technical_features(kline_df)
    
    # 集成订单流特征
    merged_df = integrate_order_flow_features(kline_df, order_flow_df)
    
    # 添加标签
    final_df = add_labels(merged_df)
    
    # 清理数据
    print("Cleaning data...")
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns
    final_df[numeric_cols] = final_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    final_df = final_df.fillna(0)
    
    print(f"Final data: {len(final_df)} rows, {len(final_df.columns)} columns")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/test_features_5m_orderflow_{timestamp}.parquet"
    final_df.to_parquet(output_file, index=False)
    print(f"Saved to: {output_file}")
    
    # 显示特征统计
    feature_cols = [col for col in final_df.columns if col not in ['timestamp', 'label']]
    order_flow_features = len([col for col in feature_cols if col in order_flow_df.columns])
    technical_features = len([col for col in feature_cols if col not in order_flow_df.columns])
    
    print(f"\n=== Feature Statistics ===")
    print(f"Total features: {len(feature_cols)}")
    print(f"Order flow features: {order_flow_features}")
    print(f"Technical features: {technical_features}")
    
    # 显示样本数据
    print(f"\n=== Sample Data ===")
    sample_cols = ['timestamp', 'close', 'ret_1', 'volatility_10', 'price_momentum_1m', 
                  'volume_ratio', 'spread_ratio', 'imbalance_trend', 'label']
    available_cols = [col for col in sample_cols if col in final_df.columns]
    print(final_df[available_cols].head(10))

if __name__ == "__main__":
    main() 