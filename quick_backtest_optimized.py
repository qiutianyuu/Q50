#!/usr/bin/env python3
"""
快速回测优化后的模型
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def backtest_15m():
    """回测15m模型"""
    print("📊 回测15m优化模型...")
    
    # 读取数据和模型
    df = pd.read_parquet('/Users/qiutianyu/data/processed/features_15m_2023_2025.parquet')
    model = xgb.XGBClassifier()
    model.load_model('xgb_15m_optimized.bin')
    
    # 准备特征
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'trend_1h', 'trend_4h']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols].fillna(0)
    
    # 预测
    predictions = model.predict_proba(X)[:, 1]
    
    # 回测参数
    long_threshold = 0.8
    short_threshold = 0.2
    holding_period = 3  # 3根15m = 45分钟
    
    # 初始化回测变量
    position = 0  # 0=空仓, 1=多头, -1=空头
    entry_price = 0
    entry_time = None
    entry_idx = None  # 添加entry_idx变量
    trades = []
    pnl = 0
    
    for i in range(len(df) - holding_period):
        current_time = df.iloc[i]['timestamp']
        current_price = df.iloc[i]['close']
        pred = predictions[i]
        
        # 检查趋势过滤
        trend_1h = df.iloc[i]['trend_1h']
        trend_4h = df.iloc[i]['trend_4h']
        
        # 开仓逻辑
        if position == 0:  # 空仓
            if pred > long_threshold and trend_1h == 1 and trend_4h == 1:
                position = 1
                entry_price = current_price
                entry_time = current_time
                entry_idx = i  # 设置entry_idx
            elif pred < short_threshold and trend_1h == -1 and trend_4h == -1:
                position = -1
                entry_price = current_price
                entry_time = current_time
                entry_idx = i  # 设置entry_idx
        
        # 平仓逻辑
        elif position != 0:
            # 检查是否到达持仓期
            if i >= entry_idx + holding_period:
                exit_price = df.iloc[i]['close']
                
                # 计算收益
                if position == 1:  # 多头
                    trade_pnl = (exit_price - entry_price) / entry_price - 0.001  # 手续费
                else:  # 空头
                    trade_pnl = (entry_price - exit_price) / entry_price - 0.001  # 手续费
                
                pnl += trade_pnl
                
                # 记录交易
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': trade_pnl,
                    'pred': pred
                })
                
                # 重置仓位
                position = 0
                entry_price = 0
                entry_time = None
    
    # 统计结果
    if trades:
        trades_df = pd.DataFrame(trades)
        win_rate = (trades_df['pnl'] > 0).mean()
        avg_pnl = trades_df['pnl'].mean()
        total_return = pnl
        num_trades = len(trades)
        
        print(f"交易次数: {num_trades}")
        print(f"胜率: {win_rate:.2%}")
        print(f"平均收益: {avg_pnl:.4f}")
        print(f"总收益: {total_return:.4f}")
        print(f"年化收益: {total_return * 365 / 800:.2%}")  # 假设800天
        
        # 保存交易记录
        trades_df.to_csv('backtest_15m_optimized.csv', index=False)
        print("✅ 15m回测结果已保存: backtest_15m_optimized.csv")
        
        return trades_df, total_return, win_rate
    else:
        print("❌ 没有产生交易信号")
        return None, 0, 0

def backtest_5m():
    """回测5m模型"""
    print("\n📊 回测5m优化模型...")
    
    # 读取数据和模型
    df = pd.read_parquet('/Users/qiutianyu/data/processed/features_5m_2023_2025.parquet')
    model = xgb.XGBClassifier()
    model.load_model('xgb_5m_optimized.bin')
    
    # 准备特征
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'trend_1h', 'trend_4h']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols].fillna(0)
    
    # 预测
    predictions = model.predict_proba(X)[:, 1]
    
    # 回测参数
    long_threshold = 0.8
    short_threshold = 0.2
    holding_period = 6  # 6根5m = 30分钟
    
    # 初始化回测变量
    position = 0
    entry_price = 0
    entry_time = None
    trades = []
    pnl = 0
    
    for i in range(len(df) - holding_period):
        current_time = df.iloc[i]['timestamp']
        current_price = df.iloc[i]['close']
        pred = predictions[i]
        
        # 检查趋势过滤
        trend_1h = df.iloc[i]['trend_1h']
        trend_4h = df.iloc[i]['trend_4h']
        
        # 开仓逻辑
        if position == 0:
            if pred > long_threshold and trend_1h == 1 and trend_4h == 1:
                position = 1
                entry_price = current_price
                entry_time = current_time
            elif pred < short_threshold and trend_1h == -1 and trend_4h == -1:
                position = -1
                entry_price = current_price
                entry_time = current_time
        
        # 平仓逻辑
        elif position != 0:
            # 检查是否到达持仓期
            if i >= entry_time_idx + holding_period:
                exit_price = df.iloc[i]['close']
                
                # 计算收益
                if position == 1:
                    trade_pnl = (exit_price - entry_price) / entry_price - 0.001
                else:
                    trade_pnl = (entry_price - exit_price) / entry_price - 0.001
                
                pnl += trade_pnl
                
                # 记录交易
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': trade_pnl,
                    'pred': pred
                })
                
                # 重置仓位
                position = 0
                entry_price = 0
                entry_time = None
    
    # 统计结果
    if trades:
        trades_df = pd.DataFrame(trades)
        win_rate = (trades_df['pnl'] > 0).mean()
        avg_pnl = trades_df['pnl'].mean()
        total_return = pnl
        num_trades = len(trades)
        
        print(f"交易次数: {num_trades}")
        print(f"胜率: {win_rate:.2%}")
        print(f"平均收益: {avg_pnl:.4f}")
        print(f"总收益: {total_return:.4f}")
        print(f"年化收益: {total_return * 365 / 800:.2%}")
        
        # 保存交易记录
        trades_df.to_csv('backtest_5m_optimized.csv', index=False)
        print("✅ 5m回测结果已保存: backtest_5m_optimized.csv")
        
        return trades_df, total_return, win_rate
    else:
        print("❌ 没有产生交易信号")
        return None, 0, 0

def main():
    print("🎯 快速回测优化后的模型...")
    
    # 回测15m
    trades_15m, return_15m, winrate_15m = backtest_15m()
    
    # 回测5m
    trades_5m, return_5m, winrate_5m = backtest_5m()
    
    # 结果总结
    print("\n📊 回测结果总结:")
    print(f"15m模型 - 总收益: {return_15m:.4f}, 胜率: {winrate_15m:.2%}")
    print(f"5m模型 - 总收益: {return_5m:.4f}, 胜率: {winrate_5m:.2%}")
    
    print("\n🎉 回测完成!")

if __name__ == "__main__":
    main() 