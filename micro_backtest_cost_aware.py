#!/usr/bin/env python3
"""
成本感知微观特征回测脚本
使用训练好的XGBoost模型进行回测，高阈值长持仓
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from datetime import datetime
import glob
import os

def load_model_and_data(model_file, features_file):
    """加载模型和数据"""
    print(f"Loading model: {model_file}")
    model = joblib.load(model_file)
    
    print(f"Loading features: {features_file}")
    df = pd.read_parquet(features_file)
    print(f"Loaded {len(df)} rows")
    
    return model, df

def prepare_features(df):
    """准备特征"""
    exclude_cols = ['timestamp', 'label', 'bid_price', 'ask_price', 'mid_price', 'rel_spread', 
                   'future_price', 'price_change', 'cost_threshold']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].fillna(0)
    return X, feature_cols

def run_backtest(model, df, threshold_long=0.9, threshold_short=0.1, holding_period=100, fee_rate=0.0005):
    """运行回测"""
    print(f"Running backtest with thresholds: long={threshold_long}, short={threshold_short}")
    print(f"Holding period: {holding_period} steps, Fee rate: {fee_rate}")
    
    # 准备特征
    X, feature_cols = prepare_features(df)
    
    # 获取预测概率
    y_proba = model.predict_proba(X)
    
    # 初始化回测变量
    positions = []  # 持仓记录
    trades = []     # 交易记录
    current_position = 0  # 当前持仓: 0=空仓, 1=多头, -1=空头
    entry_price = 0
    entry_time = None
    holding_count = 0
    
    # 遍历每个时间点
    for i in range(len(df)):
        timestamp = df.iloc[i]['timestamp']
        mid_price = df.iloc[i]['mid_price']
        
        # 获取预测概率 (注意：模型输出是[Short, Neutral, Long])
        prob_short = y_proba[i, 0]   # Short
        prob_neutral = y_proba[i, 1] # Neutral  
        prob_long = y_proba[i, 2]    # Long
        
        # 交易信号
        signal = 0
        if prob_long > threshold_long:
            signal = 1
        elif prob_short > threshold_short:
            signal = -1
        
        # 持仓管理
        if current_position == 0:  # 空仓
            if signal != 0:
                # 开仓
                current_position = signal
                entry_price = mid_price
                entry_time = timestamp
                holding_count = 0
                print(f"开仓: {timestamp} 价格={mid_price:.2f} 信号={signal} 概率={prob_long if signal==1 else prob_short:.3f}")
        else:
            holding_count += 1
            
            # 检查是否平仓
            if holding_count >= holding_period:
                # 平仓
                exit_price = mid_price
                pnl = (exit_price - entry_price) / entry_price * current_position - fee_rate * 2  # 开仓+平仓费用
                
                trade = {
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': current_position,
                    'pnl': pnl,
                    'holding_period': holding_count,
                    'prob_long': prob_long,
                    'prob_short': prob_short
                }
                trades.append(trade)
                
                print(f"平仓: {timestamp} 价格={exit_price:.2f} PnL={pnl:.4f}")
                
                # 重置
                current_position = 0
                entry_price = 0
                entry_time = None
                holding_count = 0
    
    # 统计结果
    if trades:
        trades_df = pd.DataFrame(trades)
        
        # 计算统计指标
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        max_drawdown = trades_df['pnl'].cumsum().min()
        
        # 按持仓方向分析
        long_trades = trades_df[trades_df['position'] == 1]
        short_trades = trades_df[trades_df['position'] == -1]
        
        print(f"\n=== 回测结果 ===")
        print(f"总交易次数: {total_trades}")
        print(f"盈利交易: {winning_trades}")
        print(f"胜率: {win_rate:.2%}")
        print(f"总收益: {total_pnl:.4f}")
        print(f"平均收益: {avg_pnl:.4f}")
        print(f"最大回撤: {max_drawdown:.4f}")
        
        if len(long_trades) > 0:
            long_win_rate = len(long_trades[long_trades['pnl'] > 0]) / len(long_trades)
            long_avg_pnl = long_trades['pnl'].mean()
            print(f"多头交易: {len(long_trades)}次, 胜率{long_win_rate:.2%}, 平均收益{long_avg_pnl:.4f}")
        
        if len(short_trades) > 0:
            short_win_rate = len(short_trades[short_trades['pnl'] > 0]) / len(short_trades)
            short_avg_pnl = short_trades['pnl'].mean()
            print(f"空头交易: {len(short_trades)}次, 胜率{short_win_rate:.2%}, 平均收益{short_avg_pnl:.4f}")
        
        return trades_df
    else:
        print("没有产生任何交易")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description='Cost-aware micro features backtest')
    parser.add_argument('--model', type=str, help='Model file path')
    parser.add_argument('--features', type=str, help='Features file path')
    parser.add_argument('--threshold_long', type=float, default=0.9, help='Long threshold')
    parser.add_argument('--threshold_short', type=float, default=0.1, help='Short threshold')
    parser.add_argument('--holding', type=int, default=100, help='Holding period')
    parser.add_argument('--fee', type=float, default=0.0005, help='Fee rate')
    
    args = parser.parse_args()
    
    # 如果没有指定文件，使用最新的
    if not args.model:
        model_files = glob.glob("xgb_micro_cost_aware_*.bin")
        if model_files:
            args.model = max(model_files, key=os.path.getctime)
        else:
            print("No model files found")
            return
    
    if not args.features:
        feature_files = glob.glob("data/micro_features_cost_aware_labels_*.parquet")
        if feature_files:
            args.features = max(feature_files, key=os.path.getctime)
        else:
            print("No feature files found")
            return
    
    # 加载模型和数据
    model, df = load_model_and_data(args.model, args.features)
    
    # 运行回测
    trades_df = run_backtest(
        model, df, 
        threshold_long=args.threshold_long,
        threshold_short=args.threshold_short,
        holding_period=args.holding,
        fee_rate=args.fee
    )
    
    # 保存结果
    if not trades_df.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"micro_backtest_results_cost_aware_{timestamp}.csv"
        trades_df.to_csv(output_file, index=False)
        print(f"\n回测结果已保存: {output_file}")

if __name__ == "__main__":
    main() 