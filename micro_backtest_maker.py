#!/usr/bin/env python3
"""
微观特征Maker模式回测脚本
使用最佳模型进行回测
"""

import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
import json
import glob
import os
import argparse

def load_latest_model_and_features():
    """加载最新模型和特征"""
    # 加载特征
    files = glob.glob("data/realtime_features_with_labels_*.parquet")
    if not files:
        raise FileNotFoundError("No features with labels files found")
    latest_file = max(files, key=os.path.getctime)
    df = pd.read_parquet(latest_file)
    
    # 加载最佳模型 (h60_a0.3_maker)
    model_files = glob.glob("xgb_micro_label_h60_a0.3_maker_*.bin")
    if not model_files:
        raise FileNotFoundError("No model files found")
    latest_model = max(model_files, key=os.path.getctime)
    
    model = xgb.XGBClassifier()
    model.load_model(latest_model)
    
    print(f"加载特征: {latest_file}")
    print(f"加载模型: {latest_model}")
    
    return df, model

def load_model_and_features_from_args(model_path, features_path):
    """从命令行参数加载模型和特征"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    df = pd.read_parquet(features_path)
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    print(f"加载特征: {features_path}")
    print(f"加载模型: {model_path}")
    
    return df, model

def prepare_features(df):
    """准备特征列"""
    exclude_cols = ['timestamp', 'label_h60_a0.3_maker', 'label_h120_a0.6_maker', 'label_h120_a0.3_maker']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return df[feature_cols]

def backtest_maker_strategy(df, model, threshold_long=0.7, threshold_short=0.3, holding_steps=60):
    """Maker模式回测"""
    print(f"\n=== Maker模式回测 ===")
    print(f"阈值: long>{threshold_long}, short<{threshold_short}")
    print(f"持仓步数: {holding_steps}")
    
    # 准备特征
    X = prepare_features(df)
    
    # 预测概率
    proba = model.predict_proba(X)[:, 1]  # long概率
    
    # 生成信号
    signals = np.zeros(len(df))
    signals[proba > threshold_long] = 1    # long信号
    signals[proba < threshold_short] = -1  # short信号
    
    # 回测逻辑
    positions = []
    trades = []
    current_position = 0
    entry_price = 0
    entry_step = 0
    
    for i in range(len(df)):
        mid_price = df.iloc[i]['mid_price']
        rel_spread = df.iloc[i]['rel_spread']
        
        # 检查是否需要平仓
        if current_position != 0 and i - entry_step >= holding_steps:
            # 计算Maker模式下的成交价
            if current_position == 1:  # long平仓
                exit_price = mid_price - 0.5 * (mid_price * rel_spread)  # 卖单成交价
            else:  # short平仓
                exit_price = mid_price + 0.5 * (mid_price * rel_spread)  # 买单成交价
            
            # 计算收益
            if current_position == 1:
                pnl = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) / entry_price
            
            # 扣除Maker手续费 (0.0001)
            pnl -= 0.0001
            
            trades.append({
                'entry_step': entry_step,
                'exit_step': i,
                'position': current_position,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'holding_steps': i - entry_step
            })
            
            current_position = 0
        
        # 检查新信号
        if current_position == 0 and signals[i] != 0:
            current_position = signals[i]
            entry_price = mid_price
            entry_step = i
    
    # 处理未平仓的头寸
    if current_position != 0:
        mid_price = df.iloc[-1]['mid_price']
        rel_spread = df.iloc[-1]['rel_spread']
        
        if current_position == 1:
            exit_price = mid_price - 0.5 * (mid_price * rel_spread)
        else:
            exit_price = mid_price + 0.5 * (mid_price * rel_spread)
        
        if current_position == 1:
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price
        
        pnl -= 0.0001
        
        trades.append({
            'entry_step': entry_step,
            'exit_step': len(df) - 1,
            'position': current_position,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'holding_steps': len(df) - 1 - entry_step
        })
    
    return trades

def analyze_results(trades):
    """分析回测结果"""
    if not trades:
        print("没有交易")
        return
    
    df_trades = pd.DataFrame(trades)
    
    # 基础统计
    total_trades = len(trades)
    winning_trades = (df_trades['pnl'] > 0).sum()
    losing_trades = (df_trades['pnl'] < 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 收益统计
    total_return = df_trades['pnl'].sum()
    avg_return = df_trades['pnl'].mean()
    std_return = df_trades['pnl'].std()
    sharpe = avg_return / std_return if std_return > 0 else 0
    
    # 最大回撤
    cumulative_returns = df_trades['pnl'].cumsum()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    print(f"\n📊 回测结果:")
    print(f"总交易数: {total_trades}")
    print(f"盈利交易: {winning_trades}")
    print(f"亏损交易: {losing_trades}")
    print(f"胜率: {win_rate:.1%}")
    print(f"总收益: {total_return:.4f}")
    print(f"平均收益: {avg_return:.4f}")
    print(f"收益标准差: {std_return:.4f}")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {max_drawdown:.1%}")
    
    # 按持仓方向分析
    long_trades = df_trades[df_trades['position'] == 1]
    short_trades = df_trades[df_trades['position'] == -1]
    
    if len(long_trades) > 0:
        long_win_rate = (long_trades['pnl'] > 0).sum() / len(long_trades)
        long_avg_return = long_trades['pnl'].mean()
        print(f"\n📈 Long交易: {len(long_trades)}笔, 胜率{long_win_rate:.1%}, 平均收益{long_avg_return:.4f}")
    
    if len(short_trades) > 0:
        short_win_rate = (short_trades['pnl'] > 0).sum() / len(short_trades)
        short_avg_return = short_trades['pnl'].mean()
        print(f"📉 Short交易: {len(short_trades)}笔, 胜率{short_win_rate:.1%}, 平均收益{short_avg_return:.4f}")
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'avg_return': avg_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown
    }

def main():
    parser = argparse.ArgumentParser(description='Micro Maker Backtest')
    parser.add_argument('--model', type=str, help='Path to XGBoost model file')
    parser.add_argument('--features', type=str, help='Path to features parquet file')
    parser.add_argument('--json-out', type=str, default='backtest_results.json', help='Output JSON file path')
    parser.add_argument('--threshold-long', type=float, default=0.7, help='Long threshold')
    parser.add_argument('--threshold-short', type=float, default=0.3, help='Short threshold')
    parser.add_argument('--holding-steps', type=int, default=60, help='Holding period in steps')
    
    args = parser.parse_args()
    
    # 加载数据
    if args.model and args.features:
        df, model = load_model_and_features_from_args(args.model, args.features)
    else:
        df, model = load_latest_model_and_features()
    
    # 回测参数
    thresholds = [
        (args.threshold_long, args.threshold_short),
    ]
    
    holding_steps = [args.holding_steps]
    
    all_results = []
    
    for threshold_long, threshold_short in thresholds:
        for holding in holding_steps:
            print(f"\n{'='*50}")
            trades = backtest_maker_strategy(df, model, threshold_long, threshold_short, holding)
            results = analyze_results(trades)
            
            if results:
                results.update({
                    'threshold_long': threshold_long,
                    'threshold_short': threshold_short,
                    'holding_steps': holding
                })
                all_results.append(results)
    
    # 保存结果
    with open(args.json_out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n结果已保存: {args.json_out}")

if __name__ == "__main__":
    main() 