#!/usr/bin/env python3
"""
优化回测 - 使用参数扫描找到的最优参数
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import warnings
warnings.filterwarnings('ignore')

def load_model_and_features(model_path, features_path):
    """加载模型和特征数据"""
    print(f"📁 加载模型: {model_path}")
    model = xgb.Booster()
    model.load_model(model_path)
    
    print(f"📁 加载特征: {features_path}")
    df = pd.read_parquet(features_path)
    
    return model, df

def generate_signals_with_optimal_params(model, df, optimal_params):
    """使用最优参数生成交易信号"""
    print("🎯 使用最优参数生成信号...")
    
    threshold = optimal_params['threshold']
    confidence_threshold = optimal_params['confidence_threshold']
    
    # 准备特征
    feature_cols = [col for col in df.columns if col not in 
                   ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'date', 'label']]
    
    X = df[feature_cols].fillna(0)
    
    # 预测概率
    dmatrix = xgb.DMatrix(X)
    raw_probs = model.predict(dmatrix)
    
    # 校准概率（简化版）
    calibrated_probs = np.clip(raw_probs, 0.01, 0.99)
    
    # 生成信号
    signals = pd.DataFrame({
        'timestamp': df['timestamp'],
        'close': df['close'],
        'raw_prob': raw_probs,
        'calibrated_prob': calibrated_probs,
        'signal': 0
    })
    
    # 应用阈值
    signals.loc[calibrated_probs > threshold, 'signal'] = 1
    signals.loc[calibrated_probs < (1 - threshold), 'signal'] = -1
    
    # 应用置信度过滤
    high_confidence = (calibrated_probs > confidence_threshold) | (calibrated_probs < (1 - confidence_threshold))
    signals.loc[~high_confidence, 'signal'] = 0
    
    print(f"📊 信号统计:")
    print(f"  多头信号: {len(signals[signals['signal'] == 1])}")
    print(f"  空头信号: {len(signals[signals['signal'] == -1])}")
    print(f"  无信号: {len(signals[signals['signal'] == 0])}")
    
    return signals

def calculate_returns_with_optimal_params(signals, optimal_params):
    """使用最优参数计算收益"""
    print("💰 使用最优参数计算收益...")
    
    holding_period = optimal_params['holding_period']
    transaction_cost = optimal_params['transaction_cost']
    
    signals = signals.copy()
    signals['position'] = 0
    signals['returns'] = 0.0
    signals['cumulative_returns'] = 0.0
    signals['trade_id'] = 0
    
    current_position = 0
    entry_price = 0
    entry_time = None
    trade_id = 0
    trades = []
    
    for i in range(len(signals)):
        current_time = signals.iloc[i]['timestamp']
        current_price = signals.iloc[i]['close']
        current_signal = signals.iloc[i]['signal']
        
        # 检查是否需要平仓
        if current_position != 0 and entry_time is not None:
            time_held = (current_time - entry_time).total_seconds() / 3600  # 小时
            
            if time_held >= holding_period:
                # 平仓
                if current_position == 1:  # 多头平仓
                    returns = (current_price - entry_price) / entry_price - transaction_cost
                else:  # 空头平仓
                    returns = (entry_price - current_price) / entry_price - transaction_cost
                
                # 记录交易
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': current_position,
                    'returns': returns,
                    'holding_hours': time_held,
                    'prob_at_entry': signals.iloc[i]['calibrated_prob']
                })
                
                # 更新信号
                signals.iloc[i, signals.columns.get_loc('returns')] = returns
                signals.iloc[i, signals.columns.get_loc('trade_id')] = trade_id
                trade_id += 1
                
                # 重置
                current_position = 0
                entry_price = 0
                entry_time = None
        
        # 开新仓
        if current_position == 0 and current_signal != 0:
            current_position = current_signal
            entry_price = current_price
            entry_time = current_time
        
        # 更新持仓
        signals.iloc[i, signals.columns.get_loc('position')] = current_position
    
    # 计算累积收益
    signals['cumulative_returns'] = signals['returns'].cumsum()
    
    # 转换为DataFrame
    trades_df = pd.DataFrame(trades)
    
    print(f"📊 交易统计:")
    print(f"  总交易数: {len(trades_df)}")
    if len(trades_df) > 0:
        print(f"  胜率: {len(trades_df[trades_df['returns'] > 0]) / len(trades_df):.2%}")
        print(f"  平均收益: {trades_df['returns'].mean():.4%}")
        print(f"  总收益: {trades_df['returns'].sum():.4%}")
        print(f"  最大回撤: {trades_df['returns'].min():.4%}")
        print(f"  平均持仓时间: {trades_df['holding_hours'].mean():.1f}小时")
        
        # 计算年化收益
        total_days = (signals['timestamp'].max() - signals['timestamp'].min()).days
        annualized_returns = trades_df['returns'].sum() * (365 / total_days) if total_days > 0 else 0
        print(f"  年化收益: {annualized_returns:.2%}")
    
    return signals, trades_df

def analyze_trade_quality(trades_df):
    """分析交易质量"""
    if len(trades_df) == 0:
        return
    
    print("\n🔍 交易质量分析:")
    
    # 按概率分组分析
    trades_df['prob_group'] = pd.cut(trades_df['prob_at_entry'], 
                                    bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0], 
                                    labels=['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
    
    prob_analysis = trades_df.groupby('prob_group').agg({
        'returns': ['count', 'mean', 'sum'],
        'position': 'count'
    }).round(4)
    
    print("按预测概率分组的交易表现:")
    print(prob_analysis)
    
    # 按持仓方向分析
    direction_analysis = trades_df.groupby('position').agg({
        'returns': ['count', 'mean', 'sum'],
        'holding_hours': 'mean'
    }).round(4)
    
    print("\n按持仓方向分析:")
    print(direction_analysis)

def main():
    print("🚀 优化回测 - 使用最优参数")
    
    # 加载最优参数
    with open('quick_optimal_parameters.json', 'r') as f:
        optimal_params = json.load(f)
    
    # 15m优化回测
    print("\n📊 15m优化回测...")
    model_15m, features_15m = load_model_and_features(
        'xgb_15m_optuna_optimized.bin',
        'data/features_15m_selected.parquet'
    )
    
    # 使用Sharpe最优参数
    optimal_15m_params = optimal_params['15m']['best_by_sharpe']
    print(f"🎯 使用参数: threshold={optimal_15m_params['threshold']}, "
          f"conf={optimal_15m_params['confidence_threshold']}, "
          f"holding={optimal_15m_params['holding_period']}h, "
          f"cost={optimal_15m_params['transaction_cost']:.4f}")
    
    signals_15m = generate_signals_with_optimal_params(model_15m, features_15m, optimal_15m_params)
    results_15m, trades_15m = calculate_returns_with_optimal_params(signals_15m, optimal_15m_params)
    
    # 保存结果
    results_15m.to_csv('optimized_backtest_results_15m.csv', index=False)
    if len(trades_15m) > 0:
        trades_15m.to_csv('optimized_trades_15m.csv', index=False)
        analyze_trade_quality(trades_15m)
    
    # 5m优化回测
    print("\n📊 5m优化回测...")
    model_5m, features_5m = load_model_and_features(
        'xgb_5m_optuna_optimized.bin',
        'data/features_5m_selected.parquet'
    )
    
    # 使用Sharpe最优参数
    optimal_5m_params = optimal_params['5m']['best_by_sharpe']
    print(f"🎯 使用参数: threshold={optimal_5m_params['threshold']}, "
          f"conf={optimal_5m_params['confidence_threshold']}, "
          f"holding={optimal_5m_params['holding_period']}h, "
          f"cost={optimal_5m_params['transaction_cost']:.4f}")
    
    signals_5m = generate_signals_with_optimal_params(model_5m, features_5m, optimal_5m_params)
    results_5m, trades_5m = calculate_returns_with_optimal_params(signals_5m, optimal_5m_params)
    
    # 保存结果
    results_5m.to_csv('optimized_backtest_results_5m.csv', index=False)
    if len(trades_5m) > 0:
        trades_5m.to_csv('optimized_trades_5m.csv', index=False)
        analyze_trade_quality(trades_5m)
    
    print("\n✅ 优化回测完成！")
    print("📁 结果文件:")
    print("  - optimized_backtest_results_15m.csv")
    print("  - optimized_trades_15m.csv")
    print("  - optimized_backtest_results_5m.csv")
    print("  - optimized_trades_5m.csv")

if __name__ == "__main__":
    main() 