#!/usr/bin/env python3
"""
简化成本感知回测 - 使用筛选后的特征和训练好的模型
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.calibration import IsotonicRegression
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

def generate_signals(model, df, threshold=0.6, confidence_threshold=0.8):
    """生成交易信号"""
    print("🎯 生成交易信号...")
    
    # 准备特征
    feature_cols = [col for col in df.columns if col not in 
                   ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'date', 'label']]
    
    X = df[feature_cols].fillna(0)
    
    # 预测概率
    dmatrix = xgb.DMatrix(X)
    raw_probs = model.predict(dmatrix)
    
    # 校准概率（简化版）
    # 在实际应用中应该使用训练好的校准器
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

def calculate_returns(signals, holding_period=4, transaction_cost=0.001):
    """计算收益"""
    print("💰 计算收益...")
    
    # 初始化
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
                    'holding_hours': time_held
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
    
    return signals, trades_df

def main():
    print("🚀 简化成本感知回测")
    
    # 回测参数
    threshold = 0.6
    confidence_threshold = 0.8
    holding_period = 4  # 小时
    transaction_cost = 0.001  # 0.1%
    
    # 15m回测
    print("\n📊 15m回测...")
    model_15m, features_15m = load_model_and_features(
        'xgb_15m_optuna_optimized.bin',
        'data/features_15m_selected.parquet'
    )
    
    signals_15m = generate_signals(model_15m, features_15m, threshold, confidence_threshold)
    results_15m, trades_15m = calculate_returns(signals_15m, holding_period, transaction_cost)
    
    # 保存结果
    results_15m.to_csv('backtest_results_15m_selected.csv', index=False)
    if len(trades_15m) > 0:
        trades_15m.to_csv('trades_15m_selected.csv', index=False)
    
    # 5m回测
    print("\n📊 5m回测...")
    model_5m, features_5m = load_model_and_features(
        'xgb_5m_optuna_optimized.bin',
        'data/features_5m_selected.parquet'
    )
    
    signals_5m = generate_signals(model_5m, features_5m, threshold, confidence_threshold)
    results_5m, trades_5m = calculate_returns(signals_5m, holding_period, transaction_cost)
    
    # 保存结果
    results_5m.to_csv('backtest_results_5m_selected.csv', index=False)
    if len(trades_5m) > 0:
        trades_5m.to_csv('trades_5m_selected.csv', index=False)
    
    print("\n✅ 回测完成！")

if __name__ == "__main__":
    main() 