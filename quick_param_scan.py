#!/usr/bin/env python3
"""
快速参数扫描 - 小参数空间快速验证
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from itertools import product
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

def generate_signals_with_params(model, df, threshold, confidence_threshold):
    """生成交易信号"""
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
    
    return signals

def calculate_returns_with_params(signals, holding_period, transaction_cost):
    """计算收益"""
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
    
    return signals, trades

def evaluate_strategy(trades_df):
    """评估策略性能"""
    if len(trades_df) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_returns': 0,
            'avg_returns': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'calmar_ratio': 0,
            'profit_factor': 0
        }
    
    # 基础指标
    total_trades = len(trades_df)
    win_rate = len(trades_df[trades_df['returns'] > 0]) / total_trades
    total_returns = trades_df['returns'].sum()
    avg_returns = trades_df['returns'].mean()
    max_drawdown = trades_df['returns'].min()
    
    # 计算累积收益序列
    cumulative_returns = trades_df['returns'].cumsum()
    max_cumulative = cumulative_returns.expanding().max()
    drawdown = cumulative_returns - max_cumulative
    
    # 风险调整指标
    returns_std = trades_df['returns'].std()
    sharpe_ratio = avg_returns / returns_std if returns_std > 0 else 0
    
    # Calmar比率 (年化收益 / 最大回撤)
    # 假设平均持仓4小时，一年8760小时，年化交易次数 = 8760/4 = 2190
    annualized_returns = total_returns * (2190 / total_trades) if total_trades > 0 else 0
    calmar_ratio = annualized_returns / abs(max_drawdown) if max_drawdown < 0 else 0
    
    # 盈亏比
    winning_trades = trades_df[trades_df['returns'] > 0]['returns'].sum()
    losing_trades = abs(trades_df[trades_df['returns'] < 0]['returns'].sum())
    profit_factor = winning_trades / losing_trades if losing_trades > 0 else float('inf')
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_returns': total_returns,
        'avg_returns': avg_returns,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio,
        'profit_factor': profit_factor
    }

def quick_scan_parameters(model, df, timeframe):
    """快速扫描参数空间"""
    print(f"🔍 快速参数扫描: {timeframe}")
    
    # 小参数网格 - 快速验证
    thresholds = [0.55, 0.6, 0.65, 0.7]  # 减少到4个
    confidence_thresholds = [0.7, 0.8, 0.9]  # 减少到3个
    holding_periods = [4, 8, 12, 24]  # 减少到4个
    transaction_costs = [0.001, 0.0015]  # 减少到2个
    
    results = []
    total_combinations = len(thresholds) * len(confidence_thresholds) * len(holding_periods) * len(transaction_costs)
    current = 0
    
    print(f"📊 总共需要测试 {total_combinations} 种参数组合")
    
    for threshold, conf_thresh, holding_period, tx_cost in product(
        thresholds, confidence_thresholds, holding_periods, transaction_costs
    ):
        current += 1
        print(f"⏳ 进度: {current}/{total_combinations} ({current/total_combinations*100:.1f}%)")
        
        # 生成信号
        signals = generate_signals_with_params(model, df, threshold, conf_thresh)
        
        # 计算收益
        _, trades = calculate_returns_with_params(signals, holding_period, tx_cost)
        
        # 评估策略
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            metrics = evaluate_strategy(trades_df)
            
            results.append({
                'threshold': threshold,
                'confidence_threshold': conf_thresh,
                'holding_period': holding_period,
                'transaction_cost': tx_cost,
                **metrics
            })
    
    return pd.DataFrame(results)

def find_optimal_parameters(results_df):
    """找到最优参数"""
    print("🎯 寻找最优参数...")
    
    # 过滤掉交易次数太少的组合
    filtered_df = results_df[results_df['total_trades'] >= 5].copy()
    
    if len(filtered_df) == 0:
        print("⚠️ 没有足够的有效参数组合")
        return None
    
    # 按不同指标排序
    print("\n📊 按不同指标的最优参数:")
    
    # 1. 按总收益排序
    best_by_returns = filtered_df.loc[filtered_df['total_returns'].idxmax()]
    print(f"🏆 最高总收益: {best_by_returns['total_returns']:.4f}")
    print(f"   参数: threshold={best_by_returns['threshold']}, conf={best_by_returns['confidence_threshold']}, "
          f"holding={best_by_returns['holding_period']}h, cost={best_by_returns['transaction_cost']:.4f}")
    print(f"   交易数: {best_by_returns['total_trades']}, 胜率: {best_by_returns['win_rate']:.2%}")
    
    # 2. 按Sharpe比率排序
    best_by_sharpe = filtered_df.loc[filtered_df['sharpe_ratio'].idxmax()]
    print(f"📈 最高Sharpe: {best_by_sharpe['sharpe_ratio']:.4f}")
    print(f"   参数: threshold={best_by_sharpe['threshold']}, conf={best_by_sharpe['confidence_threshold']}, "
          f"holding={best_by_sharpe['holding_period']}h, cost={best_by_sharpe['transaction_cost']:.4f}")
    print(f"   交易数: {best_by_sharpe['total_trades']}, 胜率: {best_by_sharpe['win_rate']:.2%}")
    
    # 3. 按Calmar比率排序
    best_by_calmar = filtered_df.loc[filtered_df['calmar_ratio'].idxmax()]
    print(f"🛡️ 最高Calmar: {best_by_calmar['calmar_ratio']:.4f}")
    print(f"   参数: threshold={best_by_calmar['threshold']}, conf={best_by_calmar['confidence_threshold']}, "
          f"holding={best_by_calmar['holding_period']}h, cost={best_by_calmar['transaction_cost']:.4f}")
    print(f"   交易数: {best_by_calmar['total_trades']}, 胜率: {best_by_calmar['win_rate']:.2%}")
    
    # 4. 按盈亏比排序
    best_by_profit_factor = filtered_df.loc[filtered_df['profit_factor'].idxmax()]
    print(f"💰 最高盈亏比: {best_by_profit_factor['profit_factor']:.4f}")
    print(f"   参数: threshold={best_by_profit_factor['threshold']}, conf={best_by_profit_factor['confidence_threshold']}, "
          f"holding={best_by_profit_factor['holding_period']}h, cost={best_by_profit_factor['transaction_cost']:.4f}")
    print(f"   交易数: {best_by_profit_factor['total_trades']}, 胜率: {best_by_profit_factor['win_rate']:.2%}")
    
    return {
        'best_by_returns': best_by_returns.to_dict(),
        'best_by_sharpe': best_by_sharpe.to_dict(),
        'best_by_calmar': best_by_calmar.to_dict(),
        'best_by_profit_factor': best_by_profit_factor.to_dict()
    }

def main():
    print("🚀 快速参数扫描优化器")
    
    # 扫描15m参数
    print("\n📊 扫描15m参数...")
    model_15m, features_15m = load_model_and_features(
        'xgb_15m_optuna_optimized.bin',
        'data/features_15m_selected.parquet'
    )
    
    results_15m = quick_scan_parameters(model_15m, features_15m, '15m')
    results_15m.to_csv('quick_param_scan_results_15m.csv', index=False)
    
    optimal_15m = find_optimal_parameters(results_15m)
    
    # 扫描5m参数
    print("\n📊 扫描5m参数...")
    model_5m, features_5m = load_model_and_features(
        'xgb_5m_optuna_optimized.bin',
        'data/features_5m_selected.parquet'
    )
    
    results_5m = quick_scan_parameters(model_5m, features_5m, '5m')
    results_5m.to_csv('quick_param_scan_results_5m.csv', index=False)
    
    optimal_5m = find_optimal_parameters(results_5m)
    
    # 保存最优参数
    optimal_params = {
        '15m': optimal_15m,
        '5m': optimal_5m
    }
    
    import json
    with open('quick_optimal_parameters.json', 'w') as f:
        json.dump(optimal_params, f, indent=2, default=str)
    
    print("\n✅ 快速参数扫描完成！")
    print("📁 结果文件:")
    print("  - quick_param_scan_results_15m.csv")
    print("  - quick_param_scan_results_5m.csv")
    print("  - quick_optimal_parameters.json")

if __name__ == "__main__":
    main() 