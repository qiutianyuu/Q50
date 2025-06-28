#!/usr/bin/env python3
"""
Walk-Forward训练与参数优化
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 数据路径
FEATURES_FILE = "/Users/qiutianyu/features_offline_15m.parquet"
OUTPUT_FILE = "walk_forward_results.csv"

def load_data():
    """加载数据"""
    print("📥 加载特征数据...")
    df = pd.read_parquet(FEATURES_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"数据形状: {df.shape}")
    return df

def prepare_features(df):
    """准备特征"""
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_1h', 'close_4h', 'label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols

def backtest_period(df, model, feature_cols, start_idx, end_idx, thresholds, hold_period=3, fee_rate=0.0004):
    """回测特定时间段"""
    period_df = df.iloc[start_idx:end_idx].copy()
    
    if len(period_df) == 0:
        return None
    
    # 预测
    X = period_df[feature_cols].fillna(0)
    proba = model.predict_proba(X)
    
    # 生成信号
    long_th, short_th = thresholds
    signals = pd.DataFrame({
        'timestamp': period_df['timestamp'],
        'close': period_df['close'],
        'prob_up': proba[:, 1],
        'prob_down': proba[:, 0],
        'signal': 0
    })
    
    signals.loc[signals['prob_up'] > long_th, 'signal'] = 1
    signals.loc[signals['prob_down'] > short_th, 'signal'] = -1
    
    # 计算收益
    signals['entry_price'] = signals['close']
    signals['exit_price'] = signals['close'].shift(-hold_period)
    signals = signals.dropna(subset=['exit_price'])
    
    if len(signals) == 0:
        return None
    
    signals['ret'] = (signals['exit_price'] - signals['entry_price']) / signals['entry_price'] * signals['signal']
    signals['ret_net'] = signals['ret'] - fee_rate * 2
    signals['cum_ret'] = (1 + signals['ret_net']).cumprod() - 1
    
    # 统计
    win_rate = (signals['ret_net'] > 0).mean()
    total_ret = signals['cum_ret'].iloc[-1] if len(signals) > 0 else 0
    max_dd = (signals['cum_ret'].cummax() - signals['cum_ret']).max() if len(signals) > 0 else 0
    signal_count = len(signals[signals['signal'] != 0])
    
    return {
        'win_rate': win_rate,
        'total_ret': total_ret,
        'max_dd': max_dd,
        'signal_count': signal_count,
        'start_date': period_df['timestamp'].iloc[0],
        'end_date': period_df['timestamp'].iloc[-1]
    }

def walk_forward_optimization(df, feature_cols):
    """Walk-Forward优化"""
    print("🔄 开始Walk-Forward优化...")
    
    # 参数网格
    threshold_combinations = [
        (0.7, 0.3), (0.75, 0.25), (0.8, 0.2), (0.85, 0.15), (0.9, 0.1),
        (0.65, 0.35), (0.6, 0.4), (0.55, 0.45)
    ]
    hold_periods = [2, 3, 4, 5]
    
    results = []
    
    # 时间窗口设置
    train_days = 90  # 训练90天
    test_days = 7    # 测试7天
    step_days = 7    # 每7天向前滚动
    
    # 转换为索引
    total_days = (df['timestamp'].max() - df['timestamp'].min()).days
    train_periods = int((total_days - train_days) / step_days)
    
    print(f"总训练周期数: {train_periods}")
    
    for period in range(train_periods):
        # 计算时间窗口
        start_date = df['timestamp'].min() + timedelta(days=period * step_days)
        train_start = start_date
        train_end = train_start + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)
        
        # 找到对应的索引
        train_mask = (df['timestamp'] >= train_start) & (df['timestamp'] < train_end)
        test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] < test_end)
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if len(train_df) < 1000 or len(test_df) < 100:  # 数据太少跳过
            continue
        
        print(f"周期 {period+1}/{train_periods}: {start_date.date()} -> {test_end.date()}")
        
        # 训练模型
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['label']
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # 测试不同参数组合
        for long_th, short_th in threshold_combinations:
            for hold_period in hold_periods:
                result = backtest_period(
                    df, model, feature_cols, 
                    test_df.index[0], test_df.index[-1],
                    (long_th, short_th), hold_period
                )
                
                if result:
                    result.update({
                        'period': period + 1,
                        'long_threshold': long_th,
                        'short_threshold': short_th,
                        'hold_period': hold_period
                    })
                    results.append(result)
    
    return pd.DataFrame(results)

def analyze_results(results_df):
    """分析结果"""
    print("\n📊 Walk-Forward结果分析:")
    
    # 按参数分组统计
    param_stats = results_df.groupby(['long_threshold', 'short_threshold', 'hold_period']).agg({
        'win_rate': ['mean', 'std'],
        'total_ret': ['mean', 'std'],
        'max_dd': ['mean', 'std'],
        'signal_count': 'mean'
    }).round(4)
    
    print("\n参数组合统计:")
    print(param_stats)
    
    # 找出最佳参数组合
    best_by_ret = results_df.loc[results_df['total_ret'].idxmax()]
    best_by_sharpe = results_df.loc[(results_df['total_ret'] / (results_df['max_dd'] + 0.01)).idxmax()]
    
    print(f"\n🏆 最佳收益参数:")
    print(f"长期阈值: {best_by_ret['long_threshold']}")
    print(f"短期阈值: {best_by_ret['short_threshold']}")
    print(f"持仓期: {best_by_ret['hold_period']}")
    print(f"平均收益: {best_by_ret['total_ret']:.4f}")
    print(f"平均胜率: {best_by_ret['win_rate']:.4f}")
    print(f"平均回撤: {best_by_ret['max_dd']:.4f}")
    
    print(f"\n🏆 最佳夏普参数:")
    print(f"长期阈值: {best_by_sharpe['long_threshold']}")
    print(f"短期阈值: {best_by_sharpe['short_threshold']}")
    print(f"持仓期: {best_by_sharpe['hold_period']}")
    print(f"平均收益: {best_by_sharpe['total_ret']:.4f}")
    print(f"平均胜率: {best_by_sharpe['win_rate']:.4f}")
    print(f"平均回撤: {best_by_sharpe['max_dd']:.4f}")
    
    return best_by_ret, best_by_sharpe

def main():
    """主函数"""
    print("🚀 开始Walk-Forward优化...")
    
    # 加载数据
    df = load_data()
    feature_cols = prepare_features(df)
    
    # Walk-Forward优化
    results_df = walk_forward_optimization(df, feature_cols)
    
    # 保存结果
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 结果已保存到: {OUTPUT_FILE}")
    
    # 分析结果
    best_by_ret, best_by_sharpe = analyze_results(results_df)
    
    print(f"\n✅ Walk-Forward优化完成!")

if __name__ == "__main__":
    main() 