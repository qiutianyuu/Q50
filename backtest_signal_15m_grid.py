#!/usr/bin/env python3
"""
RexKing – 15m XGBoost信号网格回测

评估不同阈值组合和止损设置对策略表现的影响
"""
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb

# 路径配置
DATA_DIR = Path("/Users/qiutianyu/data/processed")
FEATURES_FILE = DATA_DIR / "features_15m_2023_2025.parquet"
MODEL_FILE = "xgb_15m_model.bin"

# 网格搜索参数
THRESHOLD_COMBINATIONS = [
    # (long_th, short_th, trend_filter_type)
    (0.65, 0.35, "1h_only"),      # 降低阈值，只用1h趋势
    (0.60, 0.40, "4h_only"),      # 进一步降低，只用4h趋势  
    (0.55, 0.45, "no_trend"),     # 最低阈值，无趋势过滤
    (0.70, 0.30, "both_trend"),   # 原始设置作为对比
]

STOP_LOSS_OPTIONS = [0.0, -0.01, -0.015, -0.02]  # 0%, -1%, -1.5%, -2%

def run_backtest(df, model, long_th, short_th, trend_filter, stop_loss, hold_bars=3, fee=0.0004):
    """运行单次回测"""
    
    # 生成交易信号
    signal = np.where(df['pred_proba'] > long_th, 1, np.where(df['pred_proba'] < short_th, -1, 0))
    
    # 趋势过滤
    if trend_filter == "1h_only":
        signal = np.where((signal == 1) & (df['trend_1h'] == 1), 1,
                          np.where((signal == -1) & (df['trend_1h'] == 0), -1, 0))
    elif trend_filter == "4h_only":
        signal = np.where((signal == 1) & (df['trend_4h'] == 1), 1,
                          np.where((signal == -1) & (df['trend_4h'] == 0), -1, 0))
    elif trend_filter == "both_trend":
        signal = np.where((signal == 1) & (df['trend_1h'] == 1) & (df['trend_4h'] == 1), 1,
                          np.where((signal == -1) & (df['trend_1h'] == 0) & (df['trend_4h'] == 0), -1, 0))
    # no_trend: 不做过滤
    
    df['signal'] = signal
    
    # 计算收益（带止损）
    ret_raw = df['close'].shift(-hold_bars) / df['close'] - 1
    if stop_loss < 0:
        df['ret_1'] = np.clip(ret_raw, stop_loss, None)
    else:
        df['ret_1'] = ret_raw
    
    df['trade_ret'] = df['signal'] * df['ret_1'] - np.abs(df['signal']) * fee
    df['cum_ret'] = df['trade_ret'].cumsum()
    
    # 统计指标
    n_trades = (df['signal'] != 0).sum()
    if n_trades == 0:
        return {
            'long_th': long_th, 'short_th': short_th, 'trend_filter': trend_filter,
            'stop_loss': stop_loss, 'n_trades': 0, 'win_rate': 0, 'cum_pnl': 0,
            'max_dd': 0, 'sharpe': 0, 'avg_win': 0, 'avg_loss': 0
        }
    
    win_rate = (df['trade_ret'] > 0).sum() / n_trades
    cum_pnl = df['trade_ret'].sum()
    max_dd = (df['cum_ret'].cummax() - df['cum_ret']).max()
    
    # 计算夏普比率（年化）
    returns = df['trade_ret'].dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe = (returns.mean() * 96 * 365) / (returns.std() * np.sqrt(96 * 365))  # 15m数据年化
    else:
        sharpe = 0
    
    # 平均盈亏
    wins = df[df['trade_ret'] > 0]['trade_ret']
    losses = df[df['trade_ret'] < 0]['trade_ret']
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    
    return {
        'long_th': long_th, 'short_th': short_th, 'trend_filter': trend_filter,
        'stop_loss': stop_loss, 'n_trades': n_trades, 'win_rate': win_rate,
        'cum_pnl': cum_pnl, 'max_dd': max_dd, 'sharpe': sharpe,
        'avg_win': avg_win, 'avg_loss': avg_loss
    }

def main():
    print("📥 读取特征数据...")
    df = pd.read_parquet(FEATURES_FILE)
    
    # 添加缺失的特征
    df['year'] = df['timestamp'].dt.year
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_50'] = df['close'].rolling(50).mean()
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_pct'] = df['volume'].pct_change()
    
    print("🔮 加载模型...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    
    # 预测概率
    expected_features = model.feature_names_in_
    X = df[expected_features].fillna(0)
    df['pred_proba'] = model.predict_proba(X)[:, 1]
    
    print("⚡ 开始网格搜索...")
    results = []
    
    for long_th, short_th, trend_filter in THRESHOLD_COMBINATIONS:
        for stop_loss in STOP_LOSS_OPTIONS:
            print(f"测试: long_th={long_th}, short_th={short_th}, trend={trend_filter}, stop_loss={stop_loss}")
            
            result = run_backtest(df, model, long_th, short_th, trend_filter, stop_loss)
            results.append(result)
    
    # 转换为DataFrame并排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['cum_pnl', 'sharpe'], ascending=[False, False])
    
    print("\n" + "="*80)
    print("📊 网格搜索结果 (按累计收益排序)")
    print("="*80)
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # 保存结果
    results_df.to_csv('grid_search_15m_results.csv', index=False)
    print(f"\n✅ 结果已保存到: grid_search_15m_results.csv")
    
    # 显示最佳组合
    best = results_df.iloc[0]
    print(f"\n🏆 最佳组合:")
    print(f"阈值: {best['long_th']:.2f}/{best['short_th']:.2f}")
    print(f"趋势过滤: {best['trend_filter']}")
    print(f"止损: {best['stop_loss']:.3f}")
    print(f"信号数: {best['n_trades']}")
    print(f"胜率: {best['win_rate']:.3f}")
    print(f"累计收益: {best['cum_pnl']:.4f}")
    print(f"夏普比率: {best['sharpe']:.3f}")
    print(f"最大回撤: {best['max_dd']:.4f}")

if __name__ == "__main__":
    main() 