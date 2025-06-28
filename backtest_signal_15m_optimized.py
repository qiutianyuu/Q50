#!/usr/bin/env python3
"""
RexKing – 15m XGBoost信号优化回测

使用网格搜索找到的最佳参数进行回测
"""
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb

# 路径配置
DATA_DIR = Path("/Users/qiutianyu/data/processed")
FEATURES_FILE = DATA_DIR / "features_15m_2023_2025.parquet"
MODEL_FILE = "xgb_15m_model.bin"

# 最佳参数（来自网格搜索）
LONG_TH = 0.60      # 做多阈值
SHORT_TH = 0.40     # 做空阈值
TREND_FILTER = "4h_only"  # 只用4h趋势过滤
STOP_LOSS = -0.01   # -1%止损
HOLD_BARS = 3       # 持有K线数
FEE = 0.0004        # 单边手续费

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
    
    print("⚡ 生成信号...")
    
    # 生成交易信号
    signal = np.where(df['pred_proba'] > LONG_TH, 1, np.where(df['pred_proba'] < SHORT_TH, -1, 0))
    
    # 4h趋势过滤
    signal = np.where((signal == 1) & (df['trend_4h'] == 1), 1,
                      np.where((signal == -1) & (df['trend_4h'] == 0), -1, 0))
    df['signal'] = signal
    
    # 计算收益（带止损）
    ret_raw = df['close'].shift(-HOLD_BARS) / df['close'] - 1
    df['ret_1'] = np.clip(ret_raw, STOP_LOSS, None)
    
    df['trade_ret'] = df['signal'] * df['ret_1'] - np.abs(df['signal']) * FEE
    df['cum_ret'] = df['trade_ret'].cumsum()
    
    # 统计指标
    n_trades = (df['signal'] != 0).sum()
    win_rate = (df['trade_ret'] > 0).sum() / n_trades if n_trades > 0 else 0
    cum_pnl = df['trade_ret'].sum()
    max_dd = (df['cum_ret'].cummax() - df['cum_ret']).max()
    
    # 计算夏普比率（年化）
    returns = df['trade_ret'].dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe = (returns.mean() * 96 * 365) / (returns.std() * np.sqrt(96 * 365))
    else:
        sharpe = 0
    
    # 平均盈亏
    wins = df[df['trade_ret'] > 0]['trade_ret']
    losses = df[df['trade_ret'] < 0]['trade_ret']
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    
    # 信号分布
    long_signals = (df['signal'] == 1).sum()
    short_signals = (df['signal'] == -1).sum()
    
    print(f"\n📊 优化回测结果:")
    print(f"参数设置:")
    print(f"  阈值: {LONG_TH}/{SHORT_TH}")
    print(f"  趋势过滤: {TREND_FILTER}")
    print(f"  止损: {STOP_LOSS:.1%}")
    print(f"  持有时间: {HOLD_BARS}根K线")
    print(f"")
    print(f"交易统计:")
    print(f"  总信号数: {n_trades}")
    print(f"  做多信号: {long_signals}")
    print(f"  做空信号: {short_signals}")
    print(f"  胜率: {win_rate:.3f}")
    print(f"  累计收益: {cum_pnl:.4f}")
    print(f"  最大回撤: {max_dd:.4f}")
    print(f"  夏普比率: {sharpe:.3f}")
    print(f"  平均盈利: {avg_win:.4f}")
    print(f"  平均亏损: {avg_loss:.4f}")
    
    # 年化收益
    days = (df['timestamp'].max() - df['timestamp'].min()).days
    annual_return = (cum_pnl / days) * 365
    print(f"  年化收益: {annual_return:.2%}")
    
    # 保存详细结果
    out_csv = DATA_DIR / "backtest_15m_optimized.csv"
    df[['timestamp', 'close', 'pred_proba', 'signal', 'trade_ret', 'cum_ret']].to_csv(out_csv, index=False)
    print(f"\n✅ 详细结果已保存: {out_csv}")
    
    # 显示部分交易记录
    trades = df[df['signal'] != 0].copy()
    if len(trades) > 0:
        print(f"\n📈 前10笔交易:")
        print(trades[['timestamp', 'close', 'pred_proba', 'signal', 'trade_ret']].head(10).to_string(index=False))

if __name__ == "__main__":
    main() 