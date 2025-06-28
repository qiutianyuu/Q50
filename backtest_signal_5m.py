#!/usr/bin/env python3
"""
RexKing – 5m XGBoost信号简易回测

用训练好的5m模型信号做3根K线持有回测，统计收益、胜率、最大回撤等。
"""
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb

# 路径配置
DATA_DIR = Path("/Users/qiutianyu/data/processed")
FEATURES_FILE = DATA_DIR / "features_5m_2023_2025.parquet"
MODEL_FILE = "xgb_5m_model.bin"

# 回测参数
LONG_TH = 0.8   # 做多阈值 (提高)
SHORT_TH = 0.2  # 做空阈值 (降低)
HOLD_BARS = 3   # 持有K线数 (延长到3根5m)
FEE = 0.0004    # 单边手续费（可调）

# 读取数据
print("📥 读取特征数据...")
df = pd.read_parquet(FEATURES_FILE)

# 过滤数值特征
exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'trend_1h', 'trend_4h']
feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
X = df[feature_cols].fillna(0)

# 加载模型
print("🔮 加载模型...")
model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)

# 预测概率
print("⚡ 生成信号...")
df['pred_proba'] = model.predict_proba(X)[:, 1]

# 生成交易信号
# 1=做多，-1=做空，0=空仓
signal = np.where(df['pred_proba'] > LONG_TH, 1, np.where(df['pred_proba'] < SHORT_TH, -1, 0))

# 1h + 4h 双重趋势过滤
# 做多：trend_1h=1 且 trend_4h=1
# 做空：trend_1h=0 且 trend_4h=0
signal = np.where((signal == 1) & (df['trend_1h'] == 1) & (df['trend_4h'] == 1), 1,
                  np.where((signal == -1) & (df['trend_1h'] == 0) & (df['trend_4h'] == 0), -1, 0))
df['signal'] = signal

# 计算未来1根K线收益
# 收益=log(下根收盘/本根收盘) - 手续费
df['ret_1'] = np.log(df['close'].shift(-HOLD_BARS) / df['close'])
df['trade_ret'] = df['signal'] * df['ret_1'] - np.abs(df['signal']) * FEE

# 累计收益
df['cum_ret'] = df['trade_ret'].cumsum()

# 统计指标
n_trades = (df['signal'] != 0).sum()
win_rate = (df['trade_ret'] > 0).sum() / n_trades if n_trades > 0 else 0
cum_pnl = df['trade_ret'].sum()
max_dd = (df['cum_ret'].cummax() - df['cum_ret']).max()

print(f"\n📊 回测结果:")
print(f"总信号数: {n_trades}")
print(f"胜率: {win_rate:.3f}")
print(f"累计收益: {cum_pnl:.4f}")
print(f"最大回撤: {max_dd:.4f}")

# 保存信号与回测结果
out_csv = DATA_DIR / "backtest_5m_signals.csv"
df[['timestamp', 'close', 'pred_proba', 'signal', 'trade_ret', 'cum_ret']].to_csv(out_csv, index=False)
print(f"信号与回测明细已保存: {out_csv}") 