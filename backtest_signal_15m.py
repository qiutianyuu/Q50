#!/usr/bin/env python3
"""
RexKing – 15m XGBoost信号简易回测

用训练好的15m模型信号做1根K线持有回测，统计收益、胜率、最大回撤等。
"""
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb

# 路径配置
DATA_DIR = Path("/Users/qiutianyu/data/processed")
FEATURES_FILE = DATA_DIR / "features_15m_2023_2025.parquet"
MODEL_FILE = "xgb_15m_model.bin"

# 回测参数
LONG_TH = 0.7   # 做多阈值 (降低)
SHORT_TH = 0.3  # 做空阈值 (提高)
HOLD_BARS = 3   # 持有K线数 (延长到3根15m)
FEE = 0.0004    # 单边手续费（可调）

# 读取数据
print("📥 读取特征数据...")
df = pd.read_parquet(FEATURES_FILE)

# 加载模型
print("🔮 加载模型...")
model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)

# 获取模型期望的特征名
expected_features = model.feature_names_in_
print(f"模型期望特征数: {len(expected_features)}")

# 添加缺失的特征
df['year'] = df['timestamp'].dt.year
df['returns'] = df['close'].pct_change()
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
df['ma_20'] = df['close'].rolling(20).mean()
df['ma_50'] = df['close'].rolling(50).mean()
df['volume_ma_20'] = df['volume'].rolling(20).mean()
df['volume_pct'] = df['volume'].pct_change()

# 确保所有期望的特征都存在
missing_features = [f for f in expected_features if f not in df.columns]
if missing_features:
    print(f"❌ 仍有缺失特征: {missing_features}")
    exit(1)

# 按模型期望的顺序排列特征
X = df[expected_features].fillna(0)

print(f"✅ 特征矩阵形状: {X.shape}")

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
out_csv = DATA_DIR / "backtest_15m_signals.csv"
df[['timestamp', 'close', 'pred_proba', 'signal', 'trade_ret', 'cum_ret']].to_csv(out_csv, index=False)
print(f"信号与回测明细已保存: {out_csv}") 