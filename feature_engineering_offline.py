#!/usr/bin/env python3
"""
基于 ETHUSDT_15m_full.parquet 生成多时空特征
"""
import pandas as pd
import numpy as np
from ta.momentum import rsi, stoch, stoch_signal
from ta.volatility import average_true_range, bollinger_hband, bollinger_lband
from ta.trend import macd, ema_indicator
from ta.volume import volume_weighted_average_price
import warnings
warnings.filterwarnings('ignore')

SRC = "/Users/qiutianyu/ETHUSDT_15m_full.parquet"
OUT = "/Users/qiutianyu/features_offline_15m.parquet"

print("📥 读取合并数据...")
df = pd.read_parquet(SRC)
print(f"原始数据: {df.shape}")
print(f"时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

# === 价格动量特征 ===
print("🔄 生成价格动量特征...")
df["ret_15m"] = np.log(df["close"]/df["close"].shift(1))
df["ret_1h"] = np.log(df["close_1h"]/df["close_1h"].shift(4))  # 4×15m = 1h
df["ret_4h"] = np.log(df["close_4h"]/df["close_4h"].shift(16))  # 16×15m = 4h

# 多周期RSI
df["rsi_14"] = rsi(df["close"], window=14)
df["rsi_30"] = rsi(df["close"], window=30)
df["rsi_1h"] = rsi(df["close_1h"], window=14)

# 价格位置特征
df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
df["price_position_1h"] = (df["close_1h"] - df["low"]) / (df["high"] - df["low"])

# === 波动率特征 ===
print("🔄 生成波动率特征...")
df["atr_14"] = average_true_range(df["high"], df["low"], df["close"], 14)
df["atr_30"] = average_true_range(df["high"], df["low"], df["close"], 30)

# 布林带
df["bb_upper"] = bollinger_hband(df["close"], window=20, window_dev=2)
df["bb_lower"] = bollinger_lband(df["close"], window=20, window_dev=2)
df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

# 价格波动率
df["volatility_5"] = df["ret_15m"].rolling(5).std()
df["volatility_20"] = df["ret_15m"].rolling(20).std()

# === 趋势特征 ===
print("🔄 生成趋势特征...")
# MACD - 修复参数和返回值
df["macd"] = macd(df["close"], window_slow=26, window_fast=12)
df["macd_signal"] = df["macd"].rolling(9).mean()
df["macd_hist"] = df["macd"] - df["macd_signal"]

# EMA
df["ema_12"] = ema_indicator(df["close"], window=12)
df["ema_26"] = ema_indicator(df["close"], window=26)
df["ema_50"] = ema_indicator(df["close"], window=50)

# 趋势强度
df["trend_strength"] = (df["ema_12"] - df["ema_26"]) / df["ema_26"]
df["trend_strength_1h"] = (df["close"] - df["close_1h"]) / df["close_1h"]
df["trend_strength_4h"] = (df["close"] - df["close_4h"]) / df["close_4h"]

# === 成交量特征 ===
print("🔄 生成成交量特征...")
df["volume_ma_5"] = df["volume"].rolling(5).mean()
df["volume_ma_20"] = df["volume"].rolling(20).mean()
df["volume_ratio"] = df["volume"] / df["volume_ma_20"]

# VWAP
df["vwap"] = volume_weighted_average_price(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=20)
df["vwap_position"] = (df["close"] - df["vwap"]) / df["vwap"]

# 成交量标准化
df["volume_zscore"] = (np.log1p(df["volume"]) - np.log1p(df["volume"]).rolling(96).mean()) / \
                      np.log1p(df["volume"]).rolling(96).std()

# === 跨周期特征 ===
print("🔄 生成跨周期特征...")
# 价格偏离度
df["price_deviation_1h"] = (df["close"] - df["close_1h"]) / df["close_1h"]
df["price_deviation_4h"] = (df["close"] - df["close_4h"]) / df["close_4h"]

# 动量对比
df["momentum_1h_vs_15m"] = df["ret_1h"] - df["ret_15m"]
df["momentum_4h_vs_1h"] = df["ret_4h"] - df["ret_1h"]

# === 技术指标组合 ===
print("🔄 生成技术指标组合...")
df["stoch_k"] = stoch(df["high"], df["low"], df["close"], window=14, smooth_window=3)
df["stoch_d"] = stoch_signal(df["high"], df["low"], df["close"], window=14, smooth_window=3)

# 价格动量
df["price_momentum_5"] = df["close"].pct_change(5)
df["price_momentum_10"] = df["close"].pct_change(10)
df["price_momentum_20"] = df["close"].pct_change(20)

# === 目标标签 ===
print("🔄 生成目标标签...")
horizon = 3  # 预测未来3根15分钟K线
threshold = 0.003  # 0.3%阈值

future_price = df["close"].shift(-horizon)
price_change = (future_price - df["close"]) / df["close"]

df["label"] = 0
df.loc[price_change > threshold, "label"] = 1   # 上涨
df.loc[price_change < -threshold, "label"] = 2  # 下跌

# 移除最后几行（没有未来数据）
df = df.dropna(subset=["label"])

# === 特征选择 ===
print("🔄 选择最终特征...")
# 排除不需要的列
exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_1h', 'close_4h', 'label']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# 清理无穷大值和异常值
for col in feature_cols:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(method='ffill').fillna(0)

print(f"✅ 特征工程完成: {df.shape}")
print(f"特征数量: {len(feature_cols)}")
print(f"标签分布: {df['label'].value_counts().to_dict()}")

# 保存结果
df.to_parquet(OUT, compression="zstd")
print(f"💾 特征已保存到: {OUT}")

# 显示特征列表
print(f"\n📋 特征列表:")
for i, col in enumerate(feature_cols, 1):
    print(f"{i:2d}. {col}")

# 显示样本数据
print(f"\n📊 样本数据:")
print(df[['timestamp', 'close'] + feature_cols[:5] + ['label']].head()) 