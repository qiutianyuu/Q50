#!/usr/bin/env python3
"""
RexKing – Feature Engineering  (PoC on 1h timeframe)

读取已处理的 1h K 线, 计算一揽子技术指标 + whale 滚动指标,
生成二分类标签(下一根 K 线涨/跌), 保存为 Parquet & CSV.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ta.trend import ADXIndicator, MACD, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

# ---------- 路径配置 ----------
DATA_DIR = Path("/Users/qiutianyu/data/processed")
KL_FILE  = DATA_DIR / "merged_1h_2023_2025.parquet"
W1_FILE  = Path("/Users/qiutianyu/ETHUSDT-w1/w1_2023_2025.csv")

# ---------- 技术指标 ----------
def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    hi, lo, close, vol = df["high"], df["low"], df["close"], df["volume"]

    df["adx_14"]  = ADXIndicator(hi, lo, close, window=14).adx()
    df["rsi_14"]  = RSIIndicator(close, window=14).rsi()

    macd = MACD(close)
    df["macd_diff"] = macd.macd_diff()

    stoch = StochasticOscillator(hi, lo, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    bb = BollingerBands(close, window=20, window_dev=2)
    df["bb_width"]   = (bb.bollinger_hband() - bb.bollinger_lband()) / close
    df["bb_percent"] = bb.bollinger_pband()

    atr = AverageTrueRange(hi, lo, close, window=14)
    df["atr_norm"] = atr.average_true_range() / close

    # 简单动量 / 波动
    df["ret_1"]  = close.pct_change()
    df["ret_3"]  = close.pct_change(3)
    df["ret_6"]  = close.pct_change(6)
    df["volatility_10"] = close.pct_change().rolling(10).std()

    # 成交量
    df["vol_ma20"] = vol.rolling(20).mean()
    df["vol_ratio"] = vol / df["vol_ma20"]

    # EMA
    ema_50 = EMAIndicator(close, window=50).ema_indicator()
    ema_200 = EMAIndicator(close, window=200).ema_indicator()
    df["ema_50"] = ema_50
    df["ema_200"] = ema_200
    df["trend_1h"] = (ema_50 > ema_200).astype(int)

    return df

# ---------- Whale 信号 ----------
def add_whale_features(df: pd.DataFrame) -> pd.DataFrame:
    if W1_FILE.exists():
        w1 = pd.read_csv(W1_FILE, parse_dates=["timestamp"])
        w1["timestamp"] = pd.to_datetime(w1["timestamp"], utc=True)
        df = df.merge(
            w1[["timestamp", "count", "value", "w1_zscore"]],
            on="timestamp", how="left"
        )

        df[["count", "value", "w1_zscore"]] = df[["count", "value", "w1_zscore"]].fillna(0)
        df["w1_cnt_24h"]   = df["count"].rolling(24, min_periods=1).sum()
        df["w1_val_24h"]   = df["value"].rolling(24, min_periods=1).sum()
    else:
        # 文件不存在时填 0，脚本仍能跑通
        for col in ["w1_zscore", "w1_cnt_24h", "w1_val_24h"]:
            df[col] = 0
    return df

# ---------- 标签 ----------
def add_label(df: pd.DataFrame, horizon:int = 1) -> pd.DataFrame:
    df["label"] = (df["close"].shift(-horizon) > df["close"]).astype(int)
    return df

# ---------- 主流程 ----------
def main():
    print(f"📥 读取 {KL_FILE.name} ...")
    df = pd.read_parquet(KL_FILE)
    # 修正 dtype 检查，兼容 pandas 扩展类型
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("✨ 计算技术指标 ...")
    df = add_ta_features(df)

    print("🐳 整合 whale 指标 ...")
    df = add_whale_features(df)

    print("🏷️ 生成标签 ...")
    df = add_label(df)

    # 丢弃包含 NaN 的行（由指标窗口造成）
    df = df.dropna().reset_index(drop=True)

    # ---------- 保存 ----------
    out_parquet = DATA_DIR / "features_1h_2023_2025.parquet"
    out_csv     = DATA_DIR / "features_1h_2023_2025.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print(f"✅ 完成! 特征行数: {len(df)}, 列数: {df.shape[1]}")
    print("前 5 行示例:")
    print(df.head())

if __name__ == "__main__":
    main() 