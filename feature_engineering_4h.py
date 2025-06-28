#!/usr/bin/env python3
"""
RexKing – Feature Engineering 4h (Trend Filter)

读取4小时K线数据，计算EMA50/200，生成trend_4h信号
用于4小时趋势过滤
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ta.trend import EMAIndicator

# ---------- 路径配置 ----------
DATA_DIR = Path("/Users/qiutianyu/data/processed")
KL_FILE  = DATA_DIR / "merged_4h_2023_2025.parquet"

def add_4h_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算4小时趋势特征"""
    close = df["close"]
    
    # EMA指标
    ema_50 = EMAIndicator(close, window=50).ema_indicator()
    ema_200 = EMAIndicator(close, window=200).ema_indicator()
    
    df["ema_50_4h"] = ema_50
    df["ema_200_4h"] = ema_200
    df["trend_4h"] = (ema_50 > ema_200).astype(int)
    
    return df

def main():
    print(f"📥 读取 {KL_FILE.name} ...")
    df = pd.read_parquet(KL_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("✨ 计算4小时趋势特征 ...")
    df = add_4h_trend_features(df)

    # 丢弃包含 NaN 的行（由指标窗口造成）
    df = df.dropna().reset_index(drop=True)

    # ---------- 保存 ----------
    out_parquet = DATA_DIR / "features_4h_2023_2025.parquet"
    df.to_parquet(out_parquet, index=False)

    print(f"✅ 完成! 特征行数: {len(df)}, 列数: {df.shape[1]}")
    print("前 5 行示例:")
    print(df[['timestamp', 'close', 'ema_50_4h', 'ema_200_4h', 'trend_4h']].head())

if __name__ == "__main__":
    main() 