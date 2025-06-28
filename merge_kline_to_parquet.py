#!/usr/bin/env python3
"""
1. 递归读取 15 m/1 h/4 h CSV（月分片）
2. 15 m 作为基准索引；把 1 h、4 h forward-fill 对齐
3. 输出 ETHUSDT_15m_full.parquet（压缩 zstd）
用法:  python3 merge_kline_to_parquet.py
"""
import pathlib
import glob
import pandas as pd
import numpy as np

BASE = pathlib.Path("/Users/qiutianyu")
PATH_15M = BASE / "ETHUSDT-15m"
PATH_1H  = BASE / "ETHUSDT-1h"
PATH_4H  = BASE / "ETHUSDT-4h"
OUTFILE  = BASE / "ETHUSDT_15m_full.parquet"

def concat_monthly(path_pat, tz="UTC"):
    """把 YYYY-MM 文件全部拼起来"""
    files = sorted(glob.glob(str(path_pat)))
    print(f"找到 {len(files)} 个文件")
    dfs = []
    for f in files:
        try:
            # Binance格式: [timestamp, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
            df = pd.read_csv(f, header=None, names=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                'taker_buy_quote', 'ignore'
            ])
            # 只保留需要的列
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(tz)
            dfs.append(df)
        except Exception as e:
            print(f"跳过文件 {f}: {e}")
    return pd.concat(dfs, ignore_index=True)

print("⏳ 读取 15 m …")
df15 = concat_monthly(PATH_15M / "ETHUSDT-15m-*/*.csv")
df15 = df15.sort_values("timestamp").drop_duplicates("timestamp")
print(f"15m数据: {df15.shape}")

print("⏳ 读取 1 h …")
df1h  = concat_monthly(PATH_1H  / "ETHUSDT-1h-*/*.csv")
df1h = df1h[['timestamp', 'close']].rename(columns={"close":"close_1h"})
print(f"1h数据: {df1h.shape}")

print("⏳ 读取 4 h …")
df4h  = concat_monthly(PATH_4H  / "ETHUSDT-4h-*/*.csv")
df4h = df4h[['timestamp', 'close']].rename(columns={"close":"close_4h"})
print(f"4h数据: {df4h.shape}")

# 对齐到 15 m
print("🔄 合并数据...")
df = df15.merge(df1h , on="timestamp", how="left")\
         .merge(df4h , on="timestamp", how="left")

df = df.sort_values("timestamp").reset_index(drop=True)
df = df.fillna(method="ffill")

print("✅ 拼接完成:", df.shape)
print(f"时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
print(f"数据预览:")
print(df.head())

df.to_parquet(OUTFILE, compression="zstd")
print("💾 保存:", OUTFILE) 