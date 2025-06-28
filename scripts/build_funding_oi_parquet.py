#!/usr/bin/env python3
"""
把本地 fundingRate 月度 CSV +（可选）Open Interest CSV 合并为
data/funding_oi_1h_2021_2022.parquet
"""
import glob, os, pandas as pd, numpy as np
from pathlib import Path

# ======== 你需要改的两处 ========
FUNDING_ROOT = Path("/Users/qiutianyu/ETHUSDT-fundingRate-2021-2022")
OI_ROOT      = Path("/Users/qiutianyu/ETHUSDT-openInterestHist-2021-2022")  # 如果没有就留空
OUT          = Path("data/funding_oi_1h_2021_2022.parquet")
# =================================

def load_funding():
    csv_list = glob.glob(str(FUNDING_ROOT / "**/*.csv"), recursive=True)
    dfs = []
    for fp in csv_list:
        df = pd.read_csv(fp)
        df = df.rename(columns={
            "calc_time": "timestamp_ms",
            "last_funding_rate": "funding"
        })[["timestamp_ms", "funding"]]
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        dfs.append(df[["timestamp", "funding"]])
    funding = (pd.concat(dfs)
                 .sort_values("timestamp")
                 .drop_duplicates("timestamp")
                 .set_index("timestamp"))
    # 8h → 1h 频率，用前值填充
    funding = funding.resample("1H").ffill()
    return funding

def load_oi():
    if not OI_ROOT.exists():
        print("⚠️ 未找到 OI 目录，OI 全部填 0")
        return pd.DataFrame(columns=["oi"])
    csv_list = glob.glob(str(OI_ROOT / "**/*.csv"), recursive=True)
    dfs = []
    for fp in csv_list:
        df = pd.read_csv(fp)
        if "timestamp" not in df.columns:
            # Binance 官方 CSV 用毫秒时间戳
            df = df.rename(columns={"timestamp": "timestamp_ms"})
        if "timestamp_ms" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        dfs.append(df[["timestamp", "sumOpenInterest"]]
                   .rename(columns={"sumOpenInterest": "oi"}))
    oi = (pd.concat(dfs)
            .sort_values("timestamp")
            .drop_duplicates("timestamp")
            .set_index("timestamp")
            .resample("1H").ffill())
    return oi

def main():
    print("⏬ 读取 funding CSV ...")
    funding = load_funding()
    print(f"Funding 行数: {len(funding)}")

    print("⏬ 读取 OI CSV ...")
    oi = load_oi()
    print(f"OI 行数: {len(oi)}")

    merged = (funding.join(oi, how="outer")
                      .ffill()
                      .reset_index()
                      .loc[:, ["timestamp", "funding", "oi"]])

    # 没有 OI 就填 0
    merged["oi"] = merged["oi"].fillna(0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT, index=False, compression="zstd")
    print(f"✅ 已保存 {OUT}, shape={merged.shape}")

if __name__ == "__main__":
    main() 