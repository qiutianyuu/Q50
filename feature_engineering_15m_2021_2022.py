#!/usr/bin/env python3
"""
RexKing – Feature Engineering 15m for 2021-2022 Data

读取已处理的 2021-2022 15m K 线, 计算一揽子技术指标 + whale 滚动指标,
生成二分类标签(下一根 K 线涨/跌), 保存为 Parquet & CSV.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ta.trend import ADXIndicator, MACD, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

# ---------- 路径配置 ----------
DATA_DIR = Path("data")  # 使用相对路径
KL_FILE  = DATA_DIR / "merged_15m_2021_2022.parquet"
W1_FILE  = Path("/Users/qiutianyu/ETHUSDT-w1/w1_2021_2022.csv")  # 需要确认路径
FUND_OI_FILE = DATA_DIR / "funding_oi_1h_2021_2022.parquet"  # 需要确认路径
OUTPUT_FILE = DATA_DIR / "features_15m_2021_2022.parquet"

# ---------- 技术指标 ----------
def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    hi, lo, close, vol = df["high"], df["low"], df["close"], df["volume"]

    # 趋势指标
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

    # 动量指标
    df["ret_1"]  = close.pct_change()
    df["ret_3"]  = close.pct_change(3)
    df["ret_6"]  = close.pct_change(6)
    df["ret_12"] = close.pct_change(12)  # 3小时
    df["ret_24"] = close.pct_change(24)  # 6小时
    df["ret_48"] = close.pct_change(48)  # 12小时
    
    # 波动率指标
    df["volatility_10"] = close.pct_change().rolling(10).std()
    df["volatility_24"] = close.pct_change().rolling(24).std()
    df["volatility_48"] = close.pct_change().rolling(48).std()

    # EMA 斜率
    ema_50 = EMAIndicator(close, window=50).ema_indicator()
    ema_200 = EMAIndicator(close, window=200).ema_indicator()
    df["ema_50_slope"] = ema_50.pct_change(4)  # 1小时斜率
    df["ema_200_slope"] = ema_200.pct_change(16)  # 4小时斜率
    df["ema_50_200_ratio"] = ema_50 / ema_200

    # 成交量指标
    df["vol_ma20"] = vol.rolling(20).mean()
    df["vol_ratio"] = vol / df["vol_ma20"]
    df["vol_ma50"] = vol.rolling(50).mean()
    df["vol_ratio_50"] = vol / df["vol_ma50"]

    return df

# ---------- 时间特征 ----------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    
    # 时间周期特征
    df["is_asia_session"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(int)
    df["is_london_session"] = ((df["hour"] >= 8) & (df["hour"] < 16)).astype(int)
    df["is_ny_session"] = ((df["hour"] >= 13) & (df["hour"] < 21)).astype(int)
    
    return df

# ---------- Funding & OI 特征 ----------
def add_funding_oi_features(df: pd.DataFrame) -> pd.DataFrame:
    if FUND_OI_FILE.exists():
        f = pd.read_parquet(FUND_OI_FILE)
        f['timestamp'] = pd.to_datetime(f['timestamp'], utc=True)
        
        # 确保主数据也有UTC时区
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Funding z-score（滚动7天）
        f['funding_z'] = (
            (f['funding'] - f['funding'].rolling(168).mean())
            / f['funding'].rolling(168).std()
        )
        
        # 按小时先forward-fill再merge_asof
        df = pd.merge_asof(
            df.sort_values('timestamp'),
            f.sort_values('timestamp'),
            on='timestamp', direction='backward'
        )
        
        # OI变化率
        df['oi_chg_6h'] = f['oi'].pct_change(6)
        df['oi_chg_24h'] = f['oi'].pct_change(24)
        
        # 添加模型需要的原始特征
        df['funding'] = f['funding']
        df['oi'] = f['oi']
        
        # 填充NaN
        df[['funding_z', 'oi_chg_6h', 'oi_chg_24h', 'funding', 'oi']] = df[['funding_z', 'oi_chg_6h', 'oi_chg_24h', 'funding', 'oi']].fillna(0)
    else:
        # 文件不存在时填0
        df[['funding_z', 'oi_chg_6h', 'oi_chg_24h', 'funding', 'oi']] = 0
    return df

# ---------- Whale 信号 ----------
def add_whale_features(df: pd.DataFrame) -> pd.DataFrame:
    # 初始化所有whale特征为0
    whale_cols = ["w1_zscore", "w1_cnt_6h", "w1_cnt_12h", "w1_cnt_24h", 
                  "w1_val_6h", "w1_val_12h", "w1_val_24h",
                  "w1_cnt_6h_norm", "w1_val_6h_norm", "whale_dir", 
                  "whale_dir_6h", "whale_dir_12h"]
    for col in whale_cols:
        df[col] = 0
    
    if W1_FILE.exists():
        w1 = pd.read_csv(W1_FILE, parse_dates=["timestamp"])
        w1["timestamp"] = pd.to_datetime(w1["timestamp"], utc=True)
        
        # 检查whale数据是否足够
        if len(w1) > 10:  # 至少需要10个数据点
            df = df.merge(
                w1[["timestamp", "count", "value", "w1_zscore"]],
                on="timestamp", how="left"
            )

            df[["count", "value", "w1_zscore"]] = df[["count", "value", "w1_zscore"]].fillna(0)
            
            # 滚动窗口
            df["w1_cnt_6h"]   = df["count"].rolling(24, min_periods=1).sum()   # 6小时 (24根15m)
            df["w1_cnt_12h"]  = df["count"].rolling(48, min_periods=1).sum()   # 12小时 (48根15m)
            df["w1_cnt_24h"]  = df["count"].rolling(96, min_periods=1).sum()   # 24小时 (96根15m)
            
            df["w1_val_6h"]   = df["value"].rolling(24, min_periods=1).sum()
            df["w1_val_12h"]  = df["value"].rolling(48, min_periods=1).sum()
            df["w1_val_24h"]  = df["value"].rolling(96, min_periods=1).sum()
            
            # 标准化
            df["w1_cnt_6h_norm"] = df["w1_cnt_6h"] / df["w1_cnt_6h"].rolling(96).mean()
            df["w1_val_6h_norm"] = df["w1_val_6h"] / df["w1_val_6h"].rolling(96).mean()
            
            # Whale direction
            df["whale_dir"] = np.sign(df["value"])  # 1=流入, -1=流出, 0=无变化
            df["whale_dir_6h"] = df["whale_dir"].rolling(24, min_periods=1).sum()   # 6小时净方向
            df["whale_dir_12h"] = df["whale_dir"].rolling(48, min_periods=1).sum()  # 12小时净方向
        else:
            print(f"⚠️ Whale数据不足 ({len(w1)}行)，使用默认值")
    else:
        print("⚠️ Whale文件不存在，使用默认值")
    
    return df

# ---------- 标签 ----------
def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """使用简单的未来价格标签"""
    df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)
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

    print("⏰ 添加时间特征 ...")
    df = add_time_features(df)

    print("💰 添加Funding & OI特征 ...")
    df = add_funding_oi_features(df)

    print("🐋 添加Whale特征 ...")
    df = add_whale_features(df)

    print("🏷️ 生成标签 ...")
    df = add_label(df)

    # 删除包含NaN的行
    print("🧹 清理数据 ...")
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"删除NaN后: {len(df)} 行 (删除了 {initial_len - len(df)} 行)")

    # 保存结果
    print(f"💾 保存到 {OUTPUT_FILE} ...")
    df.to_parquet(OUTPUT_FILE, compression='zstd', index=False)
    df.to_csv(OUTPUT_FILE.with_suffix('.csv'), index=False)

    print("✅ 特征工程完成!")
    print(f"📊 最终数据: {df.shape}")
    print(f"📅 时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"🏷️ 标签分布: {df['label'].value_counts().to_dict()}")

if __name__ == "__main__":
    main() 