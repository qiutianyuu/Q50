#!/usr/bin/env python3
"""
RexKing – Enhanced Feature Engineering 15m

整合订单流特征 + On-chain数据 + 技术指标，生成高信息密度的15m特征
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ta.trend import ADXIndicator, MACD, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

# ---------- 路径配置 ----------
DATA_DIR = Path("/Users/qiutianyu/data/processed")
KL_FILE = DATA_DIR / "merged_15m_2023_2025.parquet"
W1_FILE = Path("/Users/qiutianyu/ETHUSDT-w1/w1_2023_2025.csv")
FUND_OI_FILE = DATA_DIR / "funding_oi_1h.parquet"
LABEL_FILE = DATA_DIR / "label_15m_h6.csv"
ORDERFLOW_FILE = Path("data/mid_features_15m_orderflow.parquet")

# ---------- 技术指标 ----------
def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    hi, lo, close, vol = df["high"], df["low"], df["close"], df["volume"]

    # 趋势指标
    df["adx_14"] = ADXIndicator(hi, lo, close, window=14).adx()
    df["rsi_14"] = RSIIndicator(close, window=14).rsi()

    macd = MACD(close)
    df["macd_diff"] = macd.macd_diff()

    stoch = StochasticOscillator(hi, lo, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    bb = BollingerBands(close, window=20, window_dev=2)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / close
    df["bb_percent"] = bb.bollinger_pband()

    atr = AverageTrueRange(hi, lo, close, window=14)
    df["atr_norm"] = atr.average_true_range() / close

    # 动量指标
    df["ret_1"] = close.pct_change()
    df["ret_3"] = close.pct_change(3)
    df["ret_6"] = close.pct_change(6)
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
    df["year"] = df["timestamp"].dt.year
    
    # 时间周期特征
    df["is_asia_session"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(int)
    df["is_london_session"] = ((df["hour"] >= 8) & (df["hour"] < 16)).astype(int)
    df["is_ny_session"] = ((df["hour"] >= 13) & (df["hour"] < 21)).astype(int)
    
    return df

# ---------- 订单流特征 ----------
def add_orderflow_features(df: pd.DataFrame) -> pd.DataFrame:
    """整合订单流特征"""
    if ORDERFLOW_FILE.exists():
        print("📊 加载订单流特征...")
        orderflow = pd.read_parquet(ORDERFLOW_FILE)
        orderflow['timestamp'] = pd.to_datetime(orderflow['timestamp'], utc=True)
        
        # 确保主数据也有UTC时区
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # 选择高信息密度的订单流特征
        orderflow_cols = [
            'liquidity_pressure', 'liquidity_pressure_ma',
            'taker_imbalance', 'taker_imbalance_ma',
            'order_flow_intensity', 'order_flow_intensity_ma',
            'liquidity_impact', 'liquidity_impact_ma',
            'buy_pressure_ratio', 'sell_pressure_ratio',
            'order_flow_strength', 'order_flow_strength_ma',
            'liquidity_stress', 'liquidity_stress_ma',
            'spread_compression', 'volume_imbalance',
            'price_pressure', 'vwap_deviation'
        ]
        
        # 只保留存在的列
        available_cols = [col for col in orderflow_cols if col in orderflow.columns]
        print(f"可用订单流特征: {len(available_cols)}")
        
        # 合并订单流特征
        df = pd.merge_asof(
            df.sort_values('timestamp'),
            orderflow[['timestamp'] + available_cols].sort_values('timestamp'),
            on='timestamp', direction='backward'
        )
        
        # 填充NaN
        df[available_cols] = df[available_cols].fillna(0)
        
        print(f"✅ 订单流特征整合完成，新增 {len(available_cols)} 个特征")
    else:
        print("⚠️ 订单流文件不存在，跳过订单流特征")
        # 初始化默认值
        orderflow_cols = ['liquidity_pressure', 'taker_imbalance', 'order_flow_intensity']
        for col in orderflow_cols:
            df[col] = 0
    
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
        
        # 填充NaN
        df[['funding_z', 'oi_chg_6h', 'oi_chg_24h']] = df[['funding_z', 'oi_chg_6h', 'oi_chg_24h']].fillna(0)
    else:
        # 文件不存在时填0
        df[['funding_z', 'oi_chg_6h', 'oi_chg_24h']] = 0
    return df

# ---------- Whale 信号 (增强版) ----------
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
            
            # 滚动窗口 (新增)
            df["w1_cnt_6h"]   = df["count"].rolling(24, min_periods=1).sum()   # 6小时 (24根15m)
            df["w1_cnt_12h"]  = df["count"].rolling(48, min_periods=1).sum()   # 12小时 (48根15m)
            df["w1_cnt_24h"]  = df["count"].rolling(96, min_periods=1).sum()   # 24小时 (96根15m)
            
            df["w1_val_6h"]   = df["value"].rolling(24, min_periods=1).sum()
            df["w1_val_12h"]  = df["value"].rolling(48, min_periods=1).sum()
            df["w1_val_24h"]  = df["value"].rolling(96, min_periods=1).sum()
            
            # 标准化
            df["w1_cnt_6h_norm"] = df["w1_cnt_6h"] / df["w1_cnt_6h"].rolling(96).mean()
            df["w1_val_6h_norm"] = df["w1_val_6h"] / df["w1_val_6h"].rolling(96).mean()
            
            # Whale direction (新增)
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
    """使用外部label文件"""
    if LABEL_FILE.exists():
        label_df = pd.read_csv(LABEL_FILE)
        label_df['timestamp'] = pd.to_datetime(label_df['timestamp'], utc=True)
        
        # 合并label
        df = df.merge(label_df[['timestamp', 'label']], on='timestamp', how='left')
        
        # 只保留有交易信号的样本（label != -1）
        df = df[df['label'] != -1].reset_index(drop=True)
        
        print(f"✅ 使用外部label，保留交易信号样本: {len(df)}")
    else:
        print("⚠️ 外部label文件不存在，使用默认标签")
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

    print("📊 整合订单流特征 ...")
    df = add_orderflow_features(df)

    print("🐳 整合 whale 指标 ...")
    df = add_whale_features(df)

    print("💰 添加 Funding & OI 特征 ...")
    df = add_funding_oi_features(df)

    print("🏷️ 生成标签 ...")
    df = add_label(df)

    # 丢弃包含 NaN 的行（由指标窗口造成）
    df = df.dropna().reset_index(drop=True)

    # 添加趋势特征（不依赖外部文件）
    df['trend_1h'] = np.where(df['ema_50_slope'] > 0, 1, -1)
    df['trend_4h'] = np.where(df['ema_200_slope'] > 0, 1, -1)

    # ---------- 保存 ----------
    out_parquet = DATA_DIR / "features_15m_enhanced.parquet"
    out_csv = DATA_DIR / "features_15m_enhanced.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print(f"✅ 完成! 特征行数: {len(df)}, 列数: {df.shape[1]}")
    print("前 5 行示例:")
    print(df.head())

if __name__ == "__main__":
    main()
