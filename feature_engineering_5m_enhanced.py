#!/usr/bin/env python3
"""
RexKing – Feature Engineering 5m (Enhanced with Order Flow)

读取已处理的 5m K 线 + 订单流特征, 计算技术指标 + whale 滚动指标,
生成二分类标签(下一根 K 线涨/跌), 保存为 Parquet & CSV.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ta.trend import ADXIndicator, MACD, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import glob
import os

# ---------- 路径配置 ----------
DATA_DIR = Path("/Users/qiutianyu/data/processed")
KL_FILE  = DATA_DIR / "merged_5m_2023_2025.parquet"
W1_FILE  = Path("/Users/qiutianyu/ETHUSDT-w1/w1_2023_2025.csv")
FUND_OI_FILE = DATA_DIR / "funding_oi_1h.parquet"
LABEL_FILE = DATA_DIR / "label_5m_h12.csv"

def load_latest_order_flow_features():
    """加载最新的订单流特征"""
    files = glob.glob("data/mid_features_5m_*.parquet")
    if not files:
        print("⚠️ 未找到订单流特征文件，跳过订单流特征")
        return None
    
    latest_file = max(files, key=os.path.getctime)
    print(f"📥 加载订单流特征: {latest_file}")
    df = pd.read_parquet(latest_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"✅ 订单流特征: {len(df)} 行")
    return df

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

    # 动量指标 (窗口缩短)
    df["ret_1"]  = close.pct_change()
    df["ret_3"]  = close.pct_change(3)
    df["ret_6"]  = close.pct_change(6)
    df["ret_12"] = close.pct_change(12)  # 1小时
    df["ret_24"] = close.pct_change(24)  # 2小时
    df["ret_72"] = close.pct_change(72)  # 6小时
    
    # 波动率指标 (窗口缩短)
    df["volatility_10"] = close.pct_change().rolling(10).std()
    df["volatility_24"] = close.pct_change().rolling(24).std()
    df["volatility_72"] = close.pct_change().rolling(72).std()

    # EMA 斜率 (窗口缩短)
    ema_50 = EMAIndicator(close, window=50).ema_indicator()
    ema_200 = EMAIndicator(close, window=200).ema_indicator()
    df["ema_50_slope"] = ema_50.pct_change(12)  # 1小时斜率
    df["ema_200_slope"] = ema_200.pct_change(48)  # 4小时斜率
    df["ema_50_200_ratio"] = ema_50 / ema_200

    # 成交量指标
    df["vol_ma20"] = vol.rolling(20).mean()
    df["vol_ratio"] = vol / df["vol_ma20"]
    df["vol_ma50"] = vol.rolling(50).mean()
    df["vol_ratio_50"] = vol / df["vol_ma50"]

    return df

# ---------- 订单流特征 ----------
def add_order_flow_features(df: pd.DataFrame, order_flow_df: pd.DataFrame) -> pd.DataFrame:
    """添加订单流特征"""
    if order_flow_df is None:
        print("⚠️ 跳过订单流特征")
        return df
    
    print("📊 添加订单流特征...")
    
    # 确保时间戳格式一致 - 都转换为UTC
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    order_flow_df['timestamp'] = pd.to_datetime(order_flow_df['timestamp'], utc=True)
    
    # 使用merge_asof进行时间对齐
    df = pd.merge_asof(
        df.sort_values('timestamp'),
        order_flow_df.sort_values('timestamp'),
        on='timestamp',
        direction='backward',
        suffixes=('', '_of')
    )
    
    # 选择重要的订单流特征
    order_flow_cols = [
        # 价格动量
        'price_momentum_1m', 'price_momentum_3m', 'price_momentum_5m',
        
        # 成交量特征
        'volume_ratio', 'volume_ma_3m', 'volume_ma_5m',
        
        # 价差特征
        'spread_ratio', 'spread_trend', 'spread_ma', 'spread_std',
        
        # 订单流特征
        'imbalance_trend', 'pressure_trend', 'volume_imbalance_ma',
        'bid_ask_imbalance', 'price_pressure_ma', 'price_pressure_std',
        
        # 流动性特征
        'liquidity_trend', 'fill_prob_trend', 'liquidity_score', 'liquidity_ma',
        'bid_fill_prob', 'ask_fill_prob', 'bid_price_impact', 'ask_price_impact',
        
        # 波动率特征
        'volatility_ma', 'volatility_ratio', 'price_volatility',
        
        # 异常检测
        'price_jump', 'volume_spike', 'spread_widening',
        
        # VWAP特征
        'vwap_deviation', 'vwap_deviation_ma',
        
        # 其他重要特征
        'buy_ratio', 'price_trend', 'trend_deviation',
        'price_impact_imbalance', 'fill_prob_imbalance'
    ]
    
    # 只保留存在的列
    available_cols = [col for col in order_flow_cols if col in order_flow_df.columns]
    print(f"✅ 使用 {len(available_cols)} 个订单流特征")
    
    # 填充NaN
    for col in available_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
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
        
        # OI变化率 (5m)
        df['oi_chg_6h'] = f['oi'].pct_change(72)
        df['oi_chg_24h'] = f['oi'].pct_change(288)
        
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
            
            # 滚动窗口 (5m)
            df["w1_cnt_6h"]   = df["count"].rolling(72, min_periods=1).sum()   # 6小时 (72根5m)
            df["w1_cnt_12h"]  = df["count"].rolling(144, min_periods=1).sum()  # 12小时 (144根5m)
            df["w1_cnt_24h"]  = df["count"].rolling(288, min_periods=1).sum()  # 24小时 (288根5m)
            
            df["w1_val_6h"]   = df["value"].rolling(72, min_periods=1).sum()
            df["w1_val_12h"]  = df["value"].rolling(144, min_periods=1).sum()
            df["w1_val_24h"]  = df["value"].rolling(288, min_periods=1).sum()
            
            # 标准化
            df["w1_cnt_6h_norm"] = df["w1_cnt_6h"] / df["w1_cnt_6h"].rolling(288).mean()
            df["w1_val_6h_norm"] = df["w1_val_6h"] / df["w1_val_6h"].rolling(288).mean()
            
            # Whale direction (5m)
            df["whale_dir"] = np.sign(df["value"])
            df["whale_dir_6h"] = df["whale_dir"].rolling(72, min_periods=1).sum()
            df["whale_dir_12h"] = df["whale_dir"].rolling(144, min_periods=1).sum()
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

    print("📊 加载订单流特征 ...")
    order_flow_df = load_latest_order_flow_features()
    
    print("🔗 集成订单流特征 ...")
    df = add_order_flow_features(df, order_flow_df)

    print("⏰ 添加时间特征 ...")
    df = add_time_features(df)

    print("🐳 整合 whale 指标 ...")
    df = add_whale_features(df)

    print("💰 整合 funding & OI 指标 ...")
    df = add_funding_oi_features(df)

    print("🏷️ 添加标签 ...")
    df = add_label(df)

    # 清理数据
    print("🧹 清理数据 ...")
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    
    # 移除无穷大值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    print(f"✅ 最终数据: {len(df)} 行, {len(df.columns)} 列")
    print(f"📊 标签分布: {df['label'].value_counts().to_dict()}")

    # 保存结果
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/features_5m_enhanced_{timestamp}.parquet"
    df.to_parquet(output_file, index=False)
    print(f"💾 保存到: {output_file}")

    # 保存特征列表
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'label']]
    feature_list = {
        'features': feature_cols,
        'total_features': len(feature_cols),
        'order_flow_features': len([col for col in feature_cols if any(x in col for x in ['momentum', 'volume_', 'spread_', 'imbalance', 'pressure', 'liquidity', 'jump', 'spike'])]),
        'technical_features': len([col for col in feature_cols if any(x in col for x in ['adx', 'rsi', 'macd', 'stoch', 'bb', 'atr', 'ema', 'ret_', 'volatility'])]),
        'whale_features': len([col for col in feature_cols if col.startswith('w1_') or col.startswith('whale_')]),
        'time_features': len([col for col in feature_cols if col in ['hour', 'weekday', 'month', 'is_asia_session', 'is_london_session', 'is_ny_session']]),
        'funding_oi_features': len([col for col in feature_cols if col.startswith('funding_') or col.startswith('oi_')])
    }
    
    feature_list_file = f"data/feature_list_5m_enhanced_{timestamp}.json"
    import json
    with open(feature_list_file, 'w') as f:
        json.dump(feature_list, f, indent=2)
    print(f"📋 特征列表保存到: {feature_list_file}")

if __name__ == "__main__":
    main() 