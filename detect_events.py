#!/usr/bin/env python3
"""
事件检测系统 - 识别市场中的关键事件
支持多种事件类型：价格突破、成交量异常、技术指标信号、鲸鱼活动等
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from ta.trend import ADXIndicator, MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator

warnings.filterwarnings('ignore')

@dataclass
class EventConfig:
    """事件检测配置"""
    # 价格突破事件
    price_breakout_threshold: float = 0.02  # 2%突破
    price_breakdown_threshold: float = -0.02  # -2%突破
    
    # 成交量异常事件
    volume_spike_threshold: float = 2.0  # 成交量是均值的2倍
    volume_dry_threshold: float = 0.3  # 成交量是均值的30%
    
    # 技术指标事件
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    bb_squeeze_threshold: float = 0.1  # 布林带宽度阈值
    macd_signal_threshold: float = 0.001  # MACD信号阈值
    
    # 鲸鱼事件
    whale_activity_threshold: float = 2.0  # 鲸鱼活动z-score阈值
    whale_volume_threshold: float = 1000000  # 鲸鱼交易量阈值(USD)
    
    # 趋势事件
    trend_strength_threshold: float = 25  # ADX趋势强度阈值
    ema_cross_threshold: float = 0.01  # EMA交叉阈值

class EventDetector:
    """事件检测器"""
    
    def __init__(self, config: EventConfig):
        self.config = config
        self.events = []
    
    def calculate_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算缺失的技术指标"""
        print("🔧 计算缺失的技术指标...")
        
        hi, lo, close, vol = df["high"], df["low"], df["close"], df["volume"]
        
        # 计算RSI
        if 'rsi_14' not in df.columns:
            df["rsi_14"] = RSIIndicator(close, window=14).rsi()
            print("  ✅ 计算RSI指标")
        
        # 计算MACD
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            macd = MACD(close)
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_diff"] = macd.macd_diff()
            print("  ✅ 计算MACD指标")
        
        # 计算布林带
        if 'bb_width' not in df.columns or 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
            bb = BollingerBands(close, window=20, window_dev=2)
            df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / close
            df["bb_upper"] = bb.bollinger_hband()
            df["bb_lower"] = bb.bollinger_lband()
            df["bb_percent"] = bb.bollinger_pband()
            print("  ✅ 计算布林带指标")
        
        # 计算ADX
        if 'adx_14' not in df.columns:
            df["adx_14"] = ADXIndicator(hi, lo, close, window=14).adx()
            print("  ✅ 计算ADX指标")
        
        # 计算EMA
        if 'ema_50' not in df.columns:
            df["ema_50"] = EMAIndicator(close, window=50).ema_indicator()
            print("  ✅ 计算EMA50指标")
        
        if 'ema_200' not in df.columns:
            df["ema_200"] = EMAIndicator(close, window=200).ema_indicator()
            print("  ✅ 计算EMA200指标")
        
        # 计算EMA斜率
        if 'ema_50_slope' not in df.columns and 'ema_50' in df.columns:
            df["ema_50_slope"] = df["ema_50"].pct_change(4)  # 1小时斜率
            print("  ✅ 计算EMA50斜率")
        
        if 'ema_200_slope' not in df.columns and 'ema_200' in df.columns:
            df["ema_200_slope"] = df["ema_200"].pct_change(16)  # 4小时斜率
            print("  ✅ 计算EMA200斜率")
        
        # 计算随机指标
        if 'stoch_k' not in df.columns:
            stoch = StochasticOscillator(hi, lo, close, window=14, smooth_window=3)
            df["stoch_k"] = stoch.stoch()
            df["stoch_d"] = stoch.stoch_signal()
            print("  ✅ 计算随机指标")
        
        # 计算ATR
        if 'atr_norm' not in df.columns:
            atr = AverageTrueRange(hi, lo, close, window=14)
            df["atr_norm"] = atr.average_true_range() / close
            print("  ✅ 计算ATR指标")
        
        return df
    
    def detect_price_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测价格相关事件"""
        print("🔍 检测价格事件...")
        
        # 计算价格变化
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # 价格突破事件
        df['price_breakout'] = (df['price_change'] > self.config.price_breakout_threshold).astype(int)
        df['price_breakdown'] = (df['price_change'] < self.config.price_breakdown_threshold).astype(int)
        
        # 价格反转事件
        df['price_reversal_up'] = (
            (df['price_change'].shift(1) < 0) & 
            (df['price_change'] > 0) & 
            (df['price_change_abs'] > 0.01)
        ).astype(int)
        
        df['price_reversal_down'] = (
            (df['price_change'].shift(1) > 0) & 
            (df['price_change'] < 0) & 
            (df['price_change_abs'] > 0.01)
        ).astype(int)
        
        # 新高新低事件
        df['new_high'] = (df['high'] == df['high'].rolling(20).max()).astype(int)
        df['new_low'] = (df['low'] == df['low'].rolling(20).min()).astype(int)
        
        return df
    
    def detect_volume_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测成交量相关事件"""
        print("📊 检测成交量事件...")
        
        # 成交量移动平均
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 成交量异常事件
        df['volume_spike'] = (df['volume_ratio'] > self.config.volume_spike_threshold).astype(int)
        df['volume_dry'] = (df['volume_ratio'] < self.config.volume_dry_threshold).astype(int)
        
        # 价量背离事件
        df['price_volume_divergence'] = (
            (df['price_change'] > 0) & (df['volume_ratio'] < 0.8)
        ).astype(int)
        
        # 放量突破事件
        df['volume_breakout'] = (
            (df['price_change'] > 0.01) & (df['volume_ratio'] > 1.5)
        ).astype(int)
        
        return df
    
    def detect_technical_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测技术指标事件"""
        print("📈 检测技术指标事件...")
        
        # RSI事件
        if 'rsi_14' in df.columns:
            df['rsi_oversold'] = (df['rsi_14'] < self.config.rsi_oversold).astype(int)
            df['rsi_overbought'] = (df['rsi_14'] > self.config.rsi_overbought).astype(int)
            df['rsi_divergence'] = (
                (df['close'] > df['close'].shift(1)) & 
                (df['rsi_14'] < df['rsi_14'].shift(1))
            ).astype(int)
        else:
            df[['rsi_oversold', 'rsi_overbought', 'rsi_divergence']] = 0
        
        # 布林带事件
        if all(col in df.columns for col in ['bb_width', 'bb_upper', 'bb_lower']):
            df['bb_squeeze'] = (df['bb_width'] < self.config.bb_squeeze_threshold).astype(int)
            df['bb_breakout_up'] = (df['close'] > df['bb_upper']).astype(int)
            df['bb_breakout_down'] = (df['close'] < df['bb_lower']).astype(int)
        else:
            df[['bb_squeeze', 'bb_breakout_up', 'bb_breakout_down']] = 0
        
        # MACD事件
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            df['macd_bullish_cross'] = (
                (df['macd'] > df['macd_signal']) & 
                (df['macd'].shift(1) <= df['macd_signal'].shift(1))
            ).astype(int)
            
            df['macd_bearish_cross'] = (
                (df['macd'] < df['macd_signal']) & 
                (df['macd'].shift(1) >= df['macd_signal'].shift(1))
            ).astype(int)
        else:
            df[['macd_bullish_cross', 'macd_bearish_cross']] = 0
        
        # 随机指标事件
        if 'stoch_k' in df.columns:
            df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
            df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        else:
            df[['stoch_oversold', 'stoch_overbought']] = 0
        
        return df
    
    def detect_trend_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测趋势相关事件"""
        print("📉 检测趋势事件...")
        
        # 趋势强度事件
        if 'adx_14' in df.columns:
            df['trend_strong'] = (df['adx_14'] > self.config.trend_strength_threshold).astype(int)
            df['trend_weak'] = (df['adx_14'] < 20).astype(int)
        else:
            df[['trend_strong', 'trend_weak']] = 0
        
        # EMA交叉事件
        if all(col in df.columns for col in ['ema_50', 'ema_200']):
            df['ema_bullish_cross'] = (
                (df['ema_50'] > df['ema_200']) & 
                (df['ema_50'].shift(1) <= df['ema_200'].shift(1))
            ).astype(int)
            
            df['ema_bearish_cross'] = (
                (df['ema_50'] < df['ema_200']) & 
                (df['ema_50'].shift(1) >= df['ema_200'].shift(1))
            ).astype(int)
        else:
            df[['ema_bullish_cross', 'ema_bearish_cross']] = 0
        
        # 趋势反转事件
        if 'ema_50_slope' in df.columns:
            df['trend_reversal_up'] = (
                (df['ema_50_slope'] > 0) & 
                (df['ema_50_slope'].shift(1) <= 0)
            ).astype(int)
            
            df['trend_reversal_down'] = (
                (df['ema_50_slope'] < 0) & 
                (df['ema_50_slope'].shift(1) >= 0)
            ).astype(int)
        else:
            df[['trend_reversal_up', 'trend_reversal_down']] = 0
        
        return df
    
    def detect_whale_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测鲸鱼相关事件"""
        print("🐳 检测鲸鱼事件...")
        
        # 初始化鲸鱼事件列
        whale_events = [
            'whale_large_inflow', 'whale_large_outflow', 'whale_activity_spike',
            'whale_accumulation', 'whale_distribution', 'whale_momentum'
        ]
        
        for event in whale_events:
            df[event] = 0
        
        # 检查是否有鲸鱼数据
        if 'w1_zscore' in df.columns and 'w1_val_6h' in df.columns:
            # 大额流入流出事件
            df['whale_large_inflow'] = (
                (df['w1_zscore'] > self.config.whale_activity_threshold) & 
                (df['w1_val_6h'] > self.config.whale_volume_threshold)
            ).astype(int)
            
            df['whale_large_outflow'] = (
                (df['w1_zscore'] < -self.config.whale_activity_threshold) & 
                (df['w1_val_6h'] > self.config.whale_volume_threshold)
            ).astype(int)
            
            # 鲸鱼活动异常
            df['whale_activity_spike'] = (df['w1_zscore'].abs() > self.config.whale_activity_threshold).astype(int)
            
            # 鲸鱼积累/分发模式
            df['whale_accumulation'] = (
                (df['whale_dir_6h'] > 0) & 
                (df['w1_cnt_6h'] > df['w1_cnt_6h'].rolling(24).mean())
            ).astype(int)
            
            df['whale_distribution'] = (
                (df['whale_dir_6h'] < 0) & 
                (df['w1_cnt_6h'] > df['w1_cnt_6h'].rolling(24).mean())
            ).astype(int)
            
            # 鲸鱼动量
            df['whale_momentum'] = (
                (df['whale_dir_6h'] > 0) & 
                (df['whale_dir_12h'] > 0)
            ).astype(int)
        
        return df
    
    def detect_market_regime_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测市场状态事件"""
        print("🏛️ 检测市场状态事件...")
        
        # 波动率状态
        if 'volatility_24' in df.columns:
            df['high_volatility'] = (df['volatility_24'] > df['volatility_24'].rolling(48).quantile(0.8)).astype(int)
            df['low_volatility'] = (df['volatility_24'] < df['volatility_24'].rolling(48).quantile(0.2)).astype(int)
        else:
            # 计算简单的波动率
            volatility = df['close'].pct_change().rolling(24).std()
            df['high_volatility'] = (volatility > volatility.rolling(48).quantile(0.8)).astype(int)
            df['low_volatility'] = (volatility < volatility.rolling(48).quantile(0.2)).astype(int)
        
        # 市场状态
        if all(col in df.columns for col in ['ema_50', 'ema_200', 'ema_50_slope']):
            df['bull_market'] = (
                (df['ema_50'] > df['ema_200']) & 
                (df['ema_50_slope'] > 0) & 
                (df['close'] > df['ema_50'])
            ).astype(int)
            
            df['bear_market'] = (
                (df['ema_50'] < df['ema_200']) & 
                (df['ema_50_slope'] < 0) & 
                (df['close'] < df['ema_50'])
            ).astype(int)
        else:
            df[['bull_market', 'bear_market']] = 0
        
        # 横盘整理
        if all(col in df.columns for col in ['adx_14', 'bb_width']):
            df['sideways_market'] = (
                (df['adx_14'] < 20) & 
                (df['bb_width'] < self.config.bb_squeeze_threshold)
            ).astype(int)
        else:
            df['sideways_market'] = 0
        
        return df
    
    def aggregate_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """聚合事件信号"""
        print("🔗 聚合事件信号...")
        
        # 事件类型分类
        bullish_events = [
            'price_breakout', 'price_reversal_up', 'new_high',
            'volume_breakout', 'rsi_oversold', 'bb_breakout_up',
            'macd_bullish_cross', 'stoch_oversold', 'ema_bullish_cross',
            'trend_reversal_up', 'whale_large_inflow', 'whale_accumulation',
            'whale_momentum', 'bull_market'
        ]
        
        bearish_events = [
            'price_breakdown', 'price_reversal_down', 'new_low',
            'volume_dry', 'rsi_overbought', 'bb_breakout_down',
            'macd_bearish_cross', 'stoch_overbought', 'ema_bearish_cross',
            'trend_reversal_down', 'whale_large_outflow', 'whale_distribution',
            'bear_market'
        ]
        
        neutral_events = [
            'volume_spike', 'price_volume_divergence', 'rsi_divergence',
            'bb_squeeze', 'trend_strong', 'trend_weak', 'whale_activity_spike',
            'high_volatility', 'low_volatility', 'sideways_market'
        ]
        
        # 过滤存在的列
        available_bullish = [col for col in bullish_events if col in df.columns]
        available_bearish = [col for col in bearish_events if col in df.columns]
        available_neutral = [col for col in neutral_events if col in df.columns]
        
        # 计算事件强度
        df['bullish_event_count'] = df[available_bullish].sum(axis=1)
        df['bearish_event_count'] = df[available_bearish].sum(axis=1)
        df['neutral_event_count'] = df[available_neutral].sum(axis=1)
        df['total_event_count'] = df['bullish_event_count'] + df['bearish_event_count'] + df['neutral_event_count']
        
        # 事件强度评分
        df['event_strength'] = (
            df['bullish_event_count'] - df['bearish_event_count']
        ) / (df['total_event_count'] + 1)  # 避免除零
        
        # 事件密度
        df['event_density'] = df['total_event_count'].rolling(6).sum()  # 1.5小时窗口
        
        # 事件一致性
        df['event_consistency'] = (
            (df['bullish_event_count'] > df['bearish_event_count']).astype(int) - 
            (df['bearish_event_count'] > df['bullish_event_count']).astype(int)
        )
        
        return df
    
    def detect_all_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测所有类型的事件"""
        print("🚀 开始全面事件检测...")
        
        # 计算缺失的技术指标
        df = self.calculate_missing_indicators(df)
        
        # 检测各类事件
        df = self.detect_price_events(df)
        df = self.detect_volume_events(df)
        df = self.detect_technical_events(df)
        df = self.detect_trend_events(df)
        df = self.detect_whale_events(df)
        df = self.detect_market_regime_events(df)
        
        # 聚合事件信号
        df = self.aggregate_events(df)
        
        # 统计事件分布
        event_columns = [col for col in df.columns if any(event_type in col for event_type in 
                        ['breakout', 'reversal', 'spike', 'cross', 'oversold', 'overbought', 'whale_'])]
        
        print(f"\n📊 事件检测统计:")
        print(f"检测到的事件类型: {len(event_columns)}")
        print(f"总样本数: {len(df):,}")
        
        for event_col in event_columns:
            event_count = df[event_col].sum()
            if event_count > 0:
                print(f"  {event_col}: {event_count:,} ({event_count/len(df)*100:.1f}%)")
        
        print(f"\n事件强度统计:")
        print(f"平均事件强度: {df['event_strength'].mean():.3f}")
        print(f"事件强度标准差: {df['event_strength'].std():.3f}")
        print(f"高事件密度样本: {(df['event_density'] > 5).sum():,}")
        
        return df

def main():
    parser = argparse.ArgumentParser(description='事件检测系统')
    parser.add_argument('--input', type=str, required=True, help='输入特征文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出事件文件路径')
    parser.add_argument('--config', type=str, help='配置文件路径(可选)')
    
    args = parser.parse_args()
    
    print("🔍 RexKing 事件检测系统")
    print(f"📁 输入文件: {args.input}")
    print(f"📁 输出文件: {args.output}")
    
    # 读取数据
    print(f"\n📥 读取数据...")
    if args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    
    # 初始化事件检测器
    config = EventConfig()
    detector = EventDetector(config)
    
    # 检测事件
    df_with_events = detector.detect_all_events(df)
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.output.endswith('.parquet'):
        df_with_events.to_parquet(args.output, index=False)
    else:
        df_with_events.to_csv(args.output, index=False)
    
    print(f"\n✅ 事件检测完成!")
    print(f"📁 结果已保存: {args.output}")
    print(f"📊 新增事件特征: {len([col for col in df_with_events.columns if col not in df.columns])}")

if __name__ == "__main__":
    main() 