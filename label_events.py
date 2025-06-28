#!/usr/bin/env python3
"""
事件标签生成系统 - 基于检测到的事件生成交易信号标签
支持多种标签策略：事件强度、事件组合、事件时序等
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import json

warnings.filterwarnings('ignore')

@dataclass
class LabelConfig:
    """标签生成配置"""
    # 事件强度阈值
    min_event_strength: float = 0.3  # 最小事件强度
    max_event_strength: float = 0.8  # 最大事件强度
    
    # 事件密度阈值
    min_event_density: int = 3  # 最小事件密度
    max_event_density: int = 10  # 最大事件密度
    
    # 事件一致性阈值
    min_event_consistency: float = 0.5  # 最小事件一致性
    
    # 成本感知参数
    taker_fee: float = 0.0004  # 吃单手续费
    maker_fee: float = 0.0002  # 挂单手续费
    slippage: float = 0.00025  # 预估滑点
    funding_fee: float = 0.0001  # 资金费率
    
    # 收益阈值
    min_profit_threshold: float = 0.001  # 最小收益阈值
    max_loss_threshold: float = -0.002  # 最大损失阈值
    
    # 持仓周期
    hold_period: int = 4  # 持仓K线数
    
    # 标签策略
    label_strategy: str = "event_strength"  # 标签策略类型

class EventLabeler:
    """事件标签生成器"""
    
    def __init__(self, config: LabelConfig):
        self.config = config
        self.scaler = StandardScaler()
    
    def calculate_future_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算未来收益"""
        print("💰 计算未来收益...")
        
        # 计算未来价格变化
        df['future_return'] = df['close'].pct_change(self.config.hold_period).shift(-self.config.hold_period)
        
        # 计算总成本
        total_cost = self.config.taker_fee * 2 + self.config.slippage + self.config.funding_fee
        
        # 计算净收益
        df['net_return'] = df['future_return'] - total_cost
        
        print(f"总成本: {total_cost:.4f} ({total_cost*100:.3f}%)")
        print(f"平均净收益: {df['net_return'].mean():.6f}")
        print(f"净收益标准差: {df['net_return'].std():.6f}")
        
        return df
    
    def generate_strength_based_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于事件强度的标签生成"""
        print("🏷️ 生成基于事件强度的标签...")
        
        # 初始化标签
        df['label'] = -1  # 默认不交易
        
        # 基于事件强度生成标签
        bullish_mask = (
            (df['event_strength'] >= self.config.min_event_strength) &
            (df['event_strength'] <= self.config.max_event_strength) &
            (df['event_density'] >= self.config.min_event_density) &
            (df['event_consistency'] >= self.config.min_event_consistency)
        )
        
        bearish_mask = (
            (df['event_strength'] <= -self.config.min_event_strength) &
            (df['event_strength'] >= -self.config.max_event_strength) &
            (df['event_density'] >= self.config.min_event_density) &
            (df['event_consistency'] <= -self.config.min_event_consistency)
        )
        
        # 分配标签
        df.loc[bullish_mask, 'label'] = 1  # 做多
        df.loc[bearish_mask, 'label'] = 0  # 做空
        
        return df
    
    def generate_combination_based_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于事件组合的标签生成"""
        print("🏷️ 生成基于事件组合的标签...")
        
        # 初始化标签
        df['label'] = -1
        
        # 定义关键事件组合
        bullish_combinations = [
            # 价格突破 + 成交量确认
            (df['price_breakout'] == 1) & (df['volume_breakout'] == 1),
            # RSI超卖 + 价格反转
            (df['rsi_oversold'] == 1) & (df['price_reversal_up'] == 1),
            # MACD金叉 + 趋势确认
            (df['macd_bullish_cross'] == 1) & (df['trend_strong'] == 1),
            # 鲸鱼流入 + 价格突破
            (df['whale_large_inflow'] == 1) & (df['price_breakout'] == 1),
            # 布林带突破 + 成交量确认
            (df['bb_breakout_up'] == 1) & (df['volume_ratio'] > 1.2)
        ]
        
        bearish_combinations = [
            # 价格跌破 + 成交量确认
            (df['price_breakdown'] == 1) & (df['volume_dry'] == 1),
            # RSI超买 + 价格反转
            (df['rsi_overbought'] == 1) & (df['price_reversal_down'] == 1),
            # MACD死叉 + 趋势确认
            (df['macd_bearish_cross'] == 1) & (df['trend_strong'] == 1),
            # 鲸鱼流出 + 价格跌破
            (df['whale_large_outflow'] == 1) & (df['price_breakdown'] == 1),
            # 布林带跌破 + 成交量确认
            (df['bb_breakout_down'] == 1) & (df['volume_ratio'] > 1.2)
        ]
        
        # 生成做多信号
        bullish_signal = np.any(bullish_combinations, axis=0)
        df.loc[bullish_signal, 'label'] = 1
        
        # 生成做空信号
        bearish_signal = np.any(bearish_combinations, axis=0)
        df.loc[bearish_signal, 'label'] = 0
        
        return df
    
    def generate_sequential_based_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于事件时序的标签生成"""
        print("🏷️ 生成基于事件时序的标签...")
        
        # 初始化标签
        df['label'] = -1
        
        # 检测事件序列模式
        for i in range(2, len(df)):
            # 检查前几个时间点的事件模式
            recent_events = df.iloc[i-2:i+1]
            
            # 做多序列模式
            bullish_sequence = (
                (recent_events['event_strength'].iloc[0] < 0) &  # 前2个时间点事件强度为负
                (recent_events['event_strength'].iloc[1] > 0) &  # 前1个时间点事件强度为正
                (recent_events['event_strength'].iloc[2] > 0.3) &  # 当前时间点事件强度较高
                (recent_events['event_density'].iloc[2] >= 3)  # 当前事件密度较高
            )
            
            # 做空序列模式
            bearish_sequence = (
                (recent_events['event_strength'].iloc[0] > 0) &  # 前2个时间点事件强度为正
                (recent_events['event_strength'].iloc[1] < 0) &  # 前1个时间点事件强度为负
                (recent_events['event_strength'].iloc[2] < -0.3) &  # 当前时间点事件强度较低
                (recent_events['event_density'].iloc[2] >= 3)  # 当前事件密度较高
            )
            
            if bullish_sequence:
                df.iloc[i, df.columns.get_loc('label')] = 1
            elif bearish_sequence:
                df.iloc[i, df.columns.get_loc('label')] = 0
        
        return df
    
    def generate_ml_based_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于机器学习的标签生成"""
        print("🏷️ 生成基于机器学习的标签...")
        
        # 初始化标签
        df['label'] = -1
        
        # 选择特征
        feature_columns = [
            'event_strength', 'event_density', 'event_consistency',
            'bullish_event_count', 'bearish_event_count', 'neutral_event_count',
            'price_change', 'volume_ratio', 'rsi_14', 'bb_width',
            'macd_diff', 'stoch_k', 'adx_14', 'ema_50_slope'
        ]
        
        # 过滤有效的特征列
        available_features = [col for col in feature_columns if col in df.columns]
        
        if len(available_features) < 5:
            print("⚠️ 可用特征不足，使用默认策略")
            return self.generate_strength_based_labels(df)
        
        # 计算特征矩阵
        X = df[available_features].fillna(0)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 简单的规则组合（可以替换为训练好的模型）
        # 基于事件强度和密度的加权评分
        event_score = (
            X_scaled[:, available_features.index('event_strength')] * 0.4 +
            X_scaled[:, available_features.index('event_density')] * 0.3 +
            X_scaled[:, available_features.index('event_consistency')] * 0.3
        )
        
        # 生成标签
        df.loc[event_score > 0.5, 'label'] = 1  # 做多
        df.loc[event_score < -0.5, 'label'] = 0  # 做空
        
        return df
    
    def apply_cost_aware_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用成本感知过滤"""
        print("💰 应用成本感知过滤...")
        
        # 只保留有交易信号的样本
        trade_mask = df['label'] != -1
        trade_df = df[trade_mask].copy()
        
        if len(trade_df) == 0:
            print("⚠️ 没有检测到交易信号")
            return df
        
        # 计算预期收益
        trade_df['expected_return'] = trade_df['net_return']
        
        # 应用收益阈值过滤
        profitable_long = (trade_df['label'] == 1) & (trade_df['expected_return'] >= self.config.min_profit_threshold)
        profitable_short = (trade_df['label'] == 0) & (trade_df['expected_return'] >= self.config.min_profit_threshold)
        
        # 应用损失阈值过滤
        acceptable_loss_long = (trade_df['label'] == 1) & (trade_df['expected_return'] >= self.config.max_loss_threshold)
        acceptable_loss_short = (trade_df['label'] == 0) & (trade_df['expected_return'] >= self.config.max_loss_threshold)
        
        # 更新标签
        valid_trades = profitable_long | profitable_short | acceptable_loss_long | acceptable_loss_short
        trade_df.loc[~valid_trades, 'label'] = -1
        
        # 更新原始数据框
        df.loc[trade_df.index, 'label'] = trade_df['label']
        
        print(f"成本过滤后保留交易信号: {valid_trades.sum()}")
        
        return df
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成标签的主函数"""
        print("🚀 开始生成事件标签...")
        
        # 计算未来收益
        df = self.calculate_future_returns(df)
        
        # 根据策略生成标签
        if self.config.label_strategy == "event_strength":
            df = self.generate_strength_based_labels(df)
        elif self.config.label_strategy == "event_combination":
            df = self.generate_combination_based_labels(df)
        elif self.config.label_strategy == "event_sequential":
            df = self.generate_sequential_based_labels(df)
        elif self.config.label_strategy == "ml_based":
            df = self.generate_ml_based_labels(df)
        else:
            print(f"⚠️ 未知标签策略: {self.config.label_strategy}，使用默认策略")
            df = self.generate_strength_based_labels(df)
        
        # 应用成本感知过滤
        df = self.apply_cost_aware_filtering(df)
        
        # 统计标签分布
        total_samples = len(df)
        long_signals = (df['label'] == 1).sum()
        short_signals = (df['label'] == 0).sum()
        no_trade = (df['label'] == -1).sum()
        
        print(f"\n📊 标签分布统计:")
        print(f"总样本: {total_samples:,}")
        print(f"做多信号: {long_signals:,} ({long_signals/total_samples*100:.1f}%)")
        print(f"做空信号: {short_signals:,} ({short_signals/total_samples*100:.1f}%)")
        print(f"不交易: {no_trade:,} ({no_trade/total_samples*100:.1f}%)")
        print(f"交易信号占比: {(long_signals+short_signals)/total_samples*100:.1f}%")
        
        # 分析交易信号的收益
        trade_mask = df['label'] != -1
        if trade_mask.sum() > 0:
            trade_returns = df.loc[trade_mask, 'net_return']
            print(f"\n💰 交易信号收益分析:")
            print(f"平均净收益: {trade_returns.mean():.6f} ({trade_returns.mean()*100:.4f}%)")
            print(f"净收益标准差: {trade_returns.std():.6f}")
            print(f"正收益占比: {(trade_returns > 0).sum() / len(trade_returns)*100:.1f}%")
            print(f"净收益分位数:")
            print(f"  25%: {trade_returns.quantile(0.25):.6f}")
            print(f"  50%: {trade_returns.quantile(0.50):.6f}")
            print(f"  75%: {trade_returns.quantile(0.75):.6f}")
        
        return df

def main():
    parser = argparse.ArgumentParser(description='事件标签生成系统')
    parser.add_argument('--input', type=str, required=True, help='输入事件文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出标签文件路径')
    parser.add_argument('--strategy', type=str, default='event_strength', 
                       choices=['event_strength', 'event_combination', 'event_sequential', 'ml_based'],
                       help='标签生成策略')
    parser.add_argument('--min_strength', type=float, default=0.3, help='最小事件强度')
    parser.add_argument('--max_strength', type=float, default=0.8, help='最大事件强度')
    parser.add_argument('--min_density', type=int, default=3, help='最小事件密度')
    parser.add_argument('--hold_period', type=int, default=4, help='持仓周期(K线数)')
    parser.add_argument('--min_profit', type=float, default=0.001, help='最小收益阈值')
    parser.add_argument('--max_loss', type=float, default=-0.002, help='最大损失阈值')
    
    args = parser.parse_args()
    
    print("🏷️ RexKing 事件标签生成系统")
    print(f"📁 输入文件: {args.input}")
    print(f"📁 输出文件: {args.output}")
    print(f"🎯 标签策略: {args.strategy}")
    
    # 读取数据
    print(f"\n📥 读取数据...")
    if args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    
    # 检查必要的事件特征
    required_features = ['event_strength', 'event_density', 'event_consistency']
    missing_features = [f for f in required_features if f not in df.columns]
    
    if missing_features:
        print(f"⚠️ 缺少必要的事件特征: {missing_features}")
        print("请先运行事件检测脚本 (detect_events.py)")
        return
    
    # 初始化标签生成器
    config = LabelConfig(
        label_strategy=args.strategy,
        min_event_strength=args.min_strength,
        max_event_strength=args.max_strength,
        min_event_density=args.min_density,
        hold_period=args.hold_period,
        min_profit_threshold=args.min_profit,
        max_loss_threshold=args.max_loss
    )
    
    labeler = EventLabeler(config)
    
    # 生成标签
    df_with_labels = labeler.generate_labels(df)
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 选择要保存的列
    save_columns = ['timestamp', 'close', 'event_strength', 'event_density', 
                   'event_consistency', 'future_return', 'net_return', 'label']
    
    # 过滤存在的列
    available_columns = [col for col in save_columns if col in df_with_labels.columns]
    output_df = df_with_labels[available_columns].copy()
    
    if args.output.endswith('.parquet'):
        output_df.to_parquet(args.output, index=False)
    else:
        output_df.to_csv(args.output, index=False)
    
    print(f"\n✅ 事件标签生成完成!")
    print(f"📁 结果已保存: {args.output}")
    print(f"📊 有效标签数: {(output_df['label'] != -1).sum():,}")

if __name__ == "__main__":
    main() 