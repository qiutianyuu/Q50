#!/usr/bin/env python3
"""
事件系统使用示例 - 展示完整的事件检测和标签生成工作流程
"""

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys
import warnings
from datetime import datetime
import argparse

warnings.filterwarnings('ignore')

def run_event_detection(input_file: str, output_file: str, config_file: str = None):
    """运行事件检测"""
    print(f"🔍 运行事件检测...")
    print(f"📁 输入: {input_file}")
    print(f"📁 输出: {output_file}")
    
    cmd = [sys.executable, "detect_events.py", "--input", input_file, "--output", output_file]
    if config_file:
        cmd.extend(["--config", config_file])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ 事件检测完成!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 事件检测失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def run_label_generation(input_file: str, output_file: str, strategy: str = "event_strength", 
                        min_strength: float = 0.3, max_strength: float = 0.8, 
                        min_density: int = 3, hold_period: int = 4):
    """运行标签生成"""
    print(f"🏷️ 运行标签生成...")
    print(f"📁 输入: {input_file}")
    print(f"📁 输出: {output_file}")
    print(f"🎯 策略: {strategy}")
    
    cmd = [
        sys.executable, "label_events.py",
        "--input", input_file,
        "--output", output_file,
        "--strategy", strategy,
        "--min_strength", str(min_strength),
        "--max_strength", str(max_strength),
        "--min_density", str(min_density),
        "--hold_period", str(hold_period),
        "--min_profit", "0.001",
        "--max_loss", "-0.002"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ 标签生成完成!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 标签生成失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def analyze_results(events_file: str, labels_file: str):
    """分析结果"""
    print(f"\n📊 分析结果...")
    
    try:
        # 读取事件数据
        events_df = pd.read_parquet(events_file)
        print(f"📈 事件数据: {len(events_df):,} 样本, {len(events_df.columns)} 特征")
        
        # 读取标签数据
        labels_df = pd.read_parquet(labels_file)
        print(f"🏷️ 标签数据: {len(labels_df):,} 样本")
        
        # 事件统计
        event_features = [col for col in events_df.columns if any(event_type in col for event_type in 
                        ['breakout', 'reversal', 'spike', 'cross', 'oversold', 'overbought', 'whale_'])]
        print(f"🔍 检测到 {len(event_features)} 种事件类型")
        
        # 事件强度分析
        if 'event_strength' in events_df.columns:
            strength_stats = events_df['event_strength'].describe()
            print(f"\n📊 事件强度统计:")
            print(f"  均值: {strength_stats['mean']:.3f}")
            print(f"  标准差: {strength_stats['std']:.3f}")
            print(f"  最小值: {strength_stats['min']:.3f}")
            print(f"  最大值: {strength_stats['max']:.3f}")
        
        # 事件密度分析
        if 'event_density' in events_df.columns:
            density_stats = events_df['event_density'].describe()
            print(f"\n📊 事件密度统计:")
            print(f"  均值: {density_stats['mean']:.1f}")
            print(f"  标准差: {density_stats['std']:.1f}")
            print(f"  最小值: {density_stats['min']:.0f}")
            print(f"  最大值: {density_stats['max']:.0f}")
        
        # 标签分析
        if 'label' in labels_df.columns:
            label_counts = labels_df['label'].value_counts()
            print(f"\n🏷️ 标签分布:")
            for label, count in label_counts.items():
                percentage = count / len(labels_df) * 100
                if label == 1:
                    print(f"  做多信号: {count:,} ({percentage:.1f}%)")
                elif label == 0:
                    print(f"  做空信号: {count:,} ({percentage:.1f}%)")
                else:
                    print(f"  不交易: {count:,} ({percentage:.1f}%)")
        
        # 收益分析
        if 'net_return' in labels_df.columns:
            trade_mask = labels_df['label'] != -1
            if trade_mask.sum() > 0:
                trade_returns = labels_df.loc[trade_mask, 'net_return']
                print(f"\n💰 交易信号收益分析:")
                print(f"  交易信号数: {trade_mask.sum():,}")
                print(f"  平均净收益: {trade_returns.mean():.6f} ({trade_returns.mean()*100:.4f}%)")
                print(f"  净收益标准差: {trade_returns.std():.6f}")
                print(f"  正收益比例: {(trade_returns > 0).sum() / len(trade_returns)*100:.1f}%")
                print(f"  最大收益: {trade_returns.max():.6f}")
                print(f"  最大损失: {trade_returns.min():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 结果分析失败: {e}")
        return False

def run_complete_workflow(input_file: str, timeframe: str = "15m", strategy: str = "event_strength"):
    """运行完整的工作流程"""
    print(f"🚀 RexKing 事件系统完整工作流程")
    print(f"📁 输入文件: {input_file}")
    print(f"⏱️ 时间框架: {timeframe}")
    print(f"🎯 标签策略: {strategy}")
    print("=" * 60)
    
    # 步骤1: 事件检测
    events_file = f"data/events_{timeframe}.parquet"
    if not run_event_detection(input_file, events_file):
        print("❌ 事件检测失败，工作流程终止")
        return False
    
    # 步骤2: 标签生成
    labels_file = f"data/labels_{timeframe}_{strategy}.parquet"
    if not run_label_generation(events_file, labels_file, strategy):
        print("❌ 标签生成失败，工作流程终止")
        return False
    
    # 步骤3: 结果分析
    if not analyze_results(events_file, labels_file):
        print("❌ 结果分析失败")
        return False
    
    print(f"\n🎉 完整工作流程成功完成!")
    print(f"📁 事件文件: {events_file}")
    print(f"📁 标签文件: {labels_file}")
    
    return True

def run_multiple_strategies(input_file: str, timeframe: str = "15m"):
    """运行多种标签策略"""
    print(f"🔄 运行多种标签策略...")
    
    strategies = [
        ("event_strength", 0.3, 0.8, 3),
        ("event_combination", 0.2, 0.9, 2),
        ("event_sequential", 0.25, 0.85, 3)
    ]
    
    results = {}
    
    for strategy, min_strength, max_strength, min_density in strategies:
        print(f"\n🎯 策略: {strategy}")
        
        events_file = f"data/events_{timeframe}.parquet"
        labels_file = f"data/labels_{timeframe}_{strategy}.parquet"
        
        # 生成标签
        if run_label_generation(events_file, labels_file, strategy, min_strength, max_strength, min_density):
            # 分析结果
            try:
                labels_df = pd.read_parquet(labels_file)
                trade_signals = (labels_df['label'] != -1).sum()
                trade_ratio = trade_signals / len(labels_df) * 100
                
                if 'net_return' in labels_df.columns:
                    trade_mask = labels_df['label'] != -1
                    if trade_mask.sum() > 0:
                        trade_returns = labels_df.loc[trade_mask, 'net_return']
                        avg_return = trade_returns.mean()
                        positive_ratio = (trade_returns > 0).sum() / len(trade_returns) * 100
                    else:
                        avg_return = 0
                        positive_ratio = 0
                else:
                    avg_return = 0
                    positive_ratio = 0
                
                results[strategy] = {
                    "trade_signals": trade_signals,
                    "trade_ratio": trade_ratio,
                    "avg_return": avg_return,
                    "positive_ratio": positive_ratio
                }
                
                print(f"  📊 交易信号: {trade_signals:,} ({trade_ratio:.1f}%)")
                print(f"  💰 平均收益: {avg_return:.6f}")
                print(f"  ✅ 正收益比例: {positive_ratio:.1f}%")
                
            except Exception as e:
                print(f"  ❌ 分析失败: {e}")
                results[strategy] = None
    
    # 策略比较
    print(f"\n📊 策略比较:")
    print(f"{'策略':<15} {'信号数':<10} {'信号比例':<10} {'平均收益':<12} {'正收益比例':<12}")
    print("-" * 70)
    
    for strategy, result in results.items():
        if result:
            print(f"{strategy:<15} {result['trade_signals']:<10,} {result['trade_ratio']:<10.1f}% "
                  f"{result['avg_return']:<12.6f} {result['positive_ratio']:<12.1f}%")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='RexKing 事件系统使用示例')
    parser.add_argument('--input', type=str, default='data/features_15m_enhanced.parquet',
                       help='输入特征文件路径')
    parser.add_argument('--timeframe', type=str, default='15m', choices=['5m', '15m', '1h'],
                       help='时间框架')
    parser.add_argument('--strategy', type=str, default='event_strength',
                       choices=['event_strength', 'event_combination', 'event_sequential'],
                       help='标签生成策略')
    parser.add_argument('--multi', action='store_true', help='运行多种策略比较')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.input).exists():
        print(f"❌ 输入文件不存在: {args.input}")
        print("请先运行特征工程脚本生成特征文件")
        return
    
    if args.multi:
        # 运行多种策略
        run_multiple_strategies(args.input, args.timeframe)
    else:
        # 运行单一策略
        run_complete_workflow(args.input, args.timeframe, args.strategy)

if __name__ == "__main__":
    main() 