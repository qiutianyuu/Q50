#!/usr/bin/env python3
"""
标签重新设计 - 将三分类转换为二分类
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def load_klines(file_path):
    """加载K线数据"""
    print(f"📁 加载K线数据: {file_path}")
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    return df

def calculate_forward_returns(df, horizon_hours, cost_pct=0.001):
    """计算未来收益率（考虑交易成本）"""
    print(f"💰 计算{horizon_hours}小时未来收益率，交易成本: {cost_pct*100}%")
    
    # 计算未来价格
    df['future_price'] = df['close'].shift(-horizon_hours)
    
    # 计算收益率
    df['returns'] = (df['future_price'] - df['close']) / df['close']
    
    # 考虑交易成本
    df['net_returns'] = df['returns'] - cost_pct
    
    return df

def generate_binary_labels(df, threshold_pct=0.002):
    """生成二分类标签"""
    print(f"🎯 生成二分类标签，阈值: {threshold_pct*100}%")
    
    # 创建标签
    df['label'] = 0  # 默认无信号
    
    # 多头信号：净收益超过阈值
    long_mask = df['net_returns'] > threshold_pct
    df.loc[long_mask, 'label'] = 1
    
    # 空头信号：净收益低于负阈值
    short_mask = df['net_returns'] < -threshold_pct
    df.loc[short_mask, 'label'] = -1
    
    return df

def analyze_labels(df):
    """分析标签分布"""
    print(f"\n📊 标签分布分析:")
    print(f"总样本数: {len(df)}")
    
    label_counts = df['label'].value_counts().sort_index()
    label_pcts = df['label'].value_counts(normalize=True).sort_index() * 100
    
    for label, count in label_counts.items():
        pct = label_pcts[label]
        if label == 1:
            print(f"多头信号: {count} ({pct:.2f}%)")
        elif label == -1:
            print(f"空头信号: {count} ({pct:.2f}%)")
        else:
            print(f"无信号: {count} ({pct:.2f}%)")
    
    # 分析收益率分布
    print(f"\n💰 收益率分析:")
    print(f"平均收益率: {df['returns'].mean():.4f}")
    print(f"收益率标准差: {df['returns'].std():.4f}")
    print(f"正收益比例: {(df['returns'] > 0).mean():.2%}")
    
    # 按标签分析收益率
    print(f"\n📈 按标签的收益率分析:")
    for label in [-1, 0, 1]:
        mask = df['label'] == label
        if mask.sum() > 0:
            avg_return = df.loc[mask, 'returns'].mean()
            if label == 1:
                print(f"多头信号平均收益: {avg_return:.4f}")
            elif label == -1:
                print(f"空头信号平均收益: {avg_return:.4f}")
            else:
                print(f"无信号平均收益: {avg_return:.4f}")

def save_labels(df, output_path):
    """保存标签"""
    # 只保存必要的列
    result_df = df[['timestamp', 'label', 'returns', 'net_returns']].copy()
    
    # 移除NaN值
    result_df = result_df.dropna()
    
    print(f"💾 保存标签到: {output_path}")
    print(f"最终样本数: {len(result_df)}")
    
    result_df.to_csv(output_path, index=False)
    return result_df

def main():
    parser = argparse.ArgumentParser(description='标签重新设计 - 二分类')
    parser.add_argument('--input', type=str, required=True, help='输入K线文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出标签文件路径')
    parser.add_argument('--horizon', type=int, default=12, help='预测时间窗口（小时）')
    parser.add_argument('--threshold', type=float, default=0.002, help='信号阈值（小数）')
    parser.add_argument('--cost', type=float, default=0.001, help='交易成本（小数）')
    
    args = parser.parse_args()
    
    print("🚀 标签重新设计 - 二分类")
    print(f"📁 输入文件: {args.input}")
    print(f"📁 输出文件: {args.output}")
    print(f"⏱️ 预测窗口: {args.horizon}小时")
    print(f"🎯 信号阈值: {args.threshold*100}%")
    print(f"💰 交易成本: {args.cost*100}%")
    
    # 加载数据
    df = load_klines(args.input)
    
    # 计算未来收益率
    df = calculate_forward_returns(df, args.horizon, args.cost)
    
    # 生成标签
    df = generate_binary_labels(df, args.threshold)
    
    # 分析标签
    analyze_labels(df)
    
    # 保存标签
    save_labels(df, args.output)
    
    print("✅ 标签生成完成！")

if __name__ == "__main__":
    main() 