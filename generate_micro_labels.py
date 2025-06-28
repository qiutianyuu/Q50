#!/usr/bin/env python3
"""
生成微观标签脚本
使用最佳参数组合生成标签并保存到特征文件
"""

import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
import argparse
from utils.labeling import make_labels, get_label_stats

def load_latest_features():
    """加载最新的特征文件"""
    files = glob.glob("data/realtime_features_*.parquet")
    if not files:
        raise FileNotFoundError("No realtime features files found")
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading: {latest_file}")
    df = pd.read_parquet(latest_file)
    print(f"Loaded {len(df)} rows")
    return df, latest_file

def load_features_from_path(features_path):
    """从指定路径加载特征文件"""
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    print(f"Loading: {features_path}")
    df = pd.read_parquet(features_path)
    print(f"Loaded {len(df)} rows")
    return df

def generate_labels_for_training(df, horizon=120, alpha=0.3, mode='maker', require_fill=True):
    """为训练生成标签"""
    # 生成标签
    labels = make_labels(df['mid_price'], df['rel_spread'], horizon, alpha, mode=mode, require_fill=require_fill)
    stats = get_label_stats(labels)
    
    # 标签列名
    label_col = f'label_h{horizon}_a{alpha}_{mode}'
    df[label_col] = labels
    
    print(f"\n{label_col}:")
    print(f"  Long: {stats['long_pct']:.1f}% ({stats['long_count']})")
    print(f"  Short: {stats['short_pct']:.1f}% ({stats['short_count']})")
    print(f"  Neutral: {stats['neutral_pct']:.1f}% ({stats['neutral_count']})")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Generate Micro Labels')
    parser.add_argument('--input', type=str, help='Input features parquet file path')
    parser.add_argument('--output', type=str, help='Output labeled features parquet file path')
    parser.add_argument('--horizon', type=int, default=120, help='Label horizon in steps')
    parser.add_argument('--alpha', type=float, default=0.3, help='Alpha parameter')
    parser.add_argument('--mode', type=str, default='maker', help='Label mode (maker/taker)')
    parser.add_argument('--require-fill', action='store_true', help='Require fill validation')
    
    args = parser.parse_args()
    
    # 加载数据
    if args.input:
        df = load_features_from_path(args.input)
    else:
        df, _ = load_latest_features()
    
    # 生成标签
    df = generate_labels_for_training(df, args.horizon, args.alpha, args.mode, args.require_fill)
    
    # 保存带标签的特征文件
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/realtime_features_with_labels_{timestamp}.parquet"
    
    df.to_parquet(output_file, index=False)
    
    print(f"\n✅ 标签生成完成，保存到: {output_file}")
    print(f"📊 特征文件包含 {len(df)} 行，{len(df.columns)} 列")
    
    # 显示标签列
    label_cols = [col for col in df.columns if col.startswith('label_')]
    print(f"🏷️  标签列: {label_cols}")
    
    return output_file

if __name__ == "__main__":
    main() 