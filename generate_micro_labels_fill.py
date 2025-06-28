#!/usr/bin/env python3
"""
生成带填单验证的微观标签脚本
使用最佳参数组合生成标签并保存到特征文件
"""

import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
from utils.labeling import make_labels, get_label_stats

def load_latest_features():
    """Load the latest micro features file"""
    files = glob.glob("data/micro_features_*.parquet")
    if not files:
        raise FileNotFoundError("No micro features files found")
    
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading: {latest_file}")
    df = pd.read_parquet(latest_file)
    print(f"Loaded {len(df)} rows")
    return df

def main():
    df = load_latest_features()
    
    # Best parameters from scan
    horizon = 240  # 4 minutes
    alpha = 0.3
    fee_rate = 0.0005
    mode = 'maker'
    require_fill = True
    
    print(f"Generating labels with: horizon={horizon}, alpha={alpha}, mode={mode}, require_fill={require_fill}")
    
    # Generate labels
    labels = make_labels(df['mid_price'], df['rel_spread'], horizon, alpha, fee_rate, mode, require_fill)
    
    # Add labels to dataframe
    df['label'] = labels
    
    # Get label statistics
    stats = get_label_stats(labels)
    print(f"\nLabel distribution:")
    print(f"Long: {stats['long_pct']:.1f}% ({stats['long_count']} samples)")
    print(f"Short: {stats['short_pct']:.1f}% ({stats['short_count']} samples)")
    print(f"Neutral: {stats['neutral_pct']:.1f}% ({stats['neutral_count']} samples)")
    
    # Save labeled data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/realtime_features_with_fill_labels_{timestamp}.parquet"
    df.to_parquet(output_file, index=False)
    print(f"\nLabeled data saved to: {output_file}")
    
    # Show sample of labeled data
    print(f"\nSample of labeled data:")
    print(df[['timestamp', 'mid_price', 'rel_spread', 'label']].head(10))

if __name__ == "__main__":
    main() 