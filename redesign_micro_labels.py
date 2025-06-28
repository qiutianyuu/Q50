#!/usr/bin/env python3
"""
重新设计微观标签 - P0优先级
规则：
- horizon = 15步（约75秒）
- |ΔP| > (2 × spread + fee) 才标记为1/-1，否则为0
- 目标比例：45%做多，45%做空，10%中性
- fee = 0.001 (0.1%)
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_latest_features():
    """加载最新的特征文件"""
    feature_files = glob.glob('data/analysis/micro_features_*.parquet')
    if not feature_files:
        raise FileNotFoundError("未找到特征文件")
    
    latest_file = max(feature_files, key=os.path.getctime)
    logger.info(f"加载特征文件: {latest_file}")
    
    df = pd.read_parquet(latest_file)
    logger.info(f"数据形状: {df.shape}")
    
    return df

def calculate_new_labels(df, horizon=15, fee_rate=0.001):
    """计算新的标签"""
    logger.info(f"计算新标签: horizon={horizon}, fee_rate={fee_rate}")
    
    # 确保必要的列存在
    required_cols = ['timestamp', 'mid_price', 'spread']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")
    
    # 计算未来价格变化
    df = df.copy()
    df['future_price'] = df['mid_price'].shift(-horizon)
    df['price_change'] = df['future_price'] - df['mid_price']
    df['price_change_pct'] = df['price_change'] / df['mid_price']
    
    # 计算阈值：1 × spread + fee (降低阈值)
    df['threshold'] = df['spread'] / df['mid_price'] + fee_rate
    
    # 生成新标签
    df['label_new'] = 0  # 默认中性
    
    # 做多：价格涨幅超过阈值
    long_mask = df['price_change_pct'] > df['threshold']
    df.loc[long_mask, 'label_new'] = 1
    
    # 做空：价格跌幅超过阈值
    short_mask = df['price_change_pct'] < -df['threshold']
    df.loc[short_mask, 'label_new'] = -1
    
    # 移除无法计算未来价格的行
    df = df.dropna(subset=['future_price', 'price_change_pct'])
    
    return df

def analyze_label_distribution(df):
    """分析标签分布"""
    logger.info("分析标签分布...")
    
    # 旧标签分布
    if 'label_binary' in df.columns:
        old_dist = df['label_binary'].value_counts().sort_index()
        logger.info(f"旧标签分布 (label_binary):")
        for label, count in old_dist.items():
            pct = count / len(df) * 100
            logger.info(f"  {label}: {count} ({pct:.1f}%)")
    
    # 新标签分布
    new_dist = df['label_new'].value_counts().sort_index()
    logger.info(f"新标签分布 (label_new):")
    for label, count in new_dist.items():
        pct = count / len(df) * 100
        logger.info(f"  {label}: {count} ({pct:.1f}%)")
    
    # 价格变化统计
    logger.info(f"价格变化统计:")
    logger.info(f"  平均变化: {df['price_change_pct'].mean():.6f}")
    logger.info(f"  标准差: {df['price_change_pct'].std():.6f}")
    logger.info(f"  最小值: {df['price_change_pct'].min():.6f}")
    logger.info(f"  最大值: {df['price_change_pct'].max():.6f}")
    
    # 阈值统计
    logger.info(f"阈值统计:")
    logger.info(f"  平均阈值: {df['threshold'].mean():.6f}")
    logger.info(f"  阈值标准差: {df['threshold'].std():.6f}")
    
    return new_dist

def save_new_features(df, timestamp):
    """保存新的特征文件"""
    # 选择需要的列
    exclude_cols = ['future_price', 'price_change', 'price_change_pct', 'threshold']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    new_df = df[feature_cols].copy()
    
    # 重命名标签列
    new_df['label_binary'] = new_df['label_new']
    new_df = new_df.drop('label_new', axis=1)
    
    # 保存文件
    output_path = f"data/analysis/micro_features_new_labels_{timestamp}.parquet"
    new_df.to_parquet(output_path, index=False)
    logger.info(f"新特征文件已保存: {output_path}")
    
    return output_path

def main():
    """主函数"""
    logger.info("开始重新设计微观标签...")
    
    # 1. 加载数据
    df = load_latest_features()
    
    # 2. 计算新标签 - 调整参数
    # 降低阈值：只用1倍spread，不用2倍
    # 增加horizon：30步（约2.5分钟）
    df = calculate_new_labels(df, horizon=30, fee_rate=0.0005)
    
    # 3. 分析分布
    label_dist = analyze_label_distribution(df)
    
    # 4. 保存新特征文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = save_new_features(df, timestamp)
    
    logger.info("标签重新设计完成!")
    logger.info(f"输出文件: {output_path}")
    
    # 5. 输出建议
    long_pct = label_dist.get(1, 0) / len(df) * 100
    short_pct = label_dist.get(-1, 0) / len(df) * 100
    neutral_pct = label_dist.get(0, 0) / len(df) * 100
    
    logger.info(f"标签比例: 做多{long_pct:.1f}%, 做空{short_pct:.1f}%, 中性{neutral_pct:.1f}%")
    
    if long_pct < 30 or short_pct < 30:
        logger.warning("标签分布可能不够平衡，建议调整阈值或horizon")
    else:
        logger.info("标签分布相对平衡，可以继续训练模型")

if __name__ == "__main__":
    main() 