#!/usr/bin/env python3
"""
快速生成新Label - 未来N根累计收益阈值
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def make_label_from_returns(df: pd.DataFrame, horizon: int = 6, pos_thr: float = 0.0015, neg_thr: float = -0.0015):
    """生成基于未来N根累计收益的标签"""
    
    # 计算未来N根的累计收益
    future_returns = df['close'].pct_change(horizon).shift(-horizon)
    
    # 生成标签
    # 1 = 未来收益 > pos_thr (做多信号)
    # 0 = 未来收益 < neg_thr (做空信号)  
    # -1 = 中间区域 (不交易)
    
    labels = np.where(future_returns > pos_thr, 1, 
                     np.where(future_returns < neg_thr, 0, -1))
    
    # 创建结果DataFrame
    result = pd.DataFrame({
        'timestamp': df['timestamp'],
        'close': df['close'],
        'future_return': future_returns,
        'label': labels
    })
    
    # 统计信息
    total_samples = len(result)
    long_signals = (result['label'] == 1).sum()
    short_signals = (result['label'] == 0).sum()
    no_trade = (result['label'] == -1).sum()
    
    print(f"📊 Label统计:")
    print(f"总样本: {total_samples}")
    print(f"做多信号 (1): {long_signals} ({long_signals/total_samples*100:.1f}%)")
    print(f"做空信号 (0): {short_signals} ({short_signals/total_samples*100:.1f}%)")
    print(f"不交易 (-1): {no_trade} ({no_trade/total_samples*100:.1f}%)")
    print(f"交易信号占比: {(long_signals+short_signals)/total_samples*100:.1f}%")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='生成基于未来收益的标签')
    parser.add_argument('--kline', required=True, help='K线数据文件路径')
    parser.add_argument('--horizon', type=int, default=6, help='未来N根K线')
    parser.add_argument('--pos_thr', type=float, default=0.0015, help='做多阈值')
    parser.add_argument('--neg_thr', type=float, default=-0.0015, help='做空阈值')
    parser.add_argument('--out', required=True, help='输出文件路径')
    
    args = parser.parse_args()
    
    print(f"📥 读取K线数据: {args.kline}")
    df = pd.read_parquet(args.kline)
    print(f"数据形状: {df.shape}")
    
    # 确保timestamp列存在
    if 'timestamp' not in df.columns:
        print("❌ 错误: 数据中没有timestamp列")
        return
    
    # 生成标签
    print(f"🏷️ 生成标签: 未来{args.horizon}根, 阈值[{args.neg_thr}, {args.pos_thr}]")
    result = make_label_from_returns(df, args.horizon, args.pos_thr, args.neg_thr)
    
    # 保存结果
    result.to_csv(args.out, index=False)
    print(f"✅ 标签已保存: {args.out}")
    
    # 显示前几行示例
    print("\n📋 前10行示例:")
    print(result.head(10))

if __name__ == "__main__":
    main() 