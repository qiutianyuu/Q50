#!/usr/bin/env python3
"""
成本感知标签重新设计 - 基于净收益生成标签
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def redesign_label_cost_aware(kline_path, horizon, pos_thr, neg_thr, out_path, 
                             taker_fee=0.0004, maker_fee=0.0002, slippage=0.00025, 
                             funding_fee=0.0001):
    """
    生成成本感知的标签
    
    Args:
        kline_path: K线数据路径
        horizon: 持仓周期（K线根数）
        pos_thr: 做多阈值（净收益）
        neg_thr: 做空阈值（净收益）
        out_path: 输出路径
        taker_fee: 吃单手续费
        maker_fee: 挂单手续费
        slippage: 预估滑点
        funding_fee: 资金费率（若持仓跨结算）
    """
    print(f"📊 读取K线数据: {kline_path}")
    df = pd.read_parquet(kline_path)
    
    # 计算未来收益
    print(f"🔧 计算未来{horizon}根K线收益...")
    df['gross_ret'] = df['close'].pct_change(horizon).shift(-horizon)
    
    # 计算总成本（最坏情况：taker-taker）
    total_cost = taker_fee * 2 + slippage + funding_fee
    print(f"💰 总成本: {total_cost:.4f} ({total_cost*100:.3f}%)")
    print(f"  - 手续费: {taker_fee*2:.4f} (taker-taker)")
    print(f"  - 滑点: {slippage:.4f}")
    print(f"  - 资金费: {funding_fee:.4f}")
    
    # 计算净收益
    df['net_ret'] = df['gross_ret'] - total_cost
    
    # 生成标签
    print(f"🏷️ 生成标签 (做多阈值: {pos_thr:.4f}, 做空阈值: {neg_thr:.4f})...")
    df['label'] = -1  # 默认不交易
    df.loc[df['net_ret'] >= pos_thr, 'label'] = 1   # 做多
    df.loc[df['net_ret'] <= neg_thr, 'label'] = 0   # 做空
    
    # 统计标签分布
    total_samples = len(df)
    long_signals = (df['label'] == 1).sum()
    short_signals = (df['label'] == 0).sum()
    no_trade = (df['label'] == -1).sum()
    
    print(f"\n📈 标签分布统计:")
    print(f"总样本: {total_samples:,}")
    print(f"做多信号: {long_signals:,} ({long_signals/total_samples*100:.1f}%)")
    print(f"做空信号: {short_signals:,} ({short_signals/total_samples*100:.1f}%)")
    print(f"不交易: {no_trade:,} ({no_trade/total_samples*100:.1f}%)")
    print(f"交易信号占比: {(long_signals+short_signals)/total_samples*100:.1f}%")
    
    # 分析净收益分布
    trade_mask = df['label'] != -1
    if trade_mask.sum() > 0:
        trade_returns = df.loc[trade_mask, 'net_ret']
        print(f"\n💰 交易信号净收益分析:")
        print(f"平均净收益: {trade_returns.mean():.6f} ({trade_returns.mean()*100:.4f}%)")
        print(f"净收益标准差: {trade_returns.std():.6f}")
        print(f"正收益占比: {(trade_returns > 0).sum() / len(trade_returns)*100:.1f}%")
        print(f"净收益分位数:")
        print(f"  25%: {trade_returns.quantile(0.25):.6f}")
        print(f"  50%: {trade_returns.quantile(0.50):.6f}")
        print(f"  75%: {trade_returns.quantile(0.75):.6f}")
    
    # 保存结果
    output_df = df[['timestamp', 'close', 'gross_ret', 'net_ret', 'label']].copy()
    output_df = output_df.dropna(subset=['label'])
    
    # 确保输出目录存在
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_df.to_csv(out_path, index=False)
    print(f"\n✅ 标签已保存: {out_path}")
    print(f"有效样本数: {len(output_df):,}")
    
    return output_df

def main():
    parser = argparse.ArgumentParser(description='成本感知标签重新设计')
    parser.add_argument('--kline', type=str, required=True, help='K线数据路径')
    parser.add_argument('--horizon', type=int, default=4, help='持仓周期(K线根数)')
    parser.add_argument('--pos_thr', type=float, default=0.00075, help='做多阈值(净收益)')
    parser.add_argument('--neg_thr', type=float, default=-0.00075, help='做空阈值(净收益)')
    parser.add_argument('--out', type=str, required=True, help='输出路径')
    parser.add_argument('--taker_fee', type=float, default=0.0004, help='吃单手续费')
    parser.add_argument('--maker_fee', type=float, default=0.0002, help='挂单手续费')
    parser.add_argument('--slippage', type=float, default=0.00025, help='预估滑点')
    parser.add_argument('--funding_fee', type=float, default=0.0001, help='资金费率')
    
    args = parser.parse_args()
    
    print("🏷️ 成本感知标签重新设计")
    print(f"📁 K线数据: {args.kline}")
    print(f"⏱️ 持仓周期: {args.horizon}根K线")
    print(f"📊 做多阈值: {args.pos_thr:.4f} ({args.pos_thr*100:.3f}%)")
    print(f"📉 做空阈值: {args.neg_thr:.4f} ({args.neg_thr*100:.3f}%)")
    
    redesign_label_cost_aware(
        kline_path=args.kline,
        horizon=args.horizon,
        pos_thr=args.pos_thr,
        neg_thr=args.neg_thr,
        out_path=args.out,
        taker_fee=args.taker_fee,
        maker_fee=args.maker_fee,
        slippage=args.slippage,
        funding_fee=args.funding_fee
    )

if __name__ == "__main__":
    main() 