#!/usr/bin/env python3
"""
分析改进回测结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_trades():
    """分析交易数据"""
    try:
        df = pd.read_csv('improved_cost_trades.csv')
        print("=== 改进回测交易分析 ===")
        print(f"总交易数: {len(df)}")
        print(f"多头交易: {(df.position==1).sum()}")
        print(f"空头交易: {(df.position==-1).sum()}")
        print(f"平均持仓时间: {df.bars_held.mean():.1f} bars")
        print(f"平均毛收益: {100*df.gross_return.mean():.3f}%")
        print(f"平均净收益: {100*df.net_return.mean():.3f}%")
        print(f"平均成本: ${df.costs.mean():.2f}")
        print(f"成本占毛收益比例: {100*df.costs.mean()/(abs(df.gross_return.mean())*1000):.1f}%")
        
        # 分别分析多头和空头
        long_trades = df[df.position == 1]
        short_trades = df[df.position == -1]
        
        print(f"\n=== 多头交易分析 ===")
        print(f"多头交易数: {len(long_trades)}")
        print(f"多头胜率: {100*(long_trades.pnl > 0).mean():.1f}%")
        print(f"多头平均收益: {100*long_trades.net_return.mean():.3f}%")
        print(f"多头平均成本: ${long_trades.costs.mean():.2f}")
        
        print(f"\n=== 空头交易分析 ===")
        print(f"空头交易数: {len(short_trades)}")
        print(f"空头胜率: {100*(short_trades.pnl > 0).mean():.1f}%")
        print(f"空头平均收益: {100*short_trades.net_return.mean():.3f}%")
        print(f"空头平均成本: ${short_trades.costs.mean():.2f}")
        
        # 分析收益分布
        print(f"\n=== 收益分布分析 ===")
        print(f"收益标准差: {100*df.net_return.std():.3f}%")
        print(f"最大单笔收益: {100*df.net_return.max():.3f}%")
        print(f"最大单笔亏损: {100*df.net_return.min():.3f}%")
        print(f"收益中位数: {100*df.net_return.median():.3f}%")
        
        # 分析时间分布
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['hour'] = df['entry_time'].dt.hour
        hourly_performance = df.groupby('hour').agg({
            'net_return': ['count', 'mean', 'std'],
            'pnl': 'sum'
        }).round(4)
        
        print(f"\n=== 按小时分析 ===")
        print("小时 | 交易数 | 平均收益 | 收益标准差 | 总PnL")
        print("-" * 50)
        for hour in range(24):
            if hour in hourly_performance.index:
                count = hourly_performance.loc[hour, ('net_return', 'count')]
                mean_ret = hourly_performance.loc[hour, ('net_return', 'mean')]
                std_ret = hourly_performance.loc[hour, ('net_return', 'std')]
                total_pnl = hourly_performance.loc[hour, ('pnl', 'sum')]
                print(f"{hour:2d}   | {count:6d} | {100*mean_ret:7.3f}% | {100*std_ret:8.3f}% | ${total_pnl:8.2f}")
        
        return df
        
    except Exception as e:
        print(f"Error analyzing trades: {e}")
        return None

def plot_results(df):
    """绘制结果图表"""
    if df is None:
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 累积收益曲线
    df_sorted = df.sort_values('entry_time')
    cumulative_pnl = df_sorted['pnl'].cumsum()
    axes[0, 0].plot(df_sorted['entry_time'], cumulative_pnl)
    axes[0, 0].set_title('Cumulative PnL')
    axes[0, 0].set_ylabel('PnL ($)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 收益分布直方图
    axes[0, 1].hist(df['net_return'] * 100, bins=50, alpha=0.7)
    axes[0, 1].set_title('Return Distribution')
    axes[0, 1].set_xlabel('Return (%)')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. 多头vs空头收益对比
    long_returns = df[df.position == 1]['net_return'] * 100
    short_returns = df[df.position == -1]['net_return'] * 100
    axes[1, 0].boxplot([long_returns, short_returns], labels=['Long', 'Short'])
    axes[1, 0].set_title('Long vs Short Returns')
    axes[1, 0].set_ylabel('Return (%)')
    
    # 4. 按小时的平均收益
    hourly_returns = df.groupby('hour')['net_return'].mean() * 100
    axes[1, 1].bar(hourly_returns.index, hourly_returns.values)
    axes[1, 1].set_title('Average Returns by Hour')
    axes[1, 1].set_xlabel('Hour')
    axes[1, 1].set_ylabel('Average Return (%)')
    
    plt.tight_layout()
    plt.savefig('improved_backtest_analysis.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 improved_backtest_analysis.png")

def main():
    print("🔍 分析改进回测结果...")
    df = analyze_trades()
    
    if df is not None:
        print("\n📊 生成分析图表...")
        plot_results(df)
        print("✅ 分析完成！")
    else:
        print("❌ 分析失败")

if __name__ == "__main__":
    main() 