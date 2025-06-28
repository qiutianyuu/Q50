#!/usr/bin/env python3
"""
现实版资金费率套利策略
加入交易成本、对冲成本、不确定性等因素
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

# 配置
FUNDING_FILE = Path("data/funding_oi_1h_2021_2022.parquet")
PRICE_FILE = Path("data/merged_15m_2021_2022.parquet")
OUTPUT_DIR = Path("analysis/funding_arbitrage")

# 成本参数
MAKER_FEE = 0.0001  # 0.01% maker fee
TAKER_FEE = 0.0005  # 0.05% taker fee
SLIPPAGE = 0.0002   # 0.02% slippage
HEDGE_SPREAD = 0.0003  # 0.03% hedge spread (现货/季度价差)
FUNDING_UNCERTAINTY = 0.3  # 30% funding变化不确定性

def load_and_merge_data():
    """加载并合并数据"""
    print("📊 加载数据...")
    
    funding_df = pd.read_parquet(FUNDING_FILE)
    funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'], utc=True)
    funding_df = funding_df.dropna().reset_index(drop=True)
    
    price_df = pd.read_parquet(PRICE_FILE)
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], utc=True)
    
    # 聚合到1小时
    price_1h = price_df.set_index('timestamp').resample('1h').agg({
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    merged_df = funding_df.merge(price_1h, on='timestamp', how='inner')
    merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ 合并完成: {len(merged_df)} 条记录")
    return merged_df

def calculate_realistic_returns(df, threshold=0.001, holding_hours=8):
    """计算现实收益"""
    print(f"\n💰 现实版套利策略 (阈值: {threshold*100:.1f}%):")
    
    trades = []
    capital = 10000
    position = 0
    entry_time = None
    entry_funding = None
    entry_price = None
    
    for i in range(len(df)):
        current_time = df.iloc[i]['timestamp']
        current_funding = df.iloc[i]['funding']
        current_price = df.iloc[i]['close']
        
        # 检查是否需要平仓
        if position != 0:
            hours_held = (current_time - entry_time).total_seconds() / 3600
            
            if hours_held >= holding_hours:
                # 计算实际funding收益（加入不确定性）
                if position == 1:  # 做多
                    funding_return = entry_funding * holding_hours / 8
                    # 加入不确定性
                    funding_return *= (1 + np.random.normal(0, FUNDING_UNCERTAINTY))
                else:  # 做空
                    funding_return = -entry_funding * holding_hours / 8
                    funding_return *= (1 + np.random.normal(0, FUNDING_UNCERTAINTY))
                
                # 计算交易成本
                entry_cost = (MAKER_FEE + SLIPPAGE) * 2  # 开仓和平仓
                hedge_cost = HEDGE_SPREAD * 2  # 对冲成本
                total_cost = entry_cost + hedge_cost
                
                # 净收益
                net_return = funding_return - total_cost
                capital *= (1 + net_return)
                
                # 记录交易
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'entry_funding': entry_funding,
                    'exit_funding': current_funding,
                    'funding_return': funding_return,
                    'total_cost': total_cost,
                    'net_return': net_return,
                    'capital': capital,
                    'hours_held': hours_held
                })
                
                position = 0
                entry_time = None
                entry_funding = None
                entry_price = None
        
        # 检查是否有新开仓机会
        if position == 0:
            if current_funding > threshold:  # 做多机会
                position = 1
                entry_time = current_time
                entry_funding = current_funding
                entry_price = current_price
            elif current_funding < -threshold:  # 做空机会
                position = -1
                entry_time = current_time
                entry_funding = current_funding
                entry_price = current_price
    
    return pd.DataFrame(trades), capital

def analyze_realistic_results(trades_df, final_capital):
    """分析现实结果"""
    if len(trades_df) == 0:
        print("❌ 没有交易记录")
        return
    
    print(f"\n📊 现实策略表现:")
    print(f"总交易数: {len(trades_df)}")
    
    # 收益统计
    total_return = (final_capital - 10000) / 10000
    winning_trades = len(trades_df[trades_df['net_return'] > 0])
    win_rate = winning_trades / len(trades_df)
    
    # 年化收益
    first_trade = trades_df['entry_time'].min()
    last_trade = trades_df['exit_time'].max()
    days = (last_trade - first_trade).days
    annual_return = total_return * (365 / days) if days > 0 else 0
    
    print(f"胜率: {100*win_rate:.1f}%")
    print(f"总收益: {100*total_return:.2f}%")
    print(f"年化收益: {100*annual_return:.2f}%")
    print(f"最终资金: ${final_capital:,.2f}")
    
    # 详细分析
    print(f"\n📈 详细统计:")
    print(f"平均单笔收益: {100*trades_df['net_return'].mean():.3f}%")
    print(f"平均funding收益: {100*trades_df['funding_return'].mean():.3f}%")
    print(f"平均交易成本: {100*trades_df['total_cost'].mean():.3f}%")
    print(f"收益标准差: {100*trades_df['net_return'].std():.3f}%")
    
    # 按方向分析
    long_trades = trades_df[trades_df['position'] == 1]
    short_trades = trades_df[trades_df['position'] == -1]
    
    if len(long_trades) > 0:
        print(f"\n做多交易 ({len(long_trades)}笔):")
        print(f"  平均收益: {100*long_trades['net_return'].mean():.3f}%")
        print(f"  胜率: {100*(long_trades['net_return'] > 0).mean():.1f}%")
    
    if len(short_trades) > 0:
        print(f"\n做空交易 ({len(short_trades)}笔):")
        print(f"  平均收益: {100*short_trades['net_return'].mean():.3f}%")
        print(f"  胜率: {100*(short_trades['net_return'] > 0).mean():.1f}%")
    
    # 最大回撤
    cumulative_returns = (trades_df['capital'] - 10000) / 10000
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / (peak + 1)
    max_drawdown = drawdown.min()
    
    print(f"\n风险指标:")
    print(f"最大回撤: {100*max_drawdown:.2f}%")
    
    # 夏普比率
    if trades_df['net_return'].std() > 0:
        sharpe = trades_df['net_return'].mean() / trades_df['net_return'].std() * np.sqrt(252 * 24)  # 小时级数据
        print(f"夏普比率: {sharpe:.2f}")
    
    return trades_df

def plot_realistic_results(trades_df):
    """绘制现实结果"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 累计收益曲线
    cumulative_returns = (trades_df['capital'] - 10000) / 10000
    axes[0, 0].plot(trades_df['exit_time'], cumulative_returns)
    axes[0, 0].set_title('累计收益曲线')
    axes[0, 0].set_xlabel('时间')
    axes[0, 0].set_ylabel('累计收益')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 单笔收益分布
    axes[0, 1].hist(trades_df['net_return'] * 100, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(trades_df['net_return'].mean() * 100, color='red', linestyle='--', label='均值')
    axes[0, 1].set_title('单笔收益分布')
    axes[0, 1].set_xlabel('收益 (%)')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].legend()
    
    # 3. Funding收益 vs 交易成本
    axes[1, 0].scatter(trades_df['funding_return'] * 100, trades_df['total_cost'] * 100, alpha=0.6)
    axes[1, 0].set_title('Funding收益 vs 交易成本')
    axes[1, 0].set_xlabel('Funding收益 (%)')
    axes[1, 0].set_ylabel('交易成本 (%)')
    
    # 4. 按持仓时间分析
    axes[1, 1].scatter(trades_df['hours_held'], trades_df['net_return'] * 100, alpha=0.6)
    axes[1, 1].set_title('持仓时间 vs 收益')
    axes[1, 1].set_xlabel('持仓时间 (小时)')
    axes[1, 1].set_ylabel('收益 (%)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'realistic_funding_arbitrage.png', dpi=300, bbox_inches='tight')
    print(f"📊 图表已保存: {OUTPUT_DIR / 'realistic_funding_arbitrage.png'}")

def main():
    print("🚀 现实版资金费率套利策略")
    
    # 设置随机种子
    np.random.seed(42)
    
    # 加载数据
    df = load_and_merge_data()
    
    # 计算现实收益
    trades_df, final_capital = calculate_realistic_returns(df, threshold=0.001, holding_hours=8)
    
    # 分析结果
    trades_df = analyze_realistic_results(trades_df, final_capital)
    
    # 绘制结果
    if trades_df is not None and len(trades_df) > 0:
        plot_realistic_results(trades_df)
        trades_df.to_csv(OUTPUT_DIR / 'realistic_funding_trades.csv', index=False)
        print(f"💾 交易记录已保存: {OUTPUT_DIR / 'realistic_funding_trades.csv'}")
    
    print("\n✅ 现实版分析完成!")

if __name__ == "__main__":
    main() 