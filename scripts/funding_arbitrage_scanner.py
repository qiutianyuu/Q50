#!/usr/bin/env python3
"""
资金费率套利扫描器
分析funding rate偏离机会，计算潜在收益
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

def load_and_merge_data():
    """加载并合并funding和价格数据"""
    print("📊 加载数据...")
    
    # 加载funding数据
    funding_df = pd.read_parquet(FUNDING_FILE)
    funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'], utc=True)
    funding_df = funding_df.dropna().reset_index(drop=True)
    
    # 加载价格数据并聚合到1小时
    price_df = pd.read_parquet(PRICE_FILE)
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], utc=True)
    
    # 聚合到1小时
    price_1h = price_df.set_index('timestamp').resample('1H').agg({
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    # 合并数据
    merged_df = funding_df.merge(price_1h, on='timestamp', how='inner')
    merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ 合并完成: {len(merged_df)} 条记录")
    print(f"时间范围: {merged_df['timestamp'].min()} 到 {merged_df['timestamp'].max()}")
    
    return merged_df

def analyze_funding_distribution(df):
    """分析funding分布"""
    print("\n📈 Funding分布分析:")
    
    funding = df['funding'].dropna()
    
    print(f"均值: {funding.mean():.6f}")
    print(f"标准差: {funding.std():.6f}")
    print(f"最小值: {funding.min():.6f}")
    print(f"最大值: {funding.max():.6f}")
    
    # 分位数
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = funding.quantile(p/100)
        print(f"{p}%分位: {value:.6f}")
    
    # 极端值统计
    extreme_positive = (funding > 0.001).sum()  # >0.1%
    extreme_negative = (funding < -0.001).sum()  # <-0.1%
    
    print(f"\n极端值统计:")
    print(f"Funding > 0.1%: {extreme_positive} 次 ({100*extreme_positive/len(funding):.1f}%)")
    print(f"Funding < -0.1%: {extreme_negative} 次 ({100*extreme_negative/len(funding):.1f}%)")
    
    return funding

def identify_arbitrage_opportunities(df, threshold=0.001):
    """识别套利机会"""
    print(f"\n🔍 识别套利机会 (阈值: {threshold*100:.1f}%):")
    
    # 标记套利机会
    df['arb_opportunity'] = 0
    df.loc[df['funding'] > threshold, 'arb_opportunity'] = 1   # 做多永续
    df.loc[df['funding'] < -threshold, 'arb_opportunity'] = -1 # 做空永续
    
    # 统计机会
    long_opps = (df['arb_opportunity'] == 1).sum()
    short_opps = (df['arb_opportunity'] == -1).sum()
    total_opps = long_opps + short_opps
    
    print(f"做多机会: {long_opps} 次")
    print(f"做空机会: {short_opps} 次") 
    print(f"总机会: {total_opps} 次 ({100*total_opps/len(df):.1f}%)")
    
    # 计算平均funding偏离
    long_funding = df[df['arb_opportunity'] == 1]['funding'].mean()
    short_funding = df[df['arb_opportunity'] == -1]['funding'].mean()
    
    print(f"做多时平均funding: {long_funding:.6f}")
    print(f"做空时平均funding: {short_funding:.6f}")
    
    return df

def simulate_arbitrage_strategy(df, threshold=0.001, holding_hours=8):
    """模拟套利策略"""
    print(f"\n💰 模拟套利策略:")
    print(f"阈值: {threshold*100:.1f}%")
    print(f"持仓时间: {holding_hours} 小时")
    
    # 初始化
    positions = []
    capital = 10000  # 初始资金
    trades = []
    
    for i in range(len(df)):
        current_time = df.iloc[i]['timestamp']
        current_funding = df.iloc[i]['funding']
        current_price = df.iloc[i]['close']
        
        # 检查是否有新机会
        if df.iloc[i]['arb_opportunity'] != 0:
            # 计算预期收益
            if df.iloc[i]['arb_opportunity'] == 1:  # 做多永续
                expected_return = current_funding * holding_hours / 8  # 8小时一个周期
                position_type = 'long'
            else:  # 做空永续
                expected_return = -current_funding * holding_hours / 8
                position_type = 'short'
            
            # 记录交易
            trades.append({
                'entry_time': current_time,
                'position_type': position_type,
                'entry_price': current_price,
                'funding_rate': current_funding,
                'expected_return': expected_return,
                'capital': capital
            })
            
            # 更新资金
            capital *= (1 + expected_return)
    
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        print("❌ 没有找到套利机会")
        return None
    
    # 计算策略表现
    total_trades = len(trades_df)
    total_return = (capital - 10000) / 10000
    winning_trades = len(trades_df[trades_df['expected_return'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 年化收益
    first_trade = trades_df['entry_time'].min()
    last_trade = trades_df['entry_time'].max()
    days = (last_trade - first_trade).days
    annual_return = total_return * (365 / days) if days > 0 else 0
    
    print(f"\n📊 策略表现:")
    print(f"总交易数: {total_trades}")
    print(f"胜率: {100*win_rate:.1f}%")
    print(f"总收益: {100*total_return:.2f}%")
    print(f"年化收益: {100*annual_return:.2f}%")
    print(f"最终资金: ${capital:,.2f}")
    
    # 按类型分析
    long_trades = trades_df[trades_df['position_type'] == 'long']
    short_trades = trades_df[trades_df['position_type'] == 'short']
    
    if len(long_trades) > 0:
        long_avg_return = long_trades['expected_return'].mean()
        print(f"做多平均收益: {100*long_avg_return:.3f}%")
    
    if len(short_trades) > 0:
        short_avg_return = short_trades['expected_return'].mean()
        print(f"做空平均收益: {100*short_avg_return:.3f}%")
    
    return trades_df

def plot_results(df, trades_df):
    """绘制结果图表"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Funding分布
    axes[0, 0].hist(df['funding'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(df['funding'].mean(), color='red', linestyle='--', label='均值')
    axes[0, 0].axvline(0.001, color='green', linestyle='--', label='+0.1%阈值')
    axes[0, 0].axvline(-0.001, color='green', linestyle='--', label='-0.1%阈值')
    axes[0, 0].set_title('Funding Rate分布')
    axes[0, 0].set_xlabel('Funding Rate')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].legend()
    
    # 2. Funding时间序列
    sample_df = df[::24]  # 每24小时取一个点
    axes[0, 1].plot(sample_df['timestamp'], sample_df['funding'])
    axes[0, 1].axhline(y=0.001, color='green', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(y=-0.001, color='green', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Funding Rate时间序列')
    axes[0, 1].set_xlabel('时间')
    axes[0, 1].set_ylabel('Funding Rate')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 价格vs Funding
    axes[1, 0].scatter(df['funding'], df['close'], alpha=0.5, s=1)
    axes[1, 0].set_title('价格 vs Funding Rate')
    axes[1, 0].set_xlabel('Funding Rate')
    axes[1, 0].set_ylabel('价格')
    
    # 4. 策略收益曲线
    if trades_df is not None and len(trades_df) > 0:
        cumulative_return = (trades_df['capital'] - 10000) / 10000
        axes[1, 1].plot(trades_df['entry_time'], cumulative_return)
        axes[1, 1].set_title('策略累计收益')
        axes[1, 1].set_xlabel('时间')
        axes[1, 1].set_ylabel('累计收益')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'funding_arbitrage_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📊 图表已保存: {OUTPUT_DIR / 'funding_arbitrage_analysis.png'}")

def main():
    print("🚀 资金费率套利扫描器启动")
    
    # 加载数据
    df = load_and_merge_data()
    
    # 分析分布
    funding = analyze_funding_distribution(df)
    
    # 识别机会
    df = identify_arbitrage_opportunities(df, threshold=0.001)
    
    # 模拟策略
    trades_df = simulate_arbitrage_strategy(df, threshold=0.001, holding_hours=8)
    
    # 绘制结果
    plot_results(df, trades_df)
    
    # 保存结果
    if trades_df is not None:
        trades_df.to_csv(OUTPUT_DIR / 'funding_arbitrage_trades.csv', index=False)
        print(f"💾 交易记录已保存: {OUTPUT_DIR / 'funding_arbitrage_trades.csv'}")
    
    print("\n✅ 分析完成!")

if __name__ == "__main__":
    main() 