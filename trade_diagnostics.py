import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def load_trades(file_path):
    """加载交易数据"""
    df = pd.read_csv(file_path)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['duration'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600  # 小时
    return df

def analyze_trades(df):
    """分析交易数据"""
    print("=== 交易诊断报告 ===\n")
    
    # 基础统计
    total_trades = len(df)
    winning_trades = len(df[df['pnl'] > 0])
    losing_trades = len(df[df['pnl'] < 0])
    win_rate = winning_trades / total_trades * 100
    
    print(f"总交易数: {total_trades}")
    print(f"盈利交易: {winning_trades} ({win_rate:.1f}%)")
    print(f"亏损交易: {losing_trades} ({100-win_rate:.1f}%)")
    
    # 盈亏分析
    total_pnl = df['pnl'].sum()
    avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
    max_win = df['pnl'].max()
    max_loss = df['pnl'].min()
    
    print(f"\n总盈亏: ${total_pnl:.2f}")
    print(f"平均盈利: ${avg_win:.2f}")
    print(f"平均亏损: ${avg_loss:.2f}")
    print(f"最大盈利: ${max_win:.2f}")
    print(f"最大亏损: ${max_loss:.2f}")
    
    # R:R分析
    if avg_loss != 0:
        rr_ratio = abs(avg_win / avg_loss)
        print(f"\n理论R:R: {rr_ratio:.2f}")
    else:
        rr_ratio = 0
        print(f"\n理论R:R: N/A (无亏损交易)")
    
    # 期望值分析
    expected_value = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
    print(f"单笔期望值: ${expected_value:.2f}")
    
    # 手续费分析
    entry_fees = df['entry_price'] * df['size'] * 0.00075
    exit_fees = df['exit_price'] * df['size'] * 0.00075
    total_fees = entry_fees.sum() + exit_fees.sum()
    fee_impact = total_fees / abs(total_pnl) * 100 if total_pnl != 0 else 0
    
    print(f"\n总手续费: ${total_fees:.2f}")
    print(f"手续费占盈亏比例: {fee_impact:.1f}%")
    
    # 持仓时间分析
    avg_duration = df['duration'].mean()
    print(f"\n平均持仓时间: {avg_duration:.1f}小时")
    
    # 按退出原因分析
    print(f"\n=== 退出原因分析 ===")
    exit_reasons = df['reason'].value_counts()
    for reason, count in exit_reasons.items():
        reason_pnl = df[df['reason'] == reason]['pnl'].sum()
        reason_win_rate = len(df[(df['reason'] == reason) & (df['pnl'] > 0)]) / count * 100
        print(f"{reason}: {count}次, 盈亏${reason_pnl:.2f}, 胜率{reason_win_rate:.1f}%")
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'rr_ratio': rr_ratio,
        'expected_value': expected_value,
        'total_fees': total_fees,
        'fee_impact': fee_impact
    }

def plot_pnl_distribution(df):
    """绘制盈亏分布图"""
    plt.figure(figsize=(12, 8))
    
    # 盈亏分布直方图
    plt.subplot(2, 2, 1)
    plt.hist(df['pnl'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('盈亏 ($)')
    plt.ylabel('频次')
    plt.title('盈亏分布')
    plt.grid(True, alpha=0.3)
    
    # 盈亏箱线图
    plt.subplot(2, 2, 2)
    plt.boxplot([df[df['pnl'] > 0]['pnl'], df[df['pnl'] < 0]['pnl']], 
                labels=['盈利', '亏损'])
    plt.ylabel('盈亏 ($)')
    plt.title('盈亏箱线图')
    plt.grid(True, alpha=0.3)
    
    # 持仓时间vs盈亏散点图
    plt.subplot(2, 2, 3)
    colors = ['green' if pnl > 0 else 'red' for pnl in df['pnl']]
    plt.scatter(df['duration'], df['pnl'], c=colors, alpha=0.6)
    plt.xlabel('持仓时间 (小时)')
    plt.ylabel('盈亏 ($)')
    plt.title('持仓时间 vs 盈亏')
    plt.grid(True, alpha=0.3)
    
    # 累计盈亏曲线
    plt.subplot(2, 2, 4)
    df_sorted = df.sort_values('exit_time').copy()
    df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()
    plt.plot(range(len(df_sorted)), df_sorted['cumulative_pnl'], linewidth=2)
    plt.xlabel('交易序号')
    plt.ylabel('累计盈亏 ($)')
    plt.title('累计盈亏曲线')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pnl_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_market_regime(df):
    """分析不同市场环境下的表现"""
    print(f"\n=== 市场环境分析 ===")
    
    # 按月份分析
    df['month'] = df['exit_time'].dt.to_period('M')
    monthly_stats = df.groupby('month').agg({
        'pnl': ['sum', 'mean', 'count'],
        'duration': 'mean'
    }).round(2)
    
    print("月度表现:")
    print(monthly_stats)
    
    # 按持仓时间分组
    df['duration_group'] = pd.cut(df['duration'], 
                                 bins=[0, 4, 24, 72, float('inf')], 
                                 labels=['<4h', '4-24h', '24-72h', '>72h'])
    
    duration_stats = df.groupby('duration_group').agg({
        'pnl': ['sum', 'mean', 'count'],
        'reason': lambda x: (x == 'tp').sum() / len(x) * 100
    }).round(2)
    
    print(f"\n按持仓时间分组:")
    print(duration_stats)

def main():
    # 加载数据
    df = load_trades('rexking_eth_10_4_trades.csv')
    
    # 基础分析
    stats = analyze_trades(df)
    
    # 绘制图表
    plot_pnl_distribution(df)
    
    # 市场环境分析
    analyze_market_regime(df)
    
    # 输出优化建议
    print(f"\n=== 优化建议 ===")
    
    if stats['win_rate'] < 45:
        print("⚠️  胜率偏低，建议:")
        print("  - 提高ADX阈值到20+")
        print("  - 加强funding过滤条件")
        print("  - 增加假突破确认机制")
    
    if stats['rr_ratio'] < 1.5:
        print("⚠️  R:R比例偏低，建议:")
        print("  - 优化止盈目标设置")
        print("  - 改进跟踪止损逻辑")
        print("  - 考虑分阶段止盈")
    
    if stats['fee_impact'] > 20:
        print("⚠️  手续费影响过大，建议:")
        print("  - 减少交易频率")
        print("  - 提高单笔盈利目标")
        print("  - 优化仓位管理")
    
    if stats['expected_value'] < 0:
        print("⚠️  期望值为负，建议:")
        print("  - 重新评估信号质量")
        print("  - 调整风险参数")
        print("  - 考虑暂停交易")

if __name__ == "__main__":
    main() 