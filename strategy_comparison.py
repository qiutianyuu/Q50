#!/usr/bin/env python3
"""
RexKing – Strategy Comparison Analysis

对比分析不同版本策略的表现
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_results():
    """加载不同版本的结果"""
    results = {}
    
    # 原始版本结果
    if Path('enhanced_trades_15m.csv').exists():
        df_original = pd.read_csv('enhanced_trades_15m.csv')
        results['Original'] = analyze_trades(df_original, 'Original')
    
    # 修复版本结果
    if Path('enhanced_trades_15m_fixed.csv').exists():
        df_fixed = pd.read_csv('enhanced_trades_15m_fixed.csv')
        results['Fixed'] = analyze_trades(df_fixed, 'Fixed')
    
    # 优化版本结果
    if Path('optimized_trades_15m.csv').exists():
        df_optimized = pd.read_csv('optimized_trades_15m.csv')
        results['Optimized'] = analyze_trades(df_optimized, 'Optimized')
    
    return results

def analyze_trades(df, version_name):
    """分析交易数据"""
    if len(df) == 0:
        return None
    
    # 基础统计
    total_trades = len(df)
    winning_trades = len(df[df['pnl'] > 0])
    win_rate = winning_trades / total_trades
    
    # 收益统计
    total_return = df['actual_pnl'].sum() / 10000  # 相对于初始资金
    avg_trade_return = df['pnl'].mean()
    
    # 风险统计
    max_loss = df['pnl'].min()
    max_gain = df['pnl'].max()
    
    # 计算最大回撤
    cumulative_pnl = df['actual_pnl'].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = (cumulative_pnl - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 计算夏普比率
    daily_pnl = df.groupby(df['exit_time'].str[:10])['actual_pnl'].sum()
    sharpe_ratio = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0
    
    # 计算年化收益
    if len(df) > 0:
        first_date = pd.to_datetime(df['entry_time'].iloc[0])
        last_date = pd.to_datetime(df['exit_time'].iloc[-1])
        days = (last_date - first_date).days
        annual_return = ((1 + total_return) ** (365 / days) - 1) * 100 if days > 0 else 0
    else:
        annual_return = 0
    
    return {
        'version': version_name,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'avg_trade_return': avg_trade_return,
        'max_loss': max_loss,
        'max_gain': max_gain,
        'final_equity': 10000 * (1 + total_return)
    }

def create_comparison_table(results):
    """创建对比表格"""
    if not results:
        print("❌ 没有找到可比较的结果")
        return
    
    # 创建对比表格
    comparison_data = []
    for version, result in results.items():
        if result is not None:
            comparison_data.append(result)
    
    if not comparison_data:
        print("❌ 没有有效的结果数据")
        return
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print("📊 策略对比分析")
    print("=" * 80)
    print(df_comparison.to_string(index=False, float_format='%.4f'))
    
    # 保存对比结果
    df_comparison.to_csv('strategy_comparison.csv', index=False)
    print(f"\n✅ 对比结果已保存到: strategy_comparison.csv")
    
    return df_comparison

def plot_comparison_charts(results):
    """绘制对比图表"""
    if not results:
        return
    
    # 准备数据
    comparison_data = []
    for version, result in results.items():
        if result is not None:
            comparison_data.append(result)
    
    if not comparison_data:
        return
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('RexKing Strategy Comparison', fontsize=16, fontweight='bold')
    
    # 1. 总收益对比
    axes[0, 0].bar(df_comparison['version'], df_comparison['total_return'] * 100, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 0].set_title('Total Return (%)')
    axes[0, 0].set_ylabel('Return (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 胜率对比
    axes[0, 1].bar(df_comparison['version'], df_comparison['win_rate'] * 100,
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 1].set_title('Win Rate (%)')
    axes[0, 1].set_ylabel('Win Rate (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 最大回撤对比
    axes[1, 0].bar(df_comparison['version'], df_comparison['max_drawdown'] * 100,
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 0].set_title('Maximum Drawdown (%)')
    axes[1, 0].set_ylabel('Drawdown (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. 夏普比率对比
    axes[1, 1].bar(df_comparison['version'], df_comparison['sharpe_ratio'],
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 1].set_title('Sharpe Ratio')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ 对比图表已保存到: strategy_comparison.png")
    plt.show()

def analyze_trade_patterns(results):
    """分析交易模式"""
    print("\n🔍 交易模式分析")
    print("=" * 50)
    
    for version, result in results.items():
        if result is None:
            continue
            
        print(f"\n📊 {version} 版本:")
        print(f"  总交易数: {result['total_trades']}")
        print(f"  胜率: {result['win_rate']:.2%}")
        print(f"  平均单笔收益: {result['avg_trade_return']:.2%}")
        print(f"  最大单笔亏损: {result['max_loss']:.2%}")
        print(f"  最大单笔盈利: {result['max_gain']:.2%}")
        print(f"  年化收益: {result['annual_return']:.2f}%")
        print(f"  最大回撤: {result['max_drawdown']:.2%}")
        print(f"  夏普比率: {result['sharpe_ratio']:.2f}")

def main():
    print("=== RexKing Strategy Comparison Analysis ===")
    
    # 加载结果
    results = load_results()
    
    if not results:
        print("❌ 没有找到任何回测结果文件")
        return
    
    # 创建对比表格
    df_comparison = create_comparison_table(results)
    
    # 分析交易模式
    analyze_trade_patterns(results)
    
    # 绘制对比图表
    try:
        plot_comparison_charts(results)
    except Exception as e:
        print(f"⚠️ 图表生成失败: {e}")
    
    # 总结
    print("\n🎯 策略优化总结:")
    print("=" * 50)
    
    if 'Original' in results and 'Optimized' in results:
        orig = results['Original']
        opt = results['Optimized']
        
        if orig and opt:
            print(f"✅ 交易数量: {orig['total_trades']} → {opt['total_trades']} ({opt['total_trades'] - orig['total_trades']:+d})")
            print(f"✅ 胜率提升: {orig['win_rate']:.2%} → {opt['win_rate']:.2%} ({opt['win_rate'] - orig['win_rate']:+.2%})")
            print(f"✅ 年化收益: {orig['annual_return']:.1f}% → {opt['annual_return']:.1f}% ({opt['annual_return'] - orig['annual_return']:+.1f}%)")
            print(f"✅ 最大回撤: {orig['max_drawdown']:.2%} → {opt['max_drawdown']:.2%} ({opt['max_drawdown'] - orig['max_drawdown']:+.2%})")
            print(f"✅ 夏普比率: {orig['sharpe_ratio']:.2f} → {opt['sharpe_ratio']:.2f} ({opt['sharpe_ratio'] - orig['sharpe_ratio']:+.2f})")
    
    print("\n🎉 对比分析完成!")

if __name__ == "__main__":
    main() 