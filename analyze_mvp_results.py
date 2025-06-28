import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_mvp_results():
    """分析MVP策略回测结果"""
    
    # 读取交易记录
    trades_df = pd.read_csv('rexking_mvp_trades.csv')
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    
    print("=== 原始MVP策略详细分析 ===")
    print(f"总交易次数: {len(trades_df)}")
    
    # 基础统计
    total_pnl = trades_df['pnl'].sum()
    avg_pnl = trades_df['pnl'].mean()
    win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
    
    print(f"总收益: ${total_pnl:.2f}")
    print(f"平均每笔收益: ${avg_pnl:.2f}")
    print(f"胜率: {win_rate*100:.1f}%")
    
    # 按平仓原因分析
    exit_reasons = trades_df['exit_reason'].value_counts()
    print(f"\n平仓原因分布:")
    for reason, count in exit_reasons.items():
        pct = count / len(trades_df) * 100
        print(f"  {reason}: {count}次 ({pct:.1f}%)")
    
    # 按仓位方向分析
    long_trades = trades_df[trades_df['position'] == 1]
    short_trades = trades_df[trades_df['position'] == -1]
    
    print(f"\n多空分析:")
    print(f"多头交易: {len(long_trades)}次, 收益: ${long_trades['pnl'].sum():.2f}, 胜率: {len(long_trades[long_trades['pnl'] > 0])/len(long_trades)*100:.1f}%")
    print(f"空头交易: {len(short_trades)}次, 收益: ${short_trades['pnl'].sum():.2f}, 胜率: {len(short_trades[short_trades['pnl'] > 0])/len(short_trades)*100:.1f}%")
    
    # 最大单笔收益和损失
    max_profit = trades_df['pnl'].max()
    max_loss = trades_df['pnl'].min()
    print(f"\n最大单笔收益: ${max_profit:.2f}")
    print(f"最大单笔损失: ${max_loss:.2f}")
    
    # 计算日化收益率
    first_trade = trades_df['entry_time'].min()
    last_trade = trades_df['exit_time'].max()
    days = (last_trade - first_trade).days
    if days > 0:
        daily_return = total_pnl / 50000 / days * 100
        print(f"\n回测期间: {days}天")
        print(f"日化收益率: {daily_return:.3f}%")
    
    # 计算最大回撤
    cumulative_pnl = trades_df['pnl'].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = (cumulative_pnl - running_max) / 50000 * 100
    max_drawdown = drawdown.min()
    print(f"最大回撤: {max_drawdown:.2f}%")
    
    # 收益分布
    print(f"\n收益分布:")
    profit_trades = trades_df[trades_df['pnl'] > 0]
    loss_trades = trades_df[trades_df['pnl'] < 0]
    
    if len(profit_trades) > 0:
        avg_profit = profit_trades['pnl'].mean()
        print(f"盈利交易平均收益: ${avg_profit:.2f}")
    
    if len(loss_trades) > 0:
        avg_loss = loss_trades['pnl'].mean()
        print(f"亏损交易平均损失: ${avg_loss:.2f}")
    
    # 连续盈亏分析
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    
    for pnl in trades_df['pnl']:
        if pnl > 0:
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        else:
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
    
    print(f"\n连续盈亏分析:")
    print(f"最长连续盈利: {max_consecutive_wins}次")
    print(f"最长连续亏损: {max_consecutive_losses}次")
    
    # 按时间段分析
    trades_df['hour'] = trades_df['entry_time'].dt.hour
    hourly_performance = trades_df.groupby('hour')['pnl'].agg(['count', 'sum', 'mean'])
    print(f"\n按小时分析 (前5个最佳时段):")
    hourly_performance = hourly_performance.sort_values('sum', ascending=False)
    for hour, row in hourly_performance.head().iterrows():
        print(f"  {hour:02d}:00 - 交易{row['count']}次, 总收益${row['sum']:.2f}, 平均${row['mean']:.2f}")
    
    # 保存详细分析结果
    analysis_results = {
        'total_trades': len(trades_df),
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'max_drawdown': max_drawdown,
        'daily_return': daily_return if days > 0 else 0
    }
    
    return analysis_results

def analyze_optimized_mvp_results():
    """分析优化MVP策略回测结果"""
    
    try:
        # 读取优化版本交易记录
        trades_df = pd.read_csv('rexking_mvp_optimized_trades.csv')
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        
        print("\n" + "="*50)
        print("=== 优化MVP策略详细分析 ===")
        print(f"总交易次数: {len(trades_df)}")
        
        # 基础统计
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
        
        print(f"总收益: ${total_pnl:.2f}")
        print(f"平均每笔收益: ${avg_pnl:.2f}")
        print(f"胜率: {win_rate*100:.1f}%")
        
        # 按平仓原因分析
        exit_reasons = trades_df['exit_reason'].value_counts()
        print(f"\n平仓原因分布:")
        for reason, count in exit_reasons.items():
            pct = count / len(trades_df) * 100
            print(f"  {reason}: {count}次 ({pct:.1f}%)")
        
        # 按仓位方向分析
        long_trades = trades_df[trades_df['position'] == 1]
        short_trades = trades_df[trades_df['position'] == -1]
        
        print(f"\n多空分析:")
        print(f"多头交易: {len(long_trades)}次, 收益: ${long_trades['pnl'].sum():.2f}, 胜率: {len(long_trades[long_trades['pnl'] > 0])/len(long_trades)*100:.1f}%")
        print(f"空头交易: {len(short_trades)}次, 收益: ${short_trades['pnl'].sum():.2f}, 胜率: {len(short_trades[short_trades['pnl'] > 0])/len(short_trades)*100:.1f}%")
        
        # 最大单笔收益和损失
        max_profit = trades_df['pnl'].max()
        max_loss = trades_df['pnl'].min()
        print(f"\n最大单笔收益: ${max_profit:.2f}")
        print(f"最大单笔损失: ${max_loss:.2f}")
        
        # 计算日化收益率
        first_trade = trades_df['entry_time'].min()
        last_trade = trades_df['exit_time'].max()
        days = (last_trade - first_trade).days
        if days > 0:
            daily_return = total_pnl / 50000 / days * 100
            print(f"\n回测期间: {days}天")
            print(f"日化收益率: {daily_return:.3f}%")
        
        # 计算最大回撤
        cumulative_pnl = trades_df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / 50000 * 100
        max_drawdown = drawdown.min()
        print(f"最大回撤: {max_drawdown:.2f}%")
        
        # 收益分布
        print(f"\n收益分布:")
        profit_trades = trades_df[trades_df['pnl'] > 0]
        loss_trades = trades_df[trades_df['pnl'] < 0]
        
        if len(profit_trades) > 0:
            avg_profit = profit_trades['pnl'].mean()
            print(f"盈利交易平均收益: ${avg_profit:.2f}")
        
        if len(loss_trades) > 0:
            avg_loss = loss_trades['pnl'].mean()
            print(f"亏损交易平均损失: ${avg_loss:.2f}")
        
        # 连续盈亏分析
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for pnl in trades_df['pnl']:
            if pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        print(f"\n连续盈亏分析:")
        print(f"最长连续盈利: {max_consecutive_wins}次")
        print(f"最长连续亏损: {max_consecutive_losses}次")
        
        # 按时间段分析
        trades_df['hour'] = trades_df['entry_time'].dt.hour
        hourly_performance = trades_df.groupby('hour')['pnl'].agg(['count', 'sum', 'mean'])
        print(f"\n按小时分析 (前5个最佳时段):")
        hourly_performance = hourly_performance.sort_values('sum', ascending=False)
        for hour, row in hourly_performance.head().iterrows():
            print(f"  {hour:02d}:00 - 交易{row['count']}次, 总收益${row['sum']:.2f}, 平均${row['mean']:.2f}")
        
        # 保存详细分析结果
        analysis_results = {
            'total_trades': len(trades_df),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'max_drawdown': max_drawdown,
            'daily_return': daily_return if days > 0 else 0
        }
        
        return analysis_results
        
    except FileNotFoundError:
        print("优化版本交易记录文件未找到")
        return None

def compare_strategies():
    """比较两个策略的表现"""
    print("\n" + "="*50)
    print("=== 策略对比 ===")
    
    # 读取两个策略的结果
    try:
        original_df = pd.read_csv('rexking_mvp_trades.csv')
        optimized_df = pd.read_csv('rexking_mvp_optimized_trades.csv')
        
        original_pnl = original_df['pnl'].sum()
        optimized_pnl = optimized_df['pnl'].sum()
        
        original_trades = len(original_df)
        optimized_trades = len(optimized_df)
        
        original_win_rate = len(original_df[original_df['pnl'] > 0]) / original_trades
        optimized_win_rate = len(optimized_df[optimized_df['pnl'] > 0]) / optimized_trades
        
        original_avg_pnl = original_df['pnl'].mean()
        optimized_avg_pnl = optimized_df['pnl'].mean()
        
        print(f"原始MVP策略:")
        print(f"  总收益: ${original_pnl:.2f}")
        print(f"  交易次数: {original_trades}")
        print(f"  胜率: {original_win_rate*100:.1f}%")
        print(f"  平均收益: ${original_avg_pnl:.2f}")
        
        print(f"\n优化MVP策略:")
        print(f"  总收益: ${optimized_pnl:.2f}")
        print(f"  交易次数: {optimized_trades}")
        print(f"  胜率: {optimized_win_rate*100:.1f}%")
        print(f"  平均收益: ${optimized_avg_pnl:.2f}")
        
        print(f"\n改进效果:")
        print(f"  收益提升: ${optimized_pnl - original_pnl:.2f} ({((optimized_pnl/original_pnl-1)*100):.1f}%)")
        print(f"  交易次数减少: {original_trades - optimized_trades}次 ({((1-optimized_trades/original_trades)*100):.1f}%)")
        print(f"  胜率变化: {(optimized_win_rate - original_win_rate)*100:.1f}%")
        print(f"  平均收益提升: ${optimized_avg_pnl - original_avg_pnl:.2f} ({((optimized_avg_pnl/original_avg_pnl-1)*100):.1f}%)")
        
    except FileNotFoundError as e:
        print(f"无法比较策略: {e}")

if __name__ == "__main__":
    results1 = analyze_mvp_results()
    results2 = analyze_optimized_mvp_results()
    compare_strategies() 