#!/usr/bin/env python3
"""
基于 websocket_signals.csv 的简易回测脚本
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def backtest_signals(signal_file, hold_period=1, fee_rate=0.0004):
    print(f"读取信号文件: {signal_file}")
    df = pd.read_csv(signal_file, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 只保留有信号的行
    signals = df[df['signal'] != 0].copy()
    print(f"信号总数: {len(signals)}")
    if len(signals) == 0:
        print("没有信号，无法回测")
        return
    
    # 计算每笔信号的开平仓价格
    signals['entry_price'] = signals['mid_price']
    signals['exit_price'] = signals['mid_price'].shift(-hold_period)
    signals['exit_time'] = signals['timestamp'].shift(-hold_period)
    
    # 移除没有平仓价格的信号
    signals = signals.dropna(subset=['exit_price'])
    
    # 计算收益
    signals['ret'] = (signals['exit_price'] - signals['entry_price']) / signals['entry_price'] * signals['signal']
    signals['ret_net'] = signals['ret'] - fee_rate * 2  # 开平各收一次手续费
    signals['cum_ret'] = (1 + signals['ret_net']).cumprod() - 1
    
    # 统计
    win_rate = (signals['ret_net'] > 0).mean()
    avg_ret = signals['ret_net'].mean()
    total_ret = signals['cum_ret'].iloc[-1]
    max_dd = (signals['cum_ret'].cummax() - signals['cum_ret']).max()
    print(f"胜率: {win_rate:.2%}")
    print(f"平均单笔收益: {avg_ret:.4%}")
    print(f"累计收益: {total_ret:.2%}")
    print(f"最大回撤: {max_dd:.2%}")
    print(f"信号区间: {signals['timestamp'].min()} ~ {signals['timestamp'].max()}")
    
    # 绘制收益曲线
    plt.figure(figsize=(10,4))
    plt.plot(signals['timestamp'], signals['cum_ret'], label='Cumulative Return')
    plt.title('Websocket Signal Backtest')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig('websocket_backtest_curve.png')
    print("收益曲线已保存为 websocket_backtest_curve.png")
    
    # 保存详细结果
    signals.to_csv('websocket_backtest_trades.csv', index=False)
    print("详细回测结果已保存为 websocket_backtest_trades.csv")

def main():
    backtest_signals('websocket_signals.csv', hold_period=1, fee_rate=0.0004)

if __name__ == "__main__":
    main() 