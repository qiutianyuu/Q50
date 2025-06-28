"""
rexking_optimized_strategy.py
--------------------------------
极简动量反转策略（防过拟合版）
• 只用短周期动量反转+简单趋势过滤
• 只用ATR止损、固定止盈、每日最大亏损风控
• 不用动态阈值/分批/复杂权重
"""

import pandas as pd
import numpy as np
import talib as ta
import warnings
warnings.filterwarnings('ignore')

CSV_PATH = '/Users/qiutianyu/ETHUSDT-5m-2025-05/ETHUSDT-5m-2025-05.csv'
FEE_RATE = 0.002
INIT_CAP = 1_000.0

# 参数（少量且直观）
ATR_LEN = 5
MOM_WINDOW = 3
RSI_LEN = 5
EMA_FAST, EMA_SLOW = 5, 13
STOP_LOSS_ATR = 1.2
TAKE_PROFIT_ATR = 2.5
MAX_DAILY_LOSS = 0.02  # 每日最大亏损2%

# 只做8:00-20:00
TRADE_START = 8
TRADE_END = 20

# === 数据与指标 ===
def load_data(csv_path):
    col = ['ts','open','high','low','close','vol','end','qvol','trades','tbvol','tbqvol','ignore']
    df = pd.read_csv(csv_path, names=col)
    df['ts'] = pd.to_datetime(df['ts'], unit='us')
    df.set_index('ts', inplace=True)
    return df

def calc_indicators(df):
    df['rsi'] = ta.RSI(df['close'], timeperiod=RSI_LEN)
    df['ema_fast'] = ta.EMA(df['close'], timeperiod=EMA_FAST)
    df['ema_slow'] = ta.EMA(df['close'], timeperiod=EMA_SLOW)
    df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=ATR_LEN)
    df['mom'] = df['close'].pct_change(MOM_WINDOW)
    df['rsi_chg'] = df['rsi'].diff(2)
    return df

# === 极简信号 ===
def simple_signal(row):
    # 反转+趋势过滤
    long_signal = (row['mom'] < -0.008) and (row['rsi'] < 35) and (row['rsi_chg'] > 0) and (row['ema_fast'] > row['ema_slow'])
    return long_signal

# === 回测主函数 ===
def run_simple_backtest(csv_path):
    print("=== 极简动量反转策略回测 ===")
    df = load_data(csv_path)
    df = calc_indicators(df)
    df.dropna(inplace=True)
    
    capital = INIT_CAP
    equity_curve = [capital]
    position = 0.0
    entry_price = 0.0
    stop = tp = 0.0
    trades = []
    last_day = None
    daily_loss = 0.0
    open_new_trade_today = True
    day_start_equity = capital
    day_trades = 0
    day_signals = 0
    
    for i in range(20, len(df)):
        now = df.index[i]
        row = df.iloc[i]
        price = row['close']
        atr = row['atr']
        this_day = now.date()
        this_hour = now.hour
        # 新的一天
        if last_day is None or this_day != last_day:
            if last_day is not None:
                # 日终强平
                if position > 0:
                    pnl = (price - entry_price) * position
                    fees = FEE_RATE * position * (entry_price + price)
                    capital += pnl - fees
                    trades.append({'entry': entry_time, 'exit': now, 'entry_price': entry_price, 'exit_price': price, 'pnl': pnl-fees, 'reason': 'day_close'})
                    position = 0.0
                # 日统计
                day_return = (capital / day_start_equity - 1) * 100
                print(f"【{last_day}】日收益: {day_return:.2f}%, 信号: {day_signals}, 交易: {day_trades}")
            day_start_equity = capital
            daily_loss = 0.0
            open_new_trade_today = True
            last_day = this_day
            day_trades = 0
            day_signals = 0
        # 只做白天
        if not (TRADE_START <= this_hour < TRADE_END):
            if this_hour == TRADE_END and position > 0:
                pnl = (price - entry_price) * position
                fees = FEE_RATE * position * (entry_price + price)
                capital += pnl - fees
                trades.append({'entry': entry_time, 'exit': now, 'entry_price': entry_price, 'exit_price': price, 'pnl': pnl-fees, 'reason': 'night_close'})
                position = 0.0
            continue
        # 有仓位，止盈止损
        if position > 0:
            if price <= stop or price >= tp:
                pnl = (price - entry_price) * position
                fees = FEE_RATE * position * (entry_price + price)
                capital += pnl - fees
                trades.append({'entry': entry_time, 'exit': now, 'entry_price': entry_price, 'exit_price': price, 'pnl': pnl-fees, 'reason': 'stop/tp'})
                daily_loss += min(0, pnl-fees)
                day_trades += 1
                position = 0.0
        # 空仓，找信号
        else:
            if not open_new_trade_today:
                continue
            if simple_signal(row):
                day_signals += 1
                # 固定仓位（不过拟合）
                qty = capital * 0.02 / price  # 每次2%仓
                position = qty
                entry_price = price
                entry_time = now
                stop = price - STOP_LOSS_ATR * atr
                tp = price + TAKE_PROFIT_ATR * atr
        # 更新权益
        equity_curve.append(capital if position==0 else capital + (price-entry_price)*position)
        # 日内最大亏损风控
        if daily_loss < -MAX_DAILY_LOSS * day_start_equity:
            open_new_trade_today = False
    return equity_curve, trades

# === 结果分析 ===
def analyze_simple_results(equity_curve, trades):
    equity_curve = np.array(equity_curve)
    total_return = (equity_curve[-1] / INIT_CAP - 1) * 100
    max_drawdown = np.max(1 - equity_curve / np.maximum.accumulate(equity_curve)) * 100
    win_trades = [t for t in trades if t['pnl'] > 0]
    win_rate = len(win_trades) / len(trades) * 100 if trades else 0
    print(f"\n=== 极简策略结果 ===")
    print(f"总收益率: {total_return:.2f}%")
    print(f"最大回撤: {max_drawdown:.2f}%")
    print(f"总交易数: {len(trades)}")
    print(f"胜率: {win_rate:.1f}%")
    if trades:
        print(f"平均盈利: ${np.mean([t['pnl'] for t in win_trades]):.2f}")
        print(f"平均亏损: ${np.mean([t['pnl'] for t in trades if t['pnl']<0]):.2f}")
    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'trades': len(trades),
        'win_rate': win_rate
    }

if __name__ == "__main__":
    equity_curve, trades = run_simple_backtest(CSV_PATH)
    results = analyze_simple_results(equity_curve, trades)
    if trades:
        pd.DataFrame(trades).to_csv('rexking_simple_trades_2025_05.csv', index=False)
        print("交易明细已保存。") 