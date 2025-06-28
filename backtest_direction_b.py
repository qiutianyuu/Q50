import pandas as pd
import numpy as np
import talib as ta
from pathlib import Path
from datetime import datetime, timezone

CSV_PATH = "/Users/qiutianyu/ETHUSDT-5m-2025-05/ETHUSDT-5m-2025-05.csv"

# Grid Strategy Parameters
GRID_LEVELS = 5  # 网格层数
INITIAL_GRID_SIZE = 0.005  # 初始网格大小（0.5%）
GRID_SIZE_ATR_MULT = 0.5  # 网格大小与ATR的倍数
POSITION_SIZE = 0.2  # 每格仓位大小
MAX_POSITION = 1.0  # 最大仓位
TRADING_HOURS = set(range(8, 11))  # 交易时间
EQUITY_START = 1.0
MAX_DAILY_TRADES = 10  # 增加每日交易次数限制
MIN_VOLUME = 1000  # 最小成交量要求

def load_data(path: str) -> pd.DataFrame:
    cols = ['startTime','open','high','low','close','volume','closeTime',
            'qav','trades','taker_base','taker_quote','ignore']
    df = pd.read_csv(path, header=None, names=cols)
    df['startTime'] = pd.to_datetime(df['startTime'], unit='us', utc=True)
    df.set_index('startTime', inplace=True)
    return df

def add_indicators(df: pd.DataFrame):
    # 计算ATR用于动态调整网格大小
    df['atr'] = ta.ATR(df['high'], df['low'], df['close'], 14)
    
    # 计算移动平均线用于趋势判断
    df['sma_20'] = ta.SMA(df['close'], 20)
    df['sma_50'] = ta.SMA(df['close'], 50)
    
    # 计算成交量移动平均
    df['volume_sma'] = ta.SMA(df['volume'], 20)

def calculate_grid_levels(price: float, atr: float, is_uptrend: bool) -> list:
    """计算网格价格水平"""
    grid_size = max(INITIAL_GRID_SIZE, atr * GRID_SIZE_ATR_MULT / price)
    if is_uptrend:
        # 上升趋势：网格向上倾斜
        return [price * (1 + grid_size * i) for i in range(GRID_LEVELS)]
    else:
        # 下降趋势：网格向下倾斜
        return [price * (1 - grid_size * i) for i in range(GRID_LEVELS)]

def get_trend(df: pd.DataFrame, current_idx: int) -> bool:
    """判断当前趋势"""
    if current_idx < 50:
        return True  # 默认上升趋势
    
    # 使用均线系统判断趋势
    sma_20 = df['sma_20'].iloc[current_idx]
    sma_50 = df['sma_50'].iloc[current_idx]
    close = df['close'].iloc[current_idx]
    
    return close > sma_20 and sma_20 > sma_50

def backtest(df: pd.DataFrame):
    equity = EQUITY_START
    equity_curve = []
    trades = []
    positions = []  # 当前持仓
    daily_trades = 0
    last_trade_date = None
    grid_levels = None
    last_grid_update = None

    for i, (ts, row) in enumerate(df.iterrows()):
        # 更新每日交易计数
        current_date = ts.date()
        if last_trade_date != current_date:
            daily_trades = 0
            last_trade_date = current_date
        
        # 更新权益曲线
        if equity_curve and equity_curve[-1][0].hour != ts.hour:
            equity_curve.append((ts, equity))
        
        # 交易时间过滤
        if ts.hour not in TRADING_HOURS:
            continue
        
        # 成交量过滤
        if row['volume'] < MIN_VOLUME:
            continue
        
        # 更新网格
        if grid_levels is None or last_grid_update is None or \
           (ts - last_grid_update).total_seconds() > 3600:  # 每小时更新一次网格
            is_uptrend = get_trend(df, i)
            grid_levels = calculate_grid_levels(row['close'], row['atr'], is_uptrend)
            last_grid_update = ts
        
        # 检查是否需要开仓
        if daily_trades < MAX_DAILY_TRADES:
            current_position = sum(p['size'] for p in positions)
            
            # 在上升趋势中，价格回调到网格时做多
            if is_uptrend and current_position < MAX_POSITION:
                for level in grid_levels:
                    if row['low'] <= level <= row['high'] and \
                       not any(abs(p['entry_price'] - level) < 0.001 for p in positions):
                        size = POSITION_SIZE
                        positions.append({
                            'entry_time': ts,
                            'entry_price': level,
                            'size': size
                        })
                        trades.append({
                            'entry_time': ts,
                            'entry_price': level,
                            'size': size,
                            'position': 1
                        })
                        daily_trades += 1
                        break
            
            # 在下降趋势中，价格反弹到网格时做空
            elif not is_uptrend and current_position > -MAX_POSITION:
                for level in grid_levels:
                    if row['low'] <= level <= row['high'] and \
                       not any(abs(p['entry_price'] - level) < 0.001 for p in positions):
                        size = -POSITION_SIZE
                        positions.append({
                            'entry_time': ts,
                            'entry_price': level,
                            'size': size
                        })
                        trades.append({
                            'entry_time': ts,
                            'entry_price': level,
                            'size': size,
                            'position': -1
                        })
                        daily_trades += 1
                        break
        
        # 检查是否需要平仓
        for pos in positions[:]:  # 使用切片创建副本以避免修改迭代中的列表
            # 计算当前持仓的盈亏
            pnl = (row['close'] - pos['entry_price']) / pos['entry_price'] * pos['size']
            
            # 止盈条件：盈利超过网格大小的一半
            if pnl > 0 and pnl > INITIAL_GRID_SIZE * 0.5:
                equity *= (1 + pnl)
                positions.remove(pos)
                trades[-1].update({
                    'exit_time': ts,
                    'exit_price': row['close'],
                    'pnl': pnl
                })
            
            # 止损条件：亏损超过网格大小
            elif pnl < 0 and abs(pnl) > INITIAL_GRID_SIZE:
                equity *= (1 + pnl)
                positions.remove(pos)
                trades[-1].update({
                    'exit_time': ts,
                    'exit_price': row['close'],
                    'pnl': pnl
                })

    # 强制平仓所有剩余持仓
    for pos in positions:
        pnl = (df['close'].iloc[-1] - pos['entry_price']) / pos['entry_price'] * pos['size']
        equity *= (1 + pnl)
        trades[-1].update({
            'exit_time': df.index[-1],
            'exit_price': df['close'].iloc[-1],
            'pnl': pnl
        })

    return equity, trades, equity_curve

def main():
    df = load_data(CSV_PATH)
    add_indicators(df)
    equity, trades, eq_curve = backtest(df)
    
    # 计算统计数据
    wins = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]
    losses = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) <= 0]
    win_rate = len(wins) / len(trades) if trades else 0
    
    # 计算回撤
    eq_values = [e[1] for e in eq_curve]
    peak = np.maximum.accumulate(eq_values)
    drawdown = (peak - eq_values) / peak
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # 计算每日收益
    daily_returns = []
    current_date = None
    daily_equity = None
    
    for ts, eq in eq_curve:
        if current_date != ts.date():
            if daily_equity is not None:
                daily_returns.append((current_date, (eq - daily_equity) / daily_equity))
            current_date = ts.date()
            daily_equity = eq
    
    avg_daily_return = np.mean([r[1] for r in daily_returns]) if daily_returns else 0
    
    print(f"Trades: {len(trades)}, Win rate: {win_rate*100:.2f}%")
    print(f"Equity final: {equity:.4f}, Return: {(equity-1)*100:.2f}%")
    print(f"Max drawdown: {max_drawdown*100:.2f}%")
    print(f"Average daily return: {avg_daily_return*100:.2f}%")
    
    # 打印交易详情
    print("\nTrade details:")
    for i, t in enumerate(trades, 1):
        print(f"Trade {i}:")
        print(f"  Entry: {t['entry_time']} @ {t['entry_price']:.2f}")
        if 'exit_time' in t:
            print(f"  Exit: {t['exit_time']} @ {t['exit_price']:.2f}")
            print(f"  PnL: {t.get('pnl', 0)*100:.2f}%")
        print(f"  Position: {'Long' if t['position'] == 1 else 'Short'}")

if __name__ == '__main__':
    main() 