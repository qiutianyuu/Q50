import pandas as pd
import numpy as np

INIT_CAP = 700.0
FEE_RATE = 0.00075  # VIP0费率
RISK_PCT = 0.015  # 1.5%风控
MIN_RISK_PCT = 0.005
TP_MULT = 5.0
SL_MULT = 0.5  # 0.5倍ATR止损
TRAIL_MULT = 5.0
MIN_POSITION = 28  # 最小仓位$28
MAX_POSITION = 70  # 最大仓位$70

CSV_PATH = '/Users/qiutianyu/ETHUSDT-4h/merged_4h_2023_2025.csv'


def load_data(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    for col in ['open','high','low','close','volume','obv','ema20','ema60','atr','bb','funding','high_15m','volmean_15m','breakout_15m','volume_surge_15m','w1_value','w1_zscore','w1_signal','w1_signal_rolling']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def generate_signals(df):
    # 4H主信号（进一步优化条件）
    cond_4h = (
        (df['obv'] > df['obv'].rolling(14).mean()) &
        (df['ema20'] > df['ema60']) &
        (df['atr'] > 0.002 * df['close']) &  # ATR > 0.2%
        (df['bb'] > 1.5)  # BB > 1.5%
    )
    
    # 1H funding（放宽条件）
    cond_1h = (df['funding'] < 0.0001)  # funding < 0.01%
    
    # 15m突破+量（进一步优化条件）
    df['volume_4h_ma'] = df['volume'].rolling(20).mean()
    cond_15m = df['breakout_15m'] & (df['volmean_15m'] > 0.1 * df['volume_4h_ma'])  # 10%的4H均值
    
    # 联合信号
    df['signal'] = cond_4h & cond_1h & cond_15m
    
    # 统计各信号触发次数
    print(f"4H信号: {cond_4h.sum()}条")
    print(f"Funding信号: {cond_1h.sum()}条") 
    print(f"15m突破信号: {cond_15m.sum()}条")
    print(f"联合信号: {df['signal'].sum()}条")
    
    # 计算日化信号频率
    total_days = (df['timestamp'].max() - df['timestamp'].min()).days
    daily_signals = df['signal'].sum() / total_days * 365
    print(f"年化信号频率: {daily_signals:.2f}条/年")
    
    return df


def backtest(df):
    capital = INIT_CAP
    trades = []
    position = 0.0
    entry_price = 0.0
    entry_time = None
    stop = tp = trail = 0.0
    
    for i in range(30, len(df)):
        row = df.iloc[i]
        price = row['close']
        atr = row['atr']
        
        # 平仓逻辑
        if position > 0:
            # 止盈/止损/追踪止损
            hit_tp = price >= tp
            hit_sl = price <= stop
            hit_trail = price <= trail
            exit_flag = hit_tp or hit_sl or hit_trail
            if exit_flag:
                pnl = (price - entry_price) * position
                fee = FEE_RATE * position * (entry_price + price)
                capital += pnl - fee
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': row['timestamp'],
                    'entry_price': entry_price,
                    'exit_price': price,
                    'size': position,
                    'pnl': pnl - fee,
                    'reason': 'tp' if hit_tp else ('sl' if hit_sl else 'trail')
                })
                position = 0.0
        
        # 开仓逻辑
        if position == 0 and row['signal']:
            # 动态仓位计算：基于ATR的风险管理
            risk = max(MIN_RISK_PCT, min(RISK_PCT, 0.015))
            pos_size = capital * risk / (atr * SL_MULT * price) if atr > 0 else 0
            pos_size = min(pos_size, capital / price)
            
            # 仓位限制
            if pos_size * price < MIN_POSITION:
                pos_size = MIN_POSITION / price
            if pos_size * price > MAX_POSITION:
                pos_size = MAX_POSITION / price
                
            position = pos_size
            entry_price = price
            entry_time = row['timestamp']
            stop = price - SL_MULT * atr
            tp = price + TP_MULT * atr
            trail = price - TRAIL_MULT * atr
    
    # 收盘强平
    if position > 0:
        price = df.iloc[-1]['close']
        pnl = (price - entry_price) * position
        fee = FEE_RATE * position * (entry_price + price)
        capital += pnl - fee
        trades.append({
            'entry_time': entry_time,
            'exit_time': df.iloc[-1]['timestamp'],
            'entry_price': entry_price,
            'exit_price': price,
            'size': position,
            'pnl': pnl - fee,
            'reason': 'close'
        })
    
    return trades, capital


def print_trades(trades):
    if not trades:
        print("无交易记录")
        return
    print(f"{'entry_time':20s} {'exit_time':20s} {'entry':8s} {'exit':8s} {'size':8s} {'pnl':8s}")
    for t in trades:
        print(f"{str(t['entry_time']):20s} {str(t['exit_time']):20s} {t['entry_price']:<8.2f} {t['exit_price']:<8.2f} {t['size']:<8.4f} {t['pnl']:<8.2f}")


def main():
    df = load_data(CSV_PATH)
    df = generate_signals(df)
    trades, final_cap = backtest(df)
    print_trades(trades)
    print(f"\n初始资金: ${INIT_CAP:.2f}，最终资金: ${final_cap:.2f}，总交易: {len(trades)}")
    if trades:
        total_pnl = sum(t['pnl'] for t in trades)
        win_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(win_trades) / len(trades) * 100
        print(f"总盈亏: ${total_pnl:.2f}，胜率: {win_rate:.1f}%")
        
        # 计算年化收益率
        total_days = (df['timestamp'].max() - df['timestamp'].min()).days
        annual_return = (final_cap / INIT_CAP - 1) * 365 / total_days * 100
        print(f"年化收益率: {annual_return:.2f}%")
        
        # 保存交易结果
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv('rexking_eth_10_3_trades.csv', index=False)
        print(f"交易结果已保存到: rexking_eth_10_3_trades.csv")

if __name__ == '__main__':
    main() 