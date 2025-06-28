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
    
    # 计算ADX指标（趋势强度）
    def calculate_adx(df, period=14):
        """计算ADX指标"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 计算+DM和-DM
        high_diff = high.diff()
        low_diff = low.diff()
        
        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        
        plus_dm[high_diff > low_diff.abs()] = high_diff[high_diff > low_diff.abs()]
        minus_dm[low_diff.abs() > high_diff] = low_diff.abs()[low_diff.abs() > high_diff]
        
        # 计算TR
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算平滑值
        tr_smooth = tr.rolling(period).mean()
        plus_dm_smooth = plus_dm.rolling(period).mean()
        minus_dm_smooth = minus_dm.rolling(period).mean()
        
        # 计算+DI和-DI
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # 计算DX和ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    df['adx'] = calculate_adx(df, 14)
    
    # 计算funding z-score
    df['funding_z'] = (df['funding'] - df['funding'].rolling(72).mean()) / df['funding'].rolling(72).std()
    
    return df


def generate_signals(df):
    # 4H主信号（优化条件：ADX>22，加强趋势过滤）
    cond_4h = (
        (df['obv'] > df['obv'].rolling(14).mean()) &
        (df['ema20'] > df['ema60']) &
        (df['atr'] > 0.0005 * df['close']) &  # ATR > 0.05%
        (df['bb'] > 0.5) &  # BB > 0.5%
        (df['close'] > df['ema20']) &  # 价格在EMA20之上，确保趋势
        (df['adx'] > 22)  # ADX > 22（大幅提升）
    )
    
    # Funding极值过滤（z-score < -1）
    cond_1h = (df['funding_z'] < -1.0)  # funding z-score < -1
    
    # 15m突破+量（优化条件：量比0.08，假突破确认）
    df['volume_4h_ma'] = df['volume'].rolling(20).mean()
    cond_15m = (
        df['breakout_15m'] & 
        (df['volmean_15m'].rolling(2).mean() > 0.08 * df['volume_4h_ma']) &  # 量比0.08
        (df['close'] > df['high_15m'] * 1.001)  # 假突破确认：突破0.1%
    )
    
    # W1信号（集成Dune+Etherscan数据，更严格条件）
    cond_w1 = (df['w1_value'] > 1000) & (df['w1_zscore'] > 0.5) & (df['w1_signal_rolling'] > 0)
    
    # 联合信号（暂时移除W1条件）
    df['signal'] = cond_4h & cond_1h & cond_15m  # 移除 & cond_w1
    
    # 统计各信号触发次数
    print(f"4H信号: {cond_4h.sum()}条")
    print(f"Funding信号: {cond_1h.sum()}条") 
    print(f"15m突破信号: {cond_15m.sum()}条")
    print(f"W1信号: {cond_w1.sum()}条")
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
            # 双阶段止盈逻辑
            tp_lvl1 = entry_price + 1.0 * atr  # 第一阶段：+1 ATR
            tp_lvl2 = entry_price + 2.5 * atr  # 第二阶段：+2.5 ATR
            stop_loss = entry_price - 0.7 * atr  # 固定止损：0.7 ATR
            
            # 触发保本：+1 ATR后SL抬到进场价
            if price >= tp_lvl1 and stop < entry_price:
                stop = entry_price
            
            # ADX转弱也平仓
            adx_weak = row['adx'] < 20
            
            # 止盈/止损/ADX转弱
            hit_tp2 = price >= tp_lvl2
            hit_sl = price <= stop_loss
            exit_flag = hit_tp2 or hit_sl or adx_weak
            
            if exit_flag:
                pnl = (price - entry_price) * position
                fee = FEE_RATE * position * (entry_price + price)
                capital += pnl - fee
                
                # 确定退出原因
                if hit_tp2:
                    reason = 'tp'
                elif hit_sl:
                    reason = 'sl'
                else:
                    reason = 'adx_weak'
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': row['timestamp'],
                    'entry_price': entry_price,
                    'exit_price': price,
                    'size': position,
                    'pnl': pnl - fee,
                    'reason': reason
                })
                
                position = 0.0
                entry_price = 0.0
                entry_time = None
        
        # 开仓逻辑
        elif row['signal'] and position == 0.0:
            # 计算仓位大小（Kelly公式简化版）
            win_rate = 0.48  # 目标胜率
            avg_win = 2.5 * atr  # 目标平均盈利
            avg_loss = 0.7 * atr  # 目标平均亏损
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0.1, min(0.9, kelly_fraction))  # 限制在10%-90%
            
            # 计算仓位金额
            position_value = capital * kelly_fraction * RISK_PCT
            position_value = max(MIN_POSITION, min(MAX_POSITION, position_value))
            
            position = position_value / price
            entry_price = price
            entry_time = row['timestamp']
            stop = entry_price - 0.7 * atr
    
    # 计算回测统计
    if trades:
        total_pnl = sum(t['pnl'] for t in trades)
        win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in trades) else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in trades) else 0
        
        print(f"\n=== 回测结果 ===")
        print(f"总交易数: {len(trades)}")
        print(f"胜率: {win_rate:.1f}%")
        print(f"总盈亏: ${total_pnl:.2f}")
        print(f"平均盈利: ${avg_win:.2f}")
        print(f"平均亏损: ${avg_loss:.2f}")
        print(f"最终资金: ${capital:.2f}")
        print(f"收益率: {(capital/INIT_CAP - 1)*100:.1f}%")
        
        if avg_loss != 0:
            rr_ratio = abs(avg_win / avg_loss)
            print(f"R:R比例: {rr_ratio:.2f}")
    
    return trades, capital


def print_trades(trades):
    """打印交易记录"""
    if not trades:
        print("无交易记录")
        return
    
    df_trades = pd.DataFrame(trades)
    df_trades.to_csv('rexking_eth_10_4_trades.csv', index=False)
    print(f"交易记录已保存到 rexking_eth_10_4_trades.csv")


def main():
    print("=== RexKing ETH 10.4 策略回测 ===")
    
    # 加载数据
    print("加载数据...")
    df = load_data(CSV_PATH)
    print(f"数据加载完成，共{len(df)}条记录")
    
    # 生成信号
    print("\n生成信号...")
    df = generate_signals(df)
    
    # 回测
    print("\n开始回测...")
    trades, final_capital = backtest(df)
    
    # 保存结果
    print_trades(trades)
    
    print("\n回测完成！")


if __name__ == "__main__":
    main() 
        if position == 0 and row['signal']:
            # 动态仓位计算：基于ATR的风险管理
            risk = max(MIN_RISK_PCT, min(RISK_PCT, 0.015))
            pos_size = capital * risk / (atr * 0.7 * price) if atr > 0 else 0  # 止损0.7×ATR
            
            # Kelly仓位优化
            if len(trades) >= 10:
                recent_trades = trades[-10:]
                win_rate = sum(1 for t in recent_trades if t['pnl'] > 0) / len(recent_trades)
                avg_win = np.mean([t['pnl'] for t in recent_trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in recent_trades) else 0
                avg_loss = abs(np.mean([t['pnl'] for t in recent_trades if t['pnl'] < 0])) if any(t['pnl'] < 0 for t in recent_trades) else 1
                
                if avg_loss > 0:
                    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                    kelly_fraction = max(0, min(0.25, kelly_fraction))  # 限制在25%
                    pos_size *= kelly_fraction
            
            # 限制仓位大小
            pos_size = max(MIN_POSITION / price, min(MAX_POSITION / price, pos_size))
            
            if pos_size > 0:
                position = pos_size
                entry_price = price
                entry_time = row['timestamp']
                # 设置初始止损
                stop = entry_price - 0.7 * atr
    
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
        
        # 计算最大回撤
        equity_curve = [INIT_CAP]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade['pnl'])
        
        max_drawdown = 0
        peak = equity_curve[0]
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        print(f"最大回撤: {max_drawdown:.2f}%")
        
        # 保存交易结果
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv('rexking_eth_10_4_trades.csv', index=False)
        print(f"交易结果已保存到: rexking_eth_10_4_trades.csv")

if __name__ == '__main__':
    main() 