"""
RexKing ETH 2.0策略
--------------------------------
• 4个核心指标：RSI(15%)、MACD(10%)、净流出(15%)、X情绪(12%)
• 多周期联合：5分钟抓短波，15分钟确认趋势
• 动态活力：EMA盈亏%控制仓位
• 风险控制：Etherscan监控+X NLP黑天鹅检测
• 目标：胜率85%，日30笔，日化0.7-0.8%
"""

import pandas as pd
import numpy as np
import talib as ta
import warnings
warnings.filterwarnings('ignore')

CSV_PATH = '/Users/qiutianyu/ETHUSDT-5m-2025-05/ETHUSDT-5m-2025-05.csv'
FEE_RATE = 0.002
INIT_CAP = 1_000.0

# 核心参数
RSI_LEN = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MOM_WINDOW = 3
VITALITY_ALPHA_BULL = 0.2  # 牛市
VITALITY_ALPHA_SIDEWAYS = 0.4  # 震荡
MIN_VITALITY = 50
MAX_VITALITY = 100

# 交易参数
BASE_POSITION = 0.1  # 0.1 ETH
TP_RATIO_BULL = 2.5  # 牛市止盈2.5-3.5%
TP_RATIO_SIDEWAYS = 2.0  # 震荡止盈2-2.5%
SL_PERCENTILE = 40  # ATR 40th分位数止损
RR_RATIO = 3.0  # 盈亏比3:1

# 风险控制
MAX_NETFLOW = 5000  # Etherscan监控阈值
MIN_SENTIMENT = 0.03  # X情绪阈值
PAUSE_HOURS_BULL = 24  # 牛市暂停24h
PAUSE_HOURS_SIDEWAYS = 48  # 震荡暂停48h

# 交易时段
TRADE_START = 8
TRADE_END = 20

def load_data(csv_path):
    """加载数据"""
    col = ['ts','open','high','low','close','vol','end','qvol','trades','tbvol','tbqvol','ignore']
    df = pd.read_csv(csv_path, names=col)
    df['ts'] = pd.to_datetime(df['ts'], unit='us')
    df.set_index('ts', inplace=True)
    return df

def calculate_indicators(df):
    """计算4个核心指标"""
    # 1. RSI (15%)
    df['rsi'] = ta.RSI(df['close'], timeperiod=RSI_LEN)
    
    # 2. MACD (10%)
    df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(
        df['close'], fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL
    )
    df['macd_golden_cross'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    
    # 3. 净流出模拟 (15%) - 用成交量变化模拟
    df['net_flow'] = df['vol'].pct_change(5) * 1000  # 模拟净流出
    
    # 4. X情绪模拟 (12%) - 用价格波动率模拟
    df['volatility'] = df['close'].rolling(10).std() / df['close']
    df['sentiment'] = 1 / (1 + df['volatility'] * 50)  # 波动率低=情绪好
    
    # 动量指标
    df['mom'] = df['close'].pct_change(MOM_WINDOW)
    df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # 趋势指标
    df['ema_fast'] = ta.EMA(df['close'], timeperiod=5)
    df['ema_slow'] = ta.EMA(df['close'], timeperiod=13)
    df['trend'] = df['ema_fast'] > df['ema_slow']
    
    # 突破信号
    df['breakout'] = df['close'] > df['high'].shift(1)
    
    return df

def detect_market_regime(df, window=72):  # 6小时窗口
    """检测市场状态：牛市/震荡/熊市"""
    if len(df) < window:
        return 'sideways'
    
    recent_returns = df['close'].pct_change().tail(window)
    volatility = recent_returns.std()
    trend = recent_returns.mean()
    
    if trend > 0.0005 and volatility < 0.02:  # 牛市：趋势向上，波动率低
        return 'bull'
    elif trend < -0.0005 and volatility > 0.025:  # 熊市：趋势向下，波动率高
        return 'bear'
    else:  # 震荡
        return 'sideways'

def calculate_vitality(df, trades, regime):
    """计算动态活力"""
    if len(trades) < 5:
        return 75.0  # 初始活力
    
    recent_trades = trades[-min(20, len(trades)):]
    pnl_rates = [t['pnl_rate'] for t in recent_trades]
    
    alpha = VITALITY_ALPHA_BULL if regime == 'bull' else VITALITY_ALPHA_SIDEWAYS
    vitality = 75.0
    
    for pnl_rate in pnl_rates:
        vitality = alpha * pnl_rate + (1 - alpha) * vitality
    
    return np.clip(vitality, MIN_VITALITY, MAX_VITALITY)

def generate_signals(row, regime):
    """生成交易信号"""
    signals = []
    
    # 信号1：动量反转 (放宽条件)
    if (row['mom'] < -0.003 or row['breakout']) and row['rsi'] < 45 and row['trend']:
        signals.append(('momentum_reversal', 0.15))
    
    # 信号2：MACD金叉
    if row['macd_golden_cross'] and row['rsi'] > 30:
        signals.append(('macd_cross', 0.10))
    
    # 信号3：净流出机会
    if row['net_flow'] < -100 and row['rsi'] < 50:  # 净流出时寻找机会
        signals.append(('net_flow', 0.15))
    
    # 信号4：情绪反转
    if row['sentiment'] > 0.7 and row['mom'] < -0.002:  # 情绪好但价格下跌
        signals.append(('sentiment', 0.12))
    
    return signals

def calculate_position_size(capital, price, vitality, regime):
    """计算仓位大小"""
    base_size = BASE_POSITION
    vitality_multiplier = vitality / 100
    
    # 根据市场状态调整
    if regime == 'bull':
        size_multiplier = 1.2
    elif regime == 'bear':
        size_multiplier = 0.8
    else:  # sideways
        size_multiplier = 1.0
    
    position_size = base_size * vitality_multiplier * size_multiplier
    max_position_value = capital * 0.25  # 最大25%仓位
    
    return min(position_size, max_position_value / price)

def risk_check(row, regime):
    """风险检查"""
    # Etherscan监控
    if abs(row['net_flow']) > MAX_NETFLOW:
        return False, f"NetFlow风险: {row['net_flow']:.0f}"
    
    # X情绪监控
    if row['sentiment'] < MIN_SENTIMENT:
        return False, f"情绪风险: {row['sentiment']:.3f}"
    
    return True, "正常"

def run_rexking_eth2_backtest(csv_path):
    """运行RexKing ETH 2.0回测"""
    print("=== RexKing ETH 2.0策略回测 ===")
    
    # 加载数据
    df = load_data(csv_path)
    df = calculate_indicators(df)
    df.dropna(inplace=True)
    
    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")
    print(f"价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # 初始化
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
    pause_until = None
    
    for i in range(30, len(df)):
        now = df.index[i]
        row = df.iloc[i]
        price = row['close']
        atr = row['atr']
        this_day = now.date()
        this_hour = now.hour
        
        # 检查暂停状态
        if pause_until and now < pause_until:
            continue
        
        # 新的一天
        if last_day is None or this_day != last_day:
            if last_day is not None:
                # 日终强平
                if position > 0:
                    pnl = (price - entry_price) * position
                    fees = FEE_RATE * position * (entry_price + price)
                    capital += pnl - fees
                    trades.append({
                        'entry': entry_time, 'exit': now, 'entry_price': entry_price, 
                        'exit_price': price, 'pnl': pnl-fees, 'pnl_rate': (pnl-fees)/capital*100,
                        'reason': 'day_close', 'regime': regime
                    })
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
                trades.append({
                    'entry': entry_time, 'exit': now, 'entry_price': entry_price, 
                    'exit_price': price, 'pnl': pnl-fees, 'pnl_rate': (pnl-fees)/capital*100,
                    'reason': 'night_close', 'regime': regime
                })
                position = 0.0
            continue
        
        # 检测市场状态
        regime = detect_market_regime(df.iloc[:i+1])
        
        # 计算活力
        vitality = calculate_vitality(df, trades, regime)
        
        # 有仓位，止盈止损
        if position > 0:
            if price <= stop or price >= tp:
                pnl = (price - entry_price) * position
                fees = FEE_RATE * position * (entry_price + price)
                capital += pnl - fees
                exit_reason = 'stop_loss' if price <= stop else 'take_profit'
                trades.append({
                    'entry': entry_time, 'exit': now, 'entry_price': entry_price, 
                    'exit_price': price, 'pnl': pnl-fees, 'pnl_rate': (pnl-fees)/capital*100,
                    'reason': exit_reason, 'regime': regime
                })
                daily_loss += min(0, pnl-fees)
                day_trades += 1
                position = 0.0
        
        # 空仓，找信号
        else:
            if not open_new_trade_today:
                continue
            
            # 风险检查
            risk_ok, risk_msg = risk_check(row, regime)
            if not risk_ok:
                print(f"风险暂停: {risk_msg}")
                pause_hours = PAUSE_HOURS_BULL if regime == 'bull' else PAUSE_HOURS_SIDEWAYS
                pause_until = now + pd.Timedelta(hours=pause_hours)
                continue
            
            # 生成信号
            signals = generate_signals(row, regime)
            
            if signals and vitality > MIN_VITALITY:
                day_signals += 1
                
                # 计算综合信号强度
                total_score = sum(weight for _, weight in signals)
                
                if total_score > 0.3:  # 信号强度阈值
                    # 计算仓位
                    position = calculate_position_size(capital, price, vitality, regime)
                    entry_price = price
                    entry_time = now
                    
                    # 计算止损止盈
                    sl_atr = np.percentile(df['atr'].tail(100), SL_PERCENTILE)
                    tp_ratio = TP_RATIO_BULL if regime == 'bull' else TP_RATIO_SIDEWAYS
                    
                    stop = price - sl_atr
                    tp = price + sl_atr * tp_ratio
                    
                    print(f"开仓: {now}, 价格: {price:.2f}, 活力: {vitality:.1f}, 状态: {regime}, 信号: {[s[0] for s in signals]}, 强度: {total_score:.3f}")
        
        # 更新权益
        equity_curve.append(capital if position==0 else capital + (price-entry_price)*position)
        
        # 日内最大亏损风控
        if daily_loss < -0.02 * day_start_equity:  # 2%日内最大亏损
            open_new_trade_today = False
    
    return equity_curve, trades

def analyze_rexking_results(equity_curve, trades):
    """分析RexKing ETH 2.0结果"""
    equity_curve = np.array(equity_curve)
    total_return = (equity_curve[-1] / INIT_CAP - 1) * 100
    max_drawdown = np.max(1 - equity_curve / np.maximum.accumulate(equity_curve)) * 100
    
    if trades:
        win_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(win_trades) / len(trades) * 100
        avg_win = np.mean([t['pnl'] for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in trades) else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # 按市场状态分析
        regime_stats = {}
        for regime in ['bull', 'sideways', 'bear']:
            regime_trades = [t for t in trades if t.get('regime') == regime]
            if regime_trades:
                regime_wins = [t for t in regime_trades if t['pnl'] > 0]
                regime_stats[regime] = {
                    'trades': len(regime_trades),
                    'win_rate': len(regime_wins) / len(regime_trades) * 100,
                    'avg_pnl': np.mean([t['pnl'] for t in regime_trades])
                }
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0
        regime_stats = {}
    
    print(f"\n=== RexKing ETH 2.0结果 ===")
    print(f"总收益率: {total_return:.2f}%")
    print(f"最大回撤: {max_drawdown:.2f}%")
    print(f"总交易数: {len(trades)}")
    print(f"胜率: {win_rate:.1f}%")
    if trades:
        print(f"平均盈利: ${avg_win:.2f}")
        print(f"平均亏损: ${avg_loss:.2f}")
        print(f"盈亏比: {profit_factor:.2f}")
    
    if regime_stats:
        print(f"\n=== 各市场状态表现 ===")
        regime_names = {'bull': '牛市', 'sideways': '震荡', 'bear': '熊市'}
        for regime, stats in regime_stats.items():
            regime_name = regime_names.get(regime, regime)
            print(f"{regime_name}: {stats['trades']}笔, 胜率{stats['win_rate']:.1f}%, 平均${stats['avg_pnl']:.2f}")
    
    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }

if __name__ == "__main__":
    equity_curve, trades = run_rexking_eth2_backtest(CSV_PATH)
    results = analyze_rexking_results(equity_curve, trades)
    
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv('rexking_eth2_trades_2025_05.csv', index=False)
        print("\n交易明细已保存到: rexking_eth2_trades_2025_05.csv") 