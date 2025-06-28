"""
rexking_multi_factor_strategy.py
--------------------------------
RexKing 多因子策略 - 基于6指标、动态活力、市场状态切换
• 数据: 币安 K 线 CSV, 第一列为微秒时间戳（16 位）
• 频率: 15-minute
• 目标: 验证 2025-04 / 2025-05
• 核心: 6指标融合 + 动态活力 + 市场状态 + 简单规则
"""

import pandas as pd
import numpy as np
import talib as ta
import warnings
warnings.filterwarnings('ignore')

# === 0. 全局配置 ===
CSV_PATH = '/Users/qiutianyu/ETHUSDT-5m-2025-05/ETHUSDT-5m-2025-05.csv'
FEE_RATE = 0.002          # 单边手续费
INIT_CAP = 1_000.0
ATR_LEN = 14
VWAP_LEN = 20
RSI_LEN = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9

# === 1. 数据读取与预处理 ===
def load_data(csv_path):
    """加载并预处理数据"""
    col = ['ts','open','high','low','close','vol','end','qvol','trades',
           'tbvol','tbqvol','ignore']
    df = pd.read_csv(csv_path, names=col)
    df['ts'] = pd.to_datetime(df['ts'], unit='us')
    df.set_index('ts', inplace=True)
    return df

def calculate_indicators(df):
    """计算6个核心指标"""
    # T1: RSI (15%)
    df['rsi'] = ta.RSI(df['close'], timeperiod=RSI_LEN)
    
    # T8: VWAP (12%)
    df['vwap'] = ta.SMA(df['close'] * df['vol'], timeperiod=VWAP_LEN) / ta.SMA(df['vol'], timeperiod=VWAP_LEN)
    df['vwap_signal'] = (df['close'] - df['vwap']) / df['vwap']
    
    # W1: 净流出模拟 (15%) - 使用价格动量作为代理
    df['net_flow'] = df['close'].pct_change(4)  # 1小时动量
    df['flow_signal'] = np.where(df['net_flow'] > 0, 1, -1)
    
    # F2: ETF模拟 (10%) - 使用成交量变化作为代理
    df['etf_signal'] = df['vol'].pct_change(8)  # 2小时成交量变化
    
    # S1: 情绪模拟 (12%) - 使用价格波动率作为代理
    df['volatility'] = df['close'].rolling(12).std() / df['close']
    df['sentiment'] = 1 / (1 + df['volatility'] * 100)  # 波动率越低，情绪越好
    
    # T2: MACD (8%)
    df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(
        df['close'], fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL
    )
    df['macd_signal_norm'] = df['macd_hist'] / df['close']
    
    # 其他技术指标
    df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=ATR_LEN)
    df['ema_fast'] = ta.EMA(df['close'], timeperiod=20)
    df['ema_slow'] = ta.EMA(df['close'], timeperiod=50)
    df['returns'] = df['close'].pct_change()
    
    return df

def calculate_vitality(df, trades, alpha=0.15):
    """计算动态活力"""
    if len(trades) < 2:
        return 50.0
    
    # 计算最近交易的盈亏率
    recent_trades = trades[-min(10, len(trades)):]
    pnl_rates = [t['pnl_rate'] for t in recent_trades]
    
    # 使用EMA计算活力
    vitality = 50.0
    for pnl_rate in pnl_rates:
        vitality = alpha * pnl_rate + (1 - alpha) * vitality
    
    return np.clip(vitality, 10, 100)

def detect_market_regime(df, window=96):  # 24小时 = 96个15分钟
    """检测市场状态 (牛市/震荡/熊市)"""
    if len(df) < window:
        return 1  # 默认震荡
    
    recent_returns = df['returns'].tail(window)
    
    # 计算波动率和趋势
    volatility = recent_returns.std()
    trend = recent_returns.mean()
    
    # 简单规则判断市场状态
    if trend > 0.001 and volatility < 0.02:  # 牛市
        return 0
    elif trend < -0.001 or volatility > 0.03:  # 熊市
        return 2
    else:  # 震荡
        return 1

def calculate_signal_score(row):
    """计算6指标综合得分"""
    # 计算各指标得分
    rsi_score = 1 if row['rsi'] < 30 else 0 if row['rsi'] > 70 else 0.5
    vwap_score = 1 if row['vwap_signal'] > 0 else 0
    flow_score = 1 if row['flow_signal'] > 0 else 0
    etf_score = 1 if row['etf_signal'] > 0 else 0
    sentiment_score = row['sentiment']
    macd_score = 1 if row['macd_signal_norm'] > 0 else 0
    
    # 加权得分
    weights = [0.15, 0.12, 0.15, 0.10, 0.12, 0.08]  # 对应6个指标权重
    scores = [rsi_score, vwap_score, flow_score, etf_score, sentiment_score, macd_score]
    total_score = sum(w * s for w, s in zip(weights, scores))
    
    return total_score

def calculate_position_size(capital, price, atr, vitality, regime):
    """计算仓位大小"""
    # 基础风险比例
    base_risk = 0.01  # 1%风险
    
    # 根据市场状态调整
    if regime == 0:  # 牛市
        risk_multiplier = 1.2
        tp_ratio = 3.0
    elif regime == 1:  # 震荡
        risk_multiplier = 1.0
        tp_ratio = 2.5
    else:  # 熊市
        risk_multiplier = 0.8
        tp_ratio = 2.0
    
    # 活力调整
    vitality_multiplier = vitality / 100
    
    # 计算仓位
    risk_amount = capital * base_risk * risk_multiplier * vitality_multiplier
    stop_loss = atr * 0.8  # 0.8%止损
    position_size = risk_amount / stop_loss
    
    return position_size, tp_ratio

def detect_black_swan(df, i):
    """检测黑天鹅事件"""
    if i < 5:
        return False
    
    # 检测大幅下跌
    recent_returns = df['returns'].iloc[i-5:i]
    max_drop = recent_returns.min()
    
    # 检测异常成交量
    recent_vol = df['vol'].iloc[i-5:i]
    vol_ratio = recent_vol.iloc[-1] / recent_vol.mean()
    
    # 黑天鹅条件
    if max_drop < -0.08 or vol_ratio > 3:  # 8%单日跌幅或3倍成交量
        return True
    
    return False

# === 2. 主回测函数 ===
def run_backtest(csv_path):
    """运行回测"""
    print("=== RexKing 多因子策略回测 ===")
    
    # 加载数据
    df = load_data(csv_path)
    df = calculate_indicators(df)
    df.dropna(inplace=True)
    
    # 初始化变量
    capital = INIT_CAP
    equity_curve = [capital]
    position_qty = 0.0
    entry_price = stop = tp = 0.0
    trades = []
    
    # 主循环
    for i in range(50, len(df)):  # 从第50个数据点开始，确保有足够历史数据
        current_time = df.index[i]
        row = df.iloc[i]
        price = row['close']
        atr = row['atr']
        
        # 更新活力
        vitality = calculate_vitality(df, trades)
        
        # 检测市场状态
        regime = detect_market_regime(df.iloc[:i+1])
        
        # 检测黑天鹅
        if detect_black_swan(df, i):
            print(f"黑天鹅事件检测到: {current_time}")
            if position_qty > 0:
                # 紧急平仓
                exit_price = price
                pnl = (exit_price - entry_price) * position_qty
                fees = FEE_RATE * position_qty * (entry_price + exit_price)
                capital += pnl - fees
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl - fees,
                    'pnl_rate': (pnl - fees) / capital * 100,
                    'position_size': position_qty,
                    'vitality': vitality,
                    'regime': regime,
                    'exit_reason': 'black_swan'
                })
                position_qty = 0.0
            continue
        
        # 有仓位：检查止盈止损
        if position_qty > 0:
            if price <= stop or price >= tp:
                exit_price = price
                pnl = (exit_price - entry_price) * position_qty
                fees = FEE_RATE * position_qty * (entry_price + exit_price)
                capital += pnl - fees
                
                exit_reason = 'stop_loss' if price <= stop else 'take_profit'
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl - fees,
                    'pnl_rate': (pnl - fees) / capital * 100,
                    'position_size': position_qty,
                    'vitality': vitality,
                    'regime': regime,
                    'exit_reason': exit_reason
                })
                position_qty = 0.0
        
        # 空仓：寻找入场机会
        else:
            # 计算综合得分
            signal_score = calculate_signal_score(row)
            
            # 入场条件
            score_threshold = 0.65 if regime == 0 else 0.70  # 牛市65%，其他70%
            
            if (signal_score > score_threshold and 
                vitality > 30 and
                row['ema_fast'] > row['ema_slow'] and
                row['rsi'] < 70 and  # RSI不过热
                row['vwap_signal'] > -0.01):  # 价格接近VWAP
                
                # 计算仓位
                position_size, tp_ratio = calculate_position_size(
                    capital, price, atr, vitality, regime
                )
                
                position_qty = position_size
                entry_price = price
                entry_time = current_time
                stop = price - atr * 0.8
                tp = price + atr * tp_ratio
                
                print(f"开仓: {current_time}, 价格: {price:.2f}, 活力: {vitality:.1f}, 状态: {regime}, 得分: {signal_score:.3f}")
        
        # 更新权益曲线
        current_equity = capital if position_qty == 0 else capital + (price - entry_price) * position_qty
        equity_curve.append(current_equity)
    
    return equity_curve, trades

# === 3. 结果分析 ===
def analyze_results(equity_curve, trades):
    """分析回测结果"""
    equity_curve = np.array(equity_curve)
    
    # 基础统计
    total_return = (equity_curve[-1] / INIT_CAP - 1) * 100
    max_drawdown = np.max(1 - equity_curve / np.maximum.accumulate(equity_curve)) * 100
    
    # 交易统计
    if trades:
        wins = [t for t in trades if t['pnl'] > 0]
        win_rate = len(wins) / len(trades) * 100
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in trades) else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # 按市场状态分析
        regime_stats = {}
        for regime in [0, 1, 2]:
            regime_trades = [t for t in trades if t['regime'] == regime]
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
    
    # 输出结果
    print(f"\n=== 回测结果 ===")
    print(f"总收益率: {total_return:.2f}%")
    print(f"最大回撤: {max_drawdown:.2f}%")
    print(f"总交易数: {len(trades)}")
    print(f"胜率: {win_rate:.1f}%")
    print(f"平均盈利: ${avg_win:.2f}")
    print(f"平均亏损: ${avg_loss:.2f}")
    print(f"盈亏比: {profit_factor:.2f}")
    
    if regime_stats:
        print(f"\n=== 市场状态分析 ===")
        for regime, stats in regime_stats.items():
            regime_name = ['牛市', '震荡', '熊市'][regime]
            print(f"{regime_name}: {stats['trades']}笔, 胜率{stats['win_rate']:.1f}%, 平均${stats['avg_pnl']:.2f}")
    
    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }

# === 4. 主执行 ===
if __name__ == "__main__":
    # 运行回测
    equity_curve, trades = run_backtest(CSV_PATH)
    
    # 分析结果
    results = analyze_results(equity_curve, trades)
    
    # 保存交易记录
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv('rexking_trades_2025_05.csv', index=False)
        print(f"\n交易记录已保存到: rexking_trades_2025_05.csv") 