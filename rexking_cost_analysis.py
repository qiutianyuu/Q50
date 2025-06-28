import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 交易成本配置
FEE_RATE = 0.0004  # Binance taker fee 0.04%
SLIPPAGE = 0.0002  # 0.02% 滑点
TOTAL_COST = FEE_RATE * 2 + SLIPPAGE  # 双向交易总成本

def load_data():
    """加载5分钟数据"""
    try:
        # 尝试加载5月数据
        df = pd.read_csv('/Users/qiutianyu/ETHUSDT-5m-2025-05/ETHUSDT-5m-2025-05.csv', header=None)
        print(f"✅ 成功加载数据: {len(df)} 行")
    except FileNotFoundError:
        print("❌ 找不到5月数据文件")
        return None
    
    # 设置列名 - 根据实际数据结构
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                  'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                  'taker_buy_quote', 'ignore']
    
    # 只保留需要的列
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    # 修正时间戳为毫秒
    df['timestamp'] = pd.to_datetime((df['timestamp'] // 1_000_000).astype(int), unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

def calculate_indicators(df):
    """计算技术指标"""
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Stochastic
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    
    # OBV
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # Dynamic Vitality
    df['vitality'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Momentum
    df['momentum'] = df['close'] / df['close'].shift(10) - 1
    
    # Volatility
    df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
    
    # Trend Strength
    df['trend_strength'] = abs(df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # 移动平均
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    
    # 预计算滚动均值以避免循环中的计算
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    df['vitality_ma'] = df['vitality'].rolling(window=20).mean()
    
    return df

def generate_signals(df):
    """生成交易信号"""
    signals = []
    position = None
    entry_price = None
    entry_time = None
    capital = 50000
    trades = []
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        
        # 跳过缺失值
        if pd.isna(row['rsi']) or pd.isna(row['macd']) or pd.isna(row['atr']):
            continue
            
        # 多空条件检查
        long_conditions = 0
        short_conditions = 0
        
        # 1. RSI条件
        if row['rsi'] < 30:
            long_conditions += 1
        elif row['rsi'] > 70:
            short_conditions += 1
            
        # 2. MACD条件
        if row['macd'] > row['macd_signal'] and row['macd_hist'] > 0:
            long_conditions += 1
        elif row['macd'] < row['macd_signal'] and row['macd_hist'] < 0:
            short_conditions += 1
            
        # 3. Bollinger Bands条件
        if row['close'] < row['bb_lower']:
            long_conditions += 1
        elif row['close'] > row['bb_upper']:
            short_conditions += 1
            
        # 4. Stochastic条件
        if row['stoch_k'] < 20:
            long_conditions += 1
        elif row['stoch_k'] > 80:
            short_conditions += 1
            
        # 5. 成交量条件
        if row['volume'] > row['vol_ma'] * 1.2:
            if long_conditions > short_conditions:
                long_conditions += 1
            elif short_conditions > long_conditions:
                short_conditions += 1
                
        # 6. Vitality条件
        if row['vitality'] > row['vitality_ma'] * 1.1:
            if long_conditions > short_conditions:
                long_conditions += 1
            elif short_conditions > long_conditions:
                short_conditions += 1
                
        # 7. 趋势强度条件
        if row['trend_strength'] > 0.05:
            if long_conditions > short_conditions:
                long_conditions += 1
            elif short_conditions > long_conditions:
                short_conditions += 1
        
        # 信号生成逻辑
        signal = None
        if long_conditions >= 4 and position != 'long':
            signal = 'long'
        elif short_conditions >= 4 and position != 'short':
            signal = 'short'
        elif position == 'long' and (short_conditions >= 3 or row['close'] < entry_price * 0.99):
            signal = 'close_long'
        elif position == 'short' and (long_conditions >= 3 or row['close'] > entry_price * 1.01):
            signal = 'close_short'
            
        # 执行交易
        if signal:
            if signal == 'long' and position is None:
                position = 'long'
                entry_price = row['close']
                entry_time = row['timestamp']
                print(f"🟢 开多: {row['timestamp']} @ {entry_price:.2f}")
                
            elif signal == 'short' and position is None:
                position = 'short'
                entry_price = row['close']
                entry_time = row['timestamp']
                print(f"🔴 开空: {row['timestamp']} @ {entry_price:.2f}")
                
            elif signal == 'close_long' and position == 'long':
                # 计算原始PnL
                raw_pnl = (row['close'] - entry_price) / entry_price
                # 扣除交易成本
                net_pnl = raw_pnl - TOTAL_COST
                capital *= (1 + net_pnl)
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': row['timestamp'],
                    'entry_price': entry_price,
                    'exit_price': row['close'],
                    'position': 'long',
                    'raw_pnl': raw_pnl,
                    'net_pnl': net_pnl,
                    'capital': capital,
                    'cost_paid': TOTAL_COST * capital
                })
                
                print(f"🟢 平多: {row['timestamp']} @ {row['close']:.2f} | PnL: {raw_pnl*100:.2f}% | 净PnL: {net_pnl*100:.2f}% | 资金: {capital:.2f}")
                
                position = None
                entry_price = None
                entry_time = None
                
            elif signal == 'close_short' and position == 'short':
                # 计算原始PnL
                raw_pnl = (entry_price - row['close']) / entry_price
                # 扣除交易成本
                net_pnl = raw_pnl - TOTAL_COST
                capital *= (1 + net_pnl)
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': row['timestamp'],
                    'entry_price': entry_price,
                    'exit_price': row['close'],
                    'position': 'short',
                    'raw_pnl': raw_pnl,
                    'net_pnl': net_pnl,
                    'capital': capital,
                    'cost_paid': TOTAL_COST * capital
                })
                
                print(f"🔴 平空: {row['timestamp']} @ {row['close']:.2f} | PnL: {raw_pnl*100:.2f}% | 净PnL: {net_pnl*100:.2f}% | 资金: {capital:.2f}")
                
                position = None
                entry_price = None
                entry_time = None
    
    return trades, capital

def analyze_results(trades, final_capital):
    """分析回测结果"""
    if not trades:
        print("❌ 没有交易记录")
        return
        
    df_trades = pd.DataFrame(trades)
    
    # 基础统计
    total_trades = len(trades)
    winning_trades = len(df_trades[df_trades['net_pnl'] > 0])
    losing_trades = len(df_trades[df_trades['net_pnl'] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 收益统计
    total_return = (final_capital - 50000) / 50000
    avg_win = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
    avg_loss = df_trades[df_trades['net_pnl'] < 0]['net_pnl'].mean() if losing_trades > 0 else 0
    
    # 风险统计
    returns = df_trades['net_pnl'].values
    volatility = np.std(returns) * np.sqrt(252 * 288)  # 年化波动率 (5分钟数据)
    sharpe_ratio = (np.mean(returns) * 252 * 288) / volatility if volatility > 0 else 0
    
    # 最大回撤
    capital_curve = [50000] + [trade['capital'] for trade in trades]
    peak = capital_curve[0]
    max_drawdown = 0
    for capital in capital_curve:
        if capital > peak:
            peak = capital
        drawdown = (peak - capital) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # 成本分析
    total_cost = df_trades['cost_paid'].sum()
    total_raw_return = df_trades['raw_pnl'].sum()
    total_net_return = df_trades['net_pnl'].sum()
    
    print("\n" + "="*60)
    print("📊 REXKING 策略回测结果 (含交易成本)")
    print("="*60)
    print(f"💰 初始资金: $50,000")
    print(f"💰 最终资金: ${final_capital:,.2f}")
    print(f"📈 总收益率: {total_return*100:.2f}%")
    print(f"📈 月化收益率: {total_return*100:.2f}%")
    print(f"🔄 总交易次数: {total_trades}")
    print(f"✅ 盈利交易: {winning_trades}")
    print(f"❌ 亏损交易: {losing_trades}")
    print(f"🎯 胜率: {win_rate*100:.1f}%")
    print(f"📊 平均盈利: {avg_win*100:.3f}%")
    print(f"📊 平均亏损: {avg_loss*100:.3f}%")
    print(f"⚡ 夏普比率: {sharpe_ratio:.2f}")
    print(f"📉 最大回撤: {max_drawdown*100:.2f}%")
    print(f"💸 总交易成本: ${total_cost:,.2f}")
    print(f"💸 成本占比: {total_cost/50000*100:.2f}%")
    print(f"📊 原始收益: {total_raw_return*100:.2f}%")
    print(f"📊 净收益: {total_net_return*100:.2f}%")
    print(f"💸 成本侵蚀: {(total_raw_return - total_net_return)*100:.2f}%")
    
    # 保存交易记录
    df_trades.to_csv('rexking_cost_analysis_trades.csv', index=False)
    print(f"\n💾 交易记录已保存到: rexking_cost_analysis_trades.csv")
    
    return df_trades

def main():
    print("🚀 开始 RexKing 策略回测 (含交易成本)")
    print(f"💸 交易成本设置: {TOTAL_COST*100:.3f}% (双向)")
    
    # 加载数据
    df = load_data()
    if df is None:
        return
    
    # 计算指标
    print("📊 计算技术指标...")
    df = calculate_indicators(df)
    
    # 生成信号并回测
    print("🎯 生成交易信号...")
    trades, final_capital = generate_signals(df)
    
    # 分析结果
    analyze_results(trades, final_capital)

if __name__ == "__main__":
    main() 