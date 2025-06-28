import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RexKingETH81Strategy:
    def __init__(self, initial_capital=1000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.trades = []
        self.daily_pnl = []
        self.max_drawdown = 0
        self.peak_capital = initial_capital
        
        # 核心指标权重 - 基于因果分析
        self.weights = {
            'RSI': 0.40,      # 供需博弈核心指标
            'Volume': 0.30,   # 资金流向
            'Trend': 0.20,    # 趋势确认
            'BB': 0.10        # 布林带突破
        }
        
        # 优化参数 - 基于问题分析
        self.signal_strength_threshold = 0.06  # 从0.10降低到0.06，提升信号密度
        self.bull_market_volatility_threshold = 0.005  # 从0.008降低到0.005，降低牛市门槛
        self.adx_threshold = 25  # ADX趋势强度
        
        # 风险控制
        self.risk_per_trade = 0.004  # 每笔风险0.4%账户
        self.max_position_size = 0.08  # 最大仓位0.08 ETH
        self.min_position_size = 0.02  # 最小仓位0.02 ETH
        
        # 交易成本 - 真实成本
        self.fee_rate = 0.001  # 0.1%手续费
        self.slippage_rate = 0.001  # 0.1%滑点
        
        # 分批止盈参数
        self.tp1_multiplier = 1.5  # 第一目标1.5×ATR
        self.partial_close_ratio = 0.6  # 60%仓位在第一目标平仓
        self.trailing_stop_multiplier = 1.0  # 移动止损1×ATR
        self.timeout_bars = 12  # 12根K线超时平仓
        
    def load_data(self, file_path):
        """加载数据，自动适配无表头的Binance标准K线格式"""
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path, header=None)
        # Binance标准K线顺序: open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base, taker_buy_quote, ignore
        df.columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ]
        # 处理微秒级时间戳
        df['timestamp'] = pd.to_datetime(df['open_time'] // 1000, unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        # 转换为float
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df['returns'] = df['close'].pct_change()
        df['high_low'] = df['high'] - df['low']
        df['close_open'] = df['close'] - df['open']
        return df
    
    def calculate_indicators(self, df):
        """计算技术指标 - 包含多时间框架"""
        print("Calculating indicators...")
        
        # 30分钟时间框架趋势过滤
        df_30m = df.set_index('timestamp').resample('30min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        }).reset_index()
        df_30m['ema20'] = df_30m['close'].ewm(span=20).mean()
        df_30m['ema50'] = df_30m['close'].ewm(span=50).mean()
        df_30m['trend_30m'] = df_30m['ema20'] > df_30m['ema50']
        
        # 合并30分钟趋势到5分钟数据
        df = df.merge(df_30m[['timestamp', 'trend_30m']], on='timestamp', how='left')
        df['trend_30m'] = df['trend_30m'].fillna(method='ffill')
        
        # RSI (14周期) - 核心供需指标
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume指标 - 资金流向
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_spike'] = df['volume'] > df['volume_ma'] * 1.5
        
        # 5分钟趋势指标
        df['ema5'] = df['close'].ewm(span=5).mean()
        df['ema10'] = df['close'].ewm(span=10).mean()
        df['trend'] = df['ema5'] > df['ema10']
        
        # 布林带指标
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_confirm'] = df['close'] < df['bb_lower']  # 布林带下轨突破
        
        # ADX趋势强度
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        
        # +DM, -DM
        df['plus_dm'] = np.where(
            (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
            np.maximum(df['high'] - df['high'].shift(1), 0),
            0
        )
        df['minus_dm'] = np.where(
            (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
            np.maximum(df['low'].shift(1) - df['low'], 0),
            0
        )
        
        # ADX计算
        df['plus_di'] = 100 * df['plus_dm'].rolling(14).mean() / df['atr']
        df['minus_di'] = 100 * df['minus_dm'].rolling(14).mean() / df['atr']
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['ADX'] = df['dx'].rolling(14).mean()
        
        # 市场状态判断 - 优化版本
        df['volatility'] = df['returns'].rolling(20).std()
        df['is_bull_market'] = (df['volatility'] > self.bull_market_volatility_threshold) & (df['trend_30m'])
        
        # 突破信号
        df['high_20'] = df['high'].rolling(20).max()
        df['breakout'] = df['close'] > df['high_20'].shift(1)
        
        return df
    
    def calculate_signal_strength(self, row):
        """计算信号强度 - 基于因果逻辑"""
        score = 0
        
        # RSI信号 (40%权重) - 动态阈值
        rsi_threshold = 45 if row['is_bull_market'] else 40
        rsi_score = 1.0 if row['RSI'] < rsi_threshold else 0
        score += rsi_score * self.weights['RSI']
        
        # Volume信号 (30%权重) - 资金流向确认
        volume_score = 1.0 if row['volume_spike'] else 0
        score += volume_score * self.weights['Volume']
        
        # 趋势信号 (20%权重) - 5分钟趋势
        trend_score = 1.0 if row['trend'] else 0
        score += trend_score * self.weights['Trend']
        
        # 布林带信号 (10%权重) - 超卖突破
        bb_score = 1.0 if row['bb_confirm'] else 0
        score += bb_score * self.weights['BB']
        
        return score
    
    def calculate_position_size(self, row, atr):
        """固定风险仓位计算"""
        risk_cap = self.risk_per_trade * self.capital  # 0.4%风险预算
        stop_loss = atr * 0.6  # 0.6×ATR止损
        
        # 计算仓位大小
        position_size = risk_cap / (stop_loss * row['close'])
        
        # 限制在合理范围内
        position_size = max(self.min_position_size, min(position_size, self.max_position_size))
        
        return position_size
    
    def should_pause_trading(self, row):
        """风险控制：暂停交易 - 黑天鹅事件"""
        # 模拟NetFlow和Sentiment数据
        netflow = np.random.normal(15000, 8000)
        sentiment = np.random.uniform(0.0005, 0.05)
        
        # 黑天鹅条件
        if netflow > 60000:  # 大额流出
            return True, netflow, sentiment
        if sentiment < 0.0005:  # 极度悲观
            return True, netflow, sentiment
        return False, netflow, sentiment
    
    def backtest(self, df):
        """回测策略 - 优化版本"""
        print("Starting backtest...")
        
        last_trade_time = None  # 同一K线限1单
        
        for i in range(50, len(df)):  # 从第50个数据点开始
            current_row = df.iloc[i]
            
            # 风险控制检查
            should_pause, netflow, sentiment = self.should_pause_trading(current_row)
            if should_pause:
                continue
            
            # 同一K线限1单
            if last_trade_time == current_row['timestamp']:
                continue
            
            # 计算信号强度
            signal_strength = self.calculate_signal_strength(current_row)
            
            # 多条件触发 - 三选二逻辑
            cond1 = current_row['volume_spike']
            cond2 = current_row['RSI'] < (45 if current_row['is_bull_market'] else 40)
            cond3 = current_row['trend']
            trigger_conditions = (cond1 + cond2 + cond3) >= 1  # 从>=2降低到>=1，放宽条件
            
            # 交易信号 - 优化条件
            if (signal_strength > self.signal_strength_threshold and 
                trigger_conditions):  # 移除30分钟趋势过滤，改为可选
                
                # 计算仓位和止损止盈
                atr = current_row['atr']
                position_size = self.calculate_position_size(current_row, atr)
                
                # 考虑滑点的入场价格
                entry_price = current_row['close'] * (1 + self.slippage_rate)
                
                # 分批止盈设置
                tp1_price = entry_price + self.tp1_multiplier * atr
                stop_loss = atr * 0.6
                trailing_sl_price = entry_price - stop_loss
                
                # 记录交易
                trade = {
                    'entry_time': current_row['timestamp'],
                    'entry_price': entry_price,
                    'position_size': position_size,
                    'tp1_price': tp1_price,
                    'trailing_sl_price': trailing_sl_price,
                    'bars_open': 0,
                    'partial_closed': False,
                    'max_price': entry_price,
                    'signal_strength': signal_strength,
                    'is_bull_market': current_row['is_bull_market'],
                    'netflow': netflow,
                    'sentiment': sentiment,
                    'atr': atr
                }
                
                self.positions.append(trade)
                last_trade_time = current_row['timestamp']
            
            # 检查现有仓位 - 分批止盈和移动止损
            for pos in self.positions[:]:
                pos['bars_open'] += 1
                current_price = current_row['close']
                
                # 更新最高价和移动止损
                pos['max_price'] = max(pos['max_price'], current_price)
                pos['trailing_sl_price'] = max(
                    pos['trailing_sl_price'], 
                    pos['max_price'] - self.trailing_stop_multiplier * pos['atr']
                )
                
                # 第一目标止盈
                if not pos['partial_closed'] and current_price >= pos['tp1_price']:
                    # 60%仓位在第一目标平仓
                    partial_size = pos['position_size'] * self.partial_close_ratio
                    gross_profit = (pos['tp1_price'] - pos['entry_price']) * partial_size
                    net_profit = gross_profit * (1 - self.fee_rate - self.slippage_rate)
                    self.capital += net_profit
                    
                    self.trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': current_row['timestamp'],
                        'entry_price': pos['entry_price'],
                        'exit_price': pos['tp1_price'],
                        'position_size': partial_size,
                        'pnl': net_profit,
                        'type': 'tp1',
                        'signal_strength': pos['signal_strength'],
                        'is_bull_market': pos['is_bull_market']
                    })
                    
                    # 更新剩余仓位
                    pos['position_size'] *= (1 - self.partial_close_ratio)
                    pos['partial_closed'] = True
                
                # 移动止损或超时平仓
                elif (current_price <= pos['trailing_sl_price'] or 
                      pos['bars_open'] >= self.timeout_bars):
                    
                    # 确定平仓价格
                    if current_price <= pos['trailing_sl_price']:
                        exit_price = pos['trailing_sl_price']
                        exit_type = 'trailing_sl'
                    else:
                        exit_price = current_price
                        exit_type = 'timeout'
                    
                    # 计算剩余仓位的盈亏
                    gross_pnl = (exit_price - pos['entry_price']) * pos['position_size']
                    net_pnl = gross_pnl * (1 - self.fee_rate - self.slippage_rate)
                    self.capital += net_pnl
                    
                    self.trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': current_row['timestamp'],
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'position_size': pos['position_size'],
                        'pnl': net_pnl,
                        'type': exit_type,
                        'signal_strength': pos['signal_strength'],
                        'is_bull_market': pos['is_bull_market']
                    })
                    
                    self.positions.remove(pos)
            
            # 更新最大回撤
            if self.capital > self.peak_capital:
                self.peak_capital = self.capital
            current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def analyze_results(self):
        """分析回测结果"""
        if not self.trades:
            print("No trades executed!")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        # 基础统计
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        
        profit_factor = avg_win * winning_trades / (avg_loss * losing_trades) if losing_trades > 0 else float('inf')
        
        # 年化收益率
        if len(trades_df) > 1:
            start_time = trades_df['entry_time'].min()
            end_time = trades_df['exit_time'].max()
            days = (end_time - start_time).days
            annual_return = (total_pnl / self.initial_capital) * (365 / days) if days > 0 else 0
        else:
            annual_return = 0
        
        # 信号密度
        signal_density = total_trades / max(1, (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days)
        
        print("\n" + "="*60)
        print("RexKing ETH 8.1 Strategy Results")
        print("="*60)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${self.capital:,.2f}")
        print(f"Total PnL: ${total_pnl:,.2f}")
        print(f"Total Return: {(self.capital/self.initial_capital - 1)*100:.2f}%")
        print(f"Annualized Return: {annual_return*100:.2f}%")
        print(f"Max Drawdown: {self.max_drawdown*100:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate*100:.1f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Signal Density: {signal_density:.1f} trades/day")
        
        # 市场状态分析
        bull_trades = trades_df[trades_df['is_bull_market'] == True]
        bear_trades = trades_df[trades_df['is_bull_market'] == False]
        
        if len(bull_trades) > 0:
            bull_win_rate = len(bull_trades[bull_trades['pnl'] > 0]) / len(bull_trades)
            bull_pnl = bull_trades['pnl'].sum()
            print(f"\nBull Market Trades: {len(bull_trades)}")
            print(f"Bull Market Win Rate: {bull_win_rate*100:.1f}%")
            print(f"Bull Market PnL: ${bull_pnl:.2f}")
        
        if len(bear_trades) > 0:
            bear_win_rate = len(bear_trades[bear_trades['pnl'] > 0]) / len(bear_trades)
            bear_pnl = bear_trades['pnl'].sum()
            print(f"Bear Market Trades: {len(bear_trades)}")
            print(f"Bear Market Win Rate: {bear_win_rate*100:.1f}%")
            print(f"Bear Market PnL: ${bear_pnl:.2f}")
        
        # 信号强度分析
        strong_signals = trades_df[trades_df['signal_strength'] > 0.6]
        weak_signals = trades_df[trades_df['signal_strength'] <= 0.6]
        
        if len(strong_signals) > 0:
            strong_win_rate = len(strong_signals[strong_signals['pnl'] > 0]) / len(strong_signals)
            print(f"\nStrong Signals (>0.6): {len(strong_signals)}")
            print(f"Strong Signal Win Rate: {strong_win_rate*100:.1f}%")
        
        if len(weak_signals) > 0:
            weak_win_rate = len(weak_signals[weak_signals['pnl'] > 0]) / len(weak_signals)
            print(f"Weak Signals (≤0.6): {len(weak_signals)}")
            print(f"Weak Signal Win Rate: {weak_win_rate*100:.1f}%")
        
        # 交易类型分析
        tp1_trades = trades_df[trades_df['type'] == 'tp1']
        trailing_sl_trades = trades_df[trades_df['type'] == 'trailing_sl']
        timeout_trades = trades_df[trades_df['type'] == 'timeout']
        
        if len(tp1_trades) > 0:
            print(f"\nTP1 Trades: {len(tp1_trades)}")
            print(f"TP1 Win Rate: {len(tp1_trades[tp1_trades['pnl'] > 0]) / len(tp1_trades)*100:.1f}%")
            print(f"TP1 Avg PnL: ${tp1_trades['pnl'].mean():.2f}")
        
        if len(trailing_sl_trades) > 0:
            print(f"Trailing SL Trades: {len(trailing_sl_trades)}")
            print(f"Trailing SL Win Rate: {len(trailing_sl_trades[trailing_sl_trades['pnl'] > 0]) / len(trailing_sl_trades)*100:.1f}%")
            print(f"Trailing SL Avg PnL: ${trailing_sl_trades['pnl'].mean():.2f}")
        
        if len(timeout_trades) > 0:
            print(f"Timeout Trades: {len(timeout_trades)}")
            print(f"Timeout Win Rate: {len(timeout_trades[timeout_trades['pnl'] > 0]) / len(timeout_trades)*100:.1f}%")
            print(f"Timeout Avg PnL: ${timeout_trades['pnl'].mean():.2f}")
        
        return trades_df

def main():
    # 初始化策略
    strategy = RexKingETH81Strategy(initial_capital=1000)
    
    # 加载数据
    df = strategy.load_data('/Users/qiutianyu/ETHUSDT-5m-2025-05/ETHUSDT-5m-2025-05.csv')
    
    # 计算指标
    df = strategy.calculate_indicators(df)
    
    # 回测
    strategy.backtest(df)
    
    # 分析结果
    trades_df = strategy.analyze_results()
    
    # 保存交易记录
    if trades_df is not None and len(trades_df) > 0:
        trades_df.to_csv('rexking_eth_8_1_trades_2025_05.csv', index=False)
        print(f"\nTrades saved to rexking_eth_8_1_trades_2025_05.csv")

if __name__ == "__main__":
    main()
