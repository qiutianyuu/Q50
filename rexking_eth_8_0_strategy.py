import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RexKingETH8Strategy:
    def __init__(self, initial_capital=1000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.trades = []
        self.daily_pnl = []
        self.max_drawdown = 0
        self.peak_capital = initial_capital
        
        # 核心指标权重 - 优化版本
        self.weights = {
            'RSI': 0.40,
            'Volume': 0.30, 
            'MACD': 0.20,
            'Trend': 0.10
        }
        
        # 动态参数 - 进一步降低信号强度阈值
        self.signal_strength_threshold = 0.03  # 从0.05降低到0.03
        self.bull_market_volatility_threshold = 0.01  # 从0.015降低到0.01
        
        # 交易成本
        self.fee_rate = 0.001  # 0.1%手续费
        self.slippage_rate = 0.0005  # 0.05%滑点
        
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
        """计算技术指标 - 优化版本"""
        print("Calculating indicators...")
        
        # RSI (14周期)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume指标
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_spike'] = df['volume'] > df['volume_ma'] * 1.5
        
        # 趋势指标 - 简化
        df['ema5'] = df['close'].ewm(span=5).mean()
        df['ema10'] = df['close'].ewm(span=10).mean()
        df['trend'] = df['ema5'] > df['ema10']
        
        # MACD指标
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_cross'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        
        # 突破信号
        df['high_20'] = df['high'].rolling(20).max()
        df['breakout'] = df['close'] > df['high_20'].shift(1)
        
        # 动量指标
        df['momentum'] = df['close'].pct_change(3)
        
        # ATR (动态止损)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['ATR'] = df['tr'].rolling(14).mean()
        
        # 市场状态判断 - 简化
        df['volatility'] = df['returns'].rolling(20).std()
        df['is_bull_market'] = (df['volatility'] > self.bull_market_volatility_threshold) | (df['macd'] > 0)
        
        return df
    
    def calculate_signal_strength(self, row):
        """计算信号强度 - 简化版本"""
        score = 0
        
        # RSI信号 (40%权重) - 动态阈值
        if row['is_bull_market']:
            rsi_score = 1.0 if row['RSI'] < 50 else 0.5 if row['RSI'] < 60 else 0
        else:
            rsi_score = 1.0 if row['RSI'] < 35 else 0.5 if row['RSI'] < 45 else 0
        score += rsi_score * self.weights['RSI']
        
        # Volume信号 (30%权重)
        volume_score = 1.0 if row['volume_spike'] else 0
        score += volume_score * self.weights['Volume']
        
        # 趋势信号 (10%权重)
        trend_score = 1.0 if row['trend'] else 0
        score += trend_score * self.weights['Trend']
        
        # MACD信号 (20%权重) - 作为加分项
        macd_score = 0.5 if row['macd_cross'] else 0
        score += macd_score * self.weights['MACD']
        
        return score
    
    def calculate_position_size(self, row, atr):
        """计算仓位大小 (半Kelly) - 优化版本"""
        # 基础仓位
        if row['is_bull_market']:
            base_position = 0.08
        else:
            base_position = 0.05
        
        # 半Kelly调整
        win_rate = 0.65  # 目标胜率
        avg_win = 0.045 if row['is_bull_market'] else 0.025  # 目标盈利
        avg_loss = 0.005  # 止损0.5%
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.5))  # 限制在0-50%
        
        position_size = base_position * kelly_fraction
        return max(0.02, min(position_size, 0.08))  # 限制在0.02-0.08 ETH
    
    def should_pause_trading(self, row):
        """风险控制：暂停交易"""
        # 模拟NetFlow和Sentiment数据
        netflow = np.random.normal(15000, 8000)
        sentiment = np.random.uniform(0.0005, 0.05)
        
        if netflow > 60000 or sentiment < 0.0005:
            return True, netflow, sentiment
        return False, netflow, sentiment
    
    def backtest(self, df):
        """回测策略 - 修正版本"""
        print("Starting backtest...")
        
        for i in range(50, len(df)):  # 从第50个数据点开始
            current_row = df.iloc[i]
            
            # 风险控制检查
            should_pause, netflow, sentiment = self.should_pause_trading(current_row)
            if should_pause:
                continue
            
            # 计算信号强度
            signal_strength = self.calculate_signal_strength(current_row)
            
            # 交易信号 - 简化条件
            if (signal_strength > self.signal_strength_threshold and 
                (current_row['volume_spike'] or current_row['breakout'])):
                
                # 计算仓位和止损止盈
                atr = current_row['ATR']
                position_size = self.calculate_position_size(current_row, atr)
                
                # 动态止损止盈
                if current_row['is_bull_market']:
                    take_profit = atr * 4.5
                    stop_loss = atr * 0.5  # 修正为0.5
                else:
                    take_profit = atr * 2.5
                    stop_loss = atr * 0.5  # 修正为0.5
                
                # 考虑滑点的入场价格
                entry_price = current_row['close'] * (1 + self.slippage_rate)
                take_profit_price = entry_price + take_profit
                stop_loss_price = entry_price - stop_loss
                
                # 记录交易
                trade = {
                    'entry_time': current_row['timestamp'],
                    'entry_price': entry_price,
                    'position_size': position_size,
                    'take_profit_price': take_profit_price,
                    'stop_loss_price': stop_loss_price,
                    'signal_strength': signal_strength,
                    'is_bull_market': current_row['is_bull_market'],
                    'netflow': netflow,
                    'sentiment': sentiment
                }
                
                self.positions.append(trade)
            
            # 检查现有仓位 - 修正止损逻辑
            for pos in self.positions[:]:
                # 使用收盘价判断止损止盈
                current_price = current_row['close']
                
                if current_price >= pos['take_profit_price']:
                    # 止盈 - 考虑交易成本
                    gross_profit = (pos['take_profit_price'] - pos['entry_price']) * pos['position_size']
                    net_profit = gross_profit * (1 - self.fee_rate - self.slippage_rate)
                    self.capital += net_profit
                    
                    self.trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': current_row['timestamp'],
                        'entry_price': pos['entry_price'],
                        'exit_price': pos['take_profit_price'],
                        'position_size': pos['position_size'],
                        'pnl': net_profit,
                        'type': 'take_profit',
                        'signal_strength': pos['signal_strength'],
                        'is_bull_market': pos['is_bull_market']
                    })
                    
                    self.positions.remove(pos)
                    
                elif current_price <= pos['stop_loss_price']:
                    # 止损 - 考虑交易成本
                    gross_loss = (pos['entry_price'] - pos['stop_loss_price']) * pos['position_size']
                    net_loss = gross_loss * (1 - self.fee_rate - self.slippage_rate)
                    self.capital -= net_loss  # 修正：减去损失而不是加上
                    
                    self.trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': current_row['timestamp'],
                        'entry_price': pos['entry_price'],
                        'exit_price': pos['stop_loss_price'],
                        'position_size': pos['position_size'],
                        'pnl': -net_loss,  # 修正：记录为负值
                        'type': 'stop_loss',
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
        print("RexKing ETH 8.0 Strategy Results")
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
        take_profit_trades = trades_df[trades_df['type'] == 'take_profit']
        stop_loss_trades = trades_df[trades_df['type'] == 'stop_loss']
        
        if len(take_profit_trades) > 0:
            print(f"\nTake Profit Trades: {len(take_profit_trades)}")
            print(f"Take Profit Win Rate: {len(take_profit_trades[take_profit_trades['pnl'] > 0]) / len(take_profit_trades)*100:.1f}%")
        
        if len(stop_loss_trades) > 0:
            print(f"Stop Loss Trades: {len(stop_loss_trades)}")
            print(f"Stop Loss Win Rate: {len(stop_loss_trades[stop_loss_trades['pnl'] > 0]) / len(stop_loss_trades)*100:.1f}%")
        
        return trades_df

def main():
    # 初始化策略
    strategy = RexKingETH8Strategy(initial_capital=1000)
    
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
        trades_df.to_csv('rexking_eth_8_0_trades_2025_05.csv', index=False)
        print(f"\nTrades saved to rexking_eth_8_0_trades_2025_05.csv")

if __name__ == "__main__":
    main() 