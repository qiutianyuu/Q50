import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RexKingETH94Strategy:
    def __init__(self, capital=10000):
        self.capital = capital
        self.positions = []
        self.trades = []
        
        # 信号质量提升 - 9.4优化
        self.signal_strength_threshold = 0.07  # 提升到0.07（Q60）
        self.rsi_bull_threshold = 50  # 牛市RSI<50
        self.rsi_range_threshold = 45  # 震荡市RSI<45
        self.volume_multiplier = 1.5  # Volume>1.5×均值
        self.atr_threshold = 0.002
        
        # 风险控制 - 降频+成本控制
        self.risk_per_trade = 0.002  # 0.2%风险（牛市0.3%）
        self.max_position_size = 0.1  # 最大10%仓位
        self.fee_rate = 0.00075  # 0.075%手续费（Binance VIP）
        self.slippage_rate = 0.00075  # 0.075%滑点
        self.total_cost_rate = 0.0015  # 总成本0.15%
        
        # 止盈止损优化 - 9.4修复
        self.tp1_multiplier = 1.5  # 第一目标1.5×ATR
        self.tp2_multiplier = 6.0  # 第二目标6×ATR（牛市）
        self.partial_close_ratio = 0.5  # 50%仓位在第一目标平仓
        self.trailing_stop_multiplier = 3.0  # 移动止损3×ATR（放宽）
        self.trailing_start_multiplier = 0.3  # 0.3×ATR开始移动止损
        self.timeout_bars = 16  # 16根K线超时（4小时）
        
        # 交易频率控制
        self.max_trades_per_hour = 1  # 每小时1笔
        self.max_trades_per_day = 5  # 每日最多5笔
        self.min_hold_bars = 16  # 最少持仓4小时
        self.trade_count = 0
        self.daily_trade_count = 0
        self.last_trade_hour = None
        self.last_trade_date = None
        
        # Etherscan参数
        self.w1_threshold = 8000  # W1>8000
        self.s1_threshold = 0.05  # S1>0.05
        
    def load_data(self, file_path):
        """加载数据"""
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path, header=None)
        df.columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ]
        df['timestamp'] = pd.to_datetime(df['open_time'] // 1000, unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        
        df['returns'] = df['close'].pct_change()
        df['high_low'] = df['high'] - df['low']
        df['close_open'] = df['close'] - df['open']
        
        return df
    
    def calculate_indicators(self, df):
        """计算技术指标 - 9.4增强信号质量"""
        print("Calculating enhanced indicators for 15-minute data...")
        
        # 30分钟时间框架 - 趋势确认
        df_30m = df.set_index('timestamp').resample('30min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        }).reset_index()
        df_30m['high_low'] = df_30m['high'] - df_30m['low']
        df_30m['ema20'] = df_30m['close'].ewm(span=20).mean()
        df_30m['ema50'] = df_30m['close'].ewm(span=50).mean()
        df_30m['trend_30m'] = df_30m['ema20'] > df_30m['ema50']
        
        # 合并30分钟数据
        df = df.merge(df_30m[['timestamp', 'trend_30m']], on='timestamp', how='left')
        df['trend_30m'] = df['trend_30m'].fillna(method='ffill')
        
        # RSI - 优化阈值
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume指标 - 提升阈值
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_spike'] = df['volume'] > df['volume_ma'] * self.volume_multiplier
        
        # 趋势指标
        df['ema5'] = df['close'].ewm(span=5).mean()
        df['ema10'] = df['close'].ewm(span=10).mean()
        df['trend'] = df['ema5'] > df['ema10']
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']
        
        # 布林带宽度 - 9.4新增
        df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
        df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        df['bb_volatile'] = df['bb_width'] > 0.01  # 震荡市过滤
        
        # 3阳线形态 - 9.4新增
        df['three_bull'] = (df['close'] > df['open']).rolling(3).sum() == 3
        
        # 市场状态检测
        df['market_state'] = 'range'
        df.loc[df['trend_30m'] & (df['RSI'] > 50), 'market_state'] = 'bull'
        df.loc[~df['trend_30m'] & (df['RSI'] < 45), 'market_state'] = 'bear'
        
        # Etherscan数据（Mock）- 9.4优化
        np.random.seed(42)
        df['w1_value'] = np.random.normal(5000, 2000, len(df))
        df['w1_zscore'] = (df['w1_value'] - df['w1_value'].rolling(100).mean()) / df['w1_value'].rolling(100).std()
        df['w1_signal'] = (df['w1_value'] > self.w1_threshold) & (df['w1_zscore'] > 2)
        
        df['s1_value'] = np.random.normal(0.03, 0.02, len(df))
        df['s1_signal'] = df['s1_value'] > self.s1_threshold
        
        return df
    
    def calculate_signal_strength(self, row):
        """计算信号强度 - 9.4增强"""
        score = 0
        
        # RSI信号
        if row['market_state'] == 'bull':
            if row['RSI'] < self.rsi_bull_threshold:
                score += 0.3
        else:  # range/bear
            if row['RSI'] < self.rsi_range_threshold:
                score += 0.3
        
        # Volume信号
        if row['volume_spike']:
            score += 0.2
        
        # 趋势信号
        if row['trend'] and row['trend_30m']:
            score += 0.2
        
        # 3阳线形态
        if row['three_bull']:
            score += 0.1
        
        # 布林带宽度（震荡市）
        if row['market_state'] == 'range' and row['bb_volatile']:
            score += 0.1
        
        # Etherscan信号
        if row['w1_signal'] and row['s1_signal']:
            score += 0.1
        
        return score
    
    def calculate_position_size(self, row, atr):
        """计算仓位大小 - 9.4优化"""
        risk_cap = self.capital * self.risk_per_trade
        
        # 根据市场状态调整风险
        if row['market_state'] == 'bull':
            risk_cap = self.capital * 0.003  # 牛市0.3%
        
        stop_loss = atr * 0.6 * row['close']  # 0.6×ATR止损
        position_size = risk_cap / stop_loss
        position_size = min(position_size, self.max_position_size)
        
        return position_size
    
    def should_pause_trading(self, row):
        """交易频率控制 - 9.4优化"""
        current_hour = row['timestamp'].hour
        current_date = row['timestamp'].date()
        
        # 重置每日计数
        if self.last_trade_date != current_date:
            self.daily_trade_count = 0
            self.last_trade_date = current_date
        
        # 每小时限制
        if self.last_trade_hour == current_hour and self.trade_count >= self.max_trades_per_hour:
            return True
        
        # 每日限制
        if self.daily_trade_count >= self.max_trades_per_day:
            return True
        
        return False
    
    def backtest(self, df):
        """回测 - 9.4优化"""
        print("Starting enhanced backtest...")
        
        for i, row in df.iterrows():
            # 更新持仓
            self.update_positions(row)
            
            # 频率控制
            if self.should_pause_trading(row):
                continue
            
            # 开仓条件
            if len(self.positions) == 0:
                signal_strength = self.calculate_signal_strength(row)
                
                if (signal_strength > self.signal_strength_threshold and 
                    row['atr_pct'] > self.atr_threshold):
                    
                    # 计算仓位
                    position_size = self.calculate_position_size(row, row['atr'])
                    
                    # 开仓
                    position = {
                        'entry_time': row['timestamp'],
                        'entry_price': row['close'],
                        'position_size': position_size,
                        'atr': row['atr'],
                        'market_state': row['market_state'],
                        'signal_strength': signal_strength,
                        'max_price': row['close'],
                        'bars_held': 0
                    }
                    
                    # 根据市场状态设置止盈止损
                    if row['market_state'] == 'bull':
                        # 牛市：TP1 1.5×ATR，TP2 6×ATR，trailing SL 3×ATR
                        position['tp1'] = row['close'] * (1 + self.tp1_multiplier * row['atr_pct'])
                        position['tp2'] = row['close'] * (1 + self.tp2_multiplier * row['atr_pct'])
                        position['trailing_sl'] = row['close'] * (1 - 0.6 * row['atr_pct'])
                        position['stop_loss'] = 0.6 * row['atr_pct'] * row['close']
                    else:
                        # 震荡市：TP 1×ATR，SL 0.6×ATR
                        position['tp1'] = row['close'] * (1 + 1.0 * row['atr_pct'])
                        position['tp2'] = row['close'] * (1 + 2.0 * row['atr_pct'])
                        position['trailing_sl'] = row['close'] * (1 - 0.6 * row['atr_pct'])
                        position['stop_loss'] = 0.6 * row['atr_pct'] * row['close']
                    
                    self.positions.append(position)
                    self.trade_count += 1
                    self.daily_trade_count += 1
                    self.last_trade_hour = row['timestamp'].hour
        
        # 平仓所有持仓
        for pos in self.positions:
            self.close_position(pos, df.iloc[-1]['close'], 'end_of_data')
        
        self.analyze_results()
    
    def update_positions(self, row):
        """更新持仓 - 9.4优化"""
        positions_to_close = []
        
        for i, pos in enumerate(self.positions):
            pos['bars_held'] += 1
            pos['max_price'] = max(pos['max_price'], row['high'])
            
            # 更新移动止损
            if row['market_state'] == 'bull':
                # 牛市：0.3×ATR开始，3×ATR移动止损
                if row['close'] > pos['entry_price'] * (1 + self.trailing_start_multiplier * pos['atr'] / pos['entry_price']):
                    new_trailing_sl = pos['max_price'] * (1 - self.trailing_stop_multiplier * pos['atr'] / pos['entry_price'])
                    pos['trailing_sl'] = max(pos['trailing_sl'], new_trailing_sl)
            else:
                # 震荡市：固定止损
                pass
            
            # 检查平仓条件
            exit_type = None
            exit_price = row['close']
            
            # TP1
            if row['high'] >= pos['tp1']:
                exit_type = 'tp1'
                exit_price = pos['tp1']
            
            # TP2
            elif row['high'] >= pos['tp2']:
                exit_type = 'tp2'
                exit_price = pos['tp2']
            
            # Trailing SL
            elif row['low'] <= pos['trailing_sl']:
                exit_type = 'trailing_sl'
                exit_price = pos['trailing_sl']
            
            # Timeout
            elif pos['bars_held'] >= self.timeout_bars:
                exit_type = 'timeout'
                exit_price = row['close']
            
            if exit_type:
                self.close_position(pos, exit_price, exit_type)
                positions_to_close.append(i)
        
        # 移除已平仓的持仓
        for i in reversed(positions_to_close):
            self.positions.pop(i)
    
    def close_position(self, pos, exit_price, exit_type):
        """平仓 - 9.4优化"""
        # 计算利润（扣除成本）
        profit = (exit_price - pos['entry_price']) * pos['position_size'] * (1 - self.total_cost_rate)
        
        # 记录交易
        trade = {
            'entry_time': pos['entry_time'],
            'exit_time': datetime.now(),
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'position_size': pos['position_size'],
            'pnl': profit,
            'exit_type': exit_type,
            'hold_bars': pos['bars_held'],
            'market_state': pos['market_state'],
            'signal_strength': pos['signal_strength']
        }
        
        self.trades.append(trade)
        
        # 更新资金
        self.capital += profit
        
        # 打印交易详情
        print(f"Trade closed: PNL={profit:.4f}, Type={exit_type}, Size={pos['position_size']:.4f}, Exit={exit_price:.2f}")
    
    def analyze_results(self):
        """分析结果 - 9.4优化"""
        if not self.trades:
            print("No trades executed!")
            return
        
        df_trades = pd.DataFrame(self.trades)
        
        # 基础统计
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = df_trades['pnl'].sum()
        total_return = total_pnl / 10000  # 假设初始资金10000
        
        # 年化收益（假设30天）
        annualized_return = total_return * 365 / 30 if total_return != 0 else 0
        
        # 平均盈亏
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Profit Factor
        gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # 最大回撤
        cumulative_pnl = df_trades['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 交易频率
        if len(df_trades) > 0:
            first_trade = df_trades['entry_time'].min()
            last_trade = df_trades['exit_time'].max()
            days = (last_trade - first_trade).days + 1
            trades_per_day = total_trades / days if days > 0 else 0
        else:
            trades_per_day = 0
        
        # 按退出类型分析
        exit_analysis = df_trades.groupby('exit_type').agg({
            'pnl': ['count', 'mean', 'sum'],
            'hold_bars': 'mean'
        }).round(4)
        
        # 按市场状态分析
        market_analysis = df_trades.groupby('market_state').agg({
            'pnl': ['count', 'mean', 'sum'],
            'exit_type': lambda x: (x == 'tp1').sum() + (x == 'tp2').sum()
        }).round(4)
        
        # 输出结果
        print("\n" + "="*60)
        print("RexKing ETH 9.4 Strategy Results")
        print("="*60)
        print(f"Initial Capital: ${10000:,.2f}")
        print(f"Final Capital: ${self.capital:,.2f}")
        print(f"Total PnL: ${total_pnl:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Trades per Day: {trades_per_day:.1f}")
        
        print(f"\nExit Type Analysis:")
        print(exit_analysis)
        
        print(f"\nMarket State Analysis:")
        print(market_analysis)
        
        # 保存交易记录
        df_trades.to_csv('rexking_eth_9_4_trades.csv', index=False)
        print(f"\nTrades saved to rexking_eth_9_4_trades.csv")

def main():
    # 初始化策略
    strategy = RexKingETH94Strategy(capital=10000)
    
    # 加载4月15分钟数据
    print("\n" + "="*60)
    print("RUNNING APRIL 2025 BACKTEST")
    print("="*60)
    df_april = strategy.load_data('/Users/qiutianyu/Downloads/ETHUSDT-15m-2025-04.csv')
    
    # 计算指标
    df_april = strategy.calculate_indicators(df_april)
    
    # 运行回测
    strategy.backtest(df_april)
    
    # 重置策略
    strategy = RexKingETH94Strategy(capital=10000)
    
    # 加载5月5分钟数据
    print("\n" + "="*60)
    print("RUNNING MAY 2025 BACKTEST")
    print("="*60)
    df_may = strategy.load_data('/Users/qiutianyu/ETHUSDT-5m-2025-05/ETHUSDT-5m-2025-05.csv')
    
    # 计算指标
    df_may = strategy.calculate_indicators(df_may)
    
    # 运行回测
    strategy.backtest(df_may)

if __name__ == "__main__":
    main() 