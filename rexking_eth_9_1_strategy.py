import pandas as pd
import numpy as np
import requests
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EtherscanAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.etherscan.io/api"
        self.db_path = "etherscan_cache.db"
        self.init_database()
    
    def init_database(self):
        """初始化SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS whale_transfers (
                timestamp INTEGER,
                from_address TEXT,
                to_address TEXT,
                value REAL,
                gas_used INTEGER,
                is_cex INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_data (
                timestamp INTEGER,
                sentiment_score REAL,
                volume_24h REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_whale_transfers(self, start_time, end_time, min_value=5000):
        """获取大额转账数据 (W1)"""
        try:
            # 从缓存获取
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT timestamp, from_address, to_address, value, gas_used, is_cex
                FROM whale_transfers
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            '''
            df = pd.read_sql_query(query, conn, params=(int(start_time.timestamp()), int(end_time.timestamp())))
            conn.close()
            
            if not df.empty:
                return self.calculate_netflow_metrics(df)
            
            # 如果缓存为空，返回模拟数据
            return self.generate_mock_whale_data(start_time, end_time)
            
        except Exception as e:
            print(f"Error fetching whale data: {e}")
            return self.generate_mock_whale_data(start_time, end_time)
    
    def generate_mock_whale_data(self, start_time, end_time):
        """生成模拟大额转账数据 - 9.1修复time_window"""
        time_range = pd.date_range(start=start_time, end=end_time, freq='10min')
        mock_data = []
        
        for ts in time_range:
            # 模拟W1数据：5000-8000 ETH范围
            w1_value = np.random.normal(6500, 1000)
            w1_value = max(5000, min(8000, w1_value))
            
            mock_data.append({
                'timestamp': ts,
                'w1_netflow': w1_value,
                'w1_zscore': np.random.normal(0, 1)
            })
        
        df = pd.DataFrame(mock_data)
        # 9.1修复：添加time_window字段
        df['time_window'] = pd.to_datetime(df['timestamp']).dt.floor('10min')
        return df
    
    def calculate_netflow_metrics(self, df, window_minutes=10):
        """计算NetFlow指标"""
        if df.empty:
            return pd.DataFrame()
        
        df['netflow'] = df.apply(lambda row: 
            row['value'] if row['is_cex'] == 0 else -row['value'], axis=1)
        
        df['time_window'] = pd.to_datetime(df['timestamp'], unit='s').dt.floor(f'{window_minutes}min')
        netflow_agg = df.groupby('time_window').agg({
            'netflow': 'sum',
            'value': 'sum'
        }).reset_index()
        
        netflow_agg['w1_netflow'] = netflow_agg['netflow']
        netflow_agg['w1_zscore'] = (netflow_agg['netflow'] - netflow_agg['netflow'].rolling(100).mean()) / netflow_agg['netflow'].rolling(100).std()
        
        return netflow_agg[['time_window', 'w1_netflow', 'w1_zscore']]

class RexKingETH91Strategy:
    def __init__(self, capital=10000):
        self.capital = capital
        self.positions = []
        self.trades = []
        self.signal_strength_threshold = 0.06
        self.max_trades_per_hour = 1
        self.atr_threshold = 0.002
        self.risk_per_trade = 0.002  # 0.2%风险
        self.fee_rate = 0.001  # 0.1%手续费
        self.slippage_rate = 0.001  # 0.1%滑点
        
        # 止盈止损参数
        self.tp1_multiplier = 1.5  # 第一目标1.5×ATR
        self.tp2_multiplier = 5.0  # 第二目标5×ATR
        self.partial_close_ratio = 0.5  # 50%仓位在第一目标平仓
        self.trailing_stop_multiplier = 2.0  # 移动止损2×ATR
        self.trailing_stop_start = 0.2  # 从0.2×ATR开始移动止损
        self.timeout_bars = 12  # 12根K线超时平仓

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
        """计算技术指标 - 9.1优化"""
        print("Calculating indicators for RexKing ETH 9.1...")
        
        # 30分钟时间框架 - 趋势确认
        df_30m = df.set_index('timestamp').resample('30min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        }).reset_index()
        df_30m['ema20'] = df_30m['close'].ewm(span=20).mean()
        df_30m['ema50'] = df_30m['close'].ewm(span=50).mean()
        df_30m['trend_30m'] = df_30m['ema20'] > df_30m['ema50']
        
        # 合并30分钟数据
        df = df.merge(df_30m[['timestamp', 'trend_30m']], on='timestamp', how='left')
        df = df.fillna({'trend_30m': False})
        
        # 15分钟指标 - 入场信号
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        df['volume_spike'] = df['volume'] > df['volume'].rolling(20).mean() * 1.2
        df['atr'] = (df['high_low'].rolling(14).mean() / df['close']).fillna(method='bfill')
        df['min_atr'] = df['atr'] > self.atr_threshold
        
        # 分位数计算
        df['score_percentile'] = df['rsi'].rolling(1000).apply(
            lambda x: pd.qcut(x, 10, labels=False).iloc[-1]/10 if len(x.dropna()) > 10 else 0.5
        )
        
        # 市场状态判断
        df['is_bull_market'] = df['trend_30m']
        df['is_range_market'] = ~df['trend_30m']
        
        # 模拟Etherscan数据
        whale_df = self.mock_etherscan_data(df)
        df = df.merge(whale_df, on='timestamp', how='left')
        df['w1_netflow'] = df['w1_netflow'].fillna(6500)
        df['w1_zscore'] = df['w1_zscore'].fillna(0)
        
        # 诊断分析
        self.diagnostics(df)
        
        return df

    def calculate_rsi(self, prices, period):
        """计算RSI指标"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_signal_strength(self, row):
        """计算信号强度 - 9.1优化"""
        score = 0
        
        # RSI信号 - 牛市/震荡市不同阈值
        rsi_threshold = 55 if row['is_bull_market'] else 50
        if row['rsi'] < rsi_threshold:
            score += 0.4
        
        # Volume信号
        if row['volume_spike']:
            score += 0.3
        
        # 30分钟趋势确认
        if row['trend_30m']:
            score += 0.3
        
        # Etherscan过滤 (W1 > 5000, S1 > 0.05)
        if row['w1_netflow'] > 5000 and row['w1_zscore'] > 0.05:
            score *= 1.2  # 增强信号
        
        return min(score, 1.0)

    def should_pause_trading(self, row):
        """检查是否应该暂停交易"""
        # 模拟Etherscan数据
        netflow = row['w1_netflow']
        sentiment = row['w1_zscore']
        
        # 黑天鹅条件
        if netflow < -10000 or sentiment < 0.0001:
            return True
        
        return False

    def mock_etherscan_data(self, df):
        """模拟Etherscan数据 - 9.1修复time_window"""
        whale_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'w1_netflow': np.random.normal(6500, 1000, len(df)),
            'w1_zscore': np.random.normal(0, 1, len(df))
        })
        whale_df['time_window'] = pd.to_datetime(whale_df['timestamp']).dt.floor('10min')
        return whale_df

    def backtest(self, df):
        """回测策略 - 9.1优化"""
        print("Starting RexKing ETH 9.1 backtest...")
        
        last_trade_time = None
        hourly_trades = 0
        last_hour = None
        
        for i in range(50, len(df)):
            current_row = df.iloc[i]
            current_hour = current_row['timestamp'].floor('H')
            
            # 重置小时交易计数
            if current_hour != last_hour:
                hourly_trades = 0
                last_hour = current_hour
            
            # 交易频率控制
            if hourly_trades >= self.max_trades_per_hour:
                continue
            
            # 暂停交易检查
            if self.should_pause_trading(current_row):
                continue
            
            # 计算信号强度
            signal_strength = self.calculate_signal_strength(current_row)
            
            # 调试打印 - 前100行
            if i < 150:
                print(f"Row {i}, Time: {current_row['timestamp']}, Signal: {signal_strength:.2f}, "
                      f"Trend_30m: {current_row['trend_30m']}, Min_ATR: {current_row['min_atr']}, "
                      f"RSI: {current_row['rsi']:.2f}, Volume_Spike: {current_row['volume_spike']}")
            
            # 开仓条件 - 9.1优化（去掉4小时ATR）
            if (signal_strength > self.signal_strength_threshold and 
                current_row['trend_30m'] and 
                current_row['min_atr']):
                
                print(f"Open signal at {current_row['timestamp']}, Signal: {signal_strength:.2f}")
                
                # 计算仓位
                atr = current_row['atr']
                risk_cap = self.risk_per_trade * self.capital
                stop_loss = atr * 0.8
                
                if stop_loss <= 0:
                    continue
                
                position_size = risk_cap / (stop_loss * current_row['close'])
                position_size = min(position_size, 0.08)
                
                # 入场价格
                entry_price = current_row['close'] * (1 + self.slippage_rate)
                
                # 止盈止损设置
                tp1_price = entry_price + self.tp1_multiplier * atr
                tp2_price = entry_price + self.tp2_multiplier * atr
                trailing_sl_price = entry_price - self.trailing_stop_start * atr
                
                # 记录交易
                trade = {
                    'entry_time': current_row['timestamp'],
                    'entry_price': entry_price,
                    'position_size': position_size,
                    'tp1_price': tp1_price,
                    'tp2_price': tp2_price,
                    'trailing_sl_price': trailing_sl_price,
                    'bars_open': 0,
                    'partial_closed': False,
                    'tp2_closed': False,
                    'max_price': entry_price,
                    'signal_strength': signal_strength,
                    'is_bull_market': current_row['is_bull_market'],
                    'atr': atr,
                    'w1_netflow': current_row['w1_netflow']
                }
                
                self.positions.append(trade)
                last_trade_time = current_row['timestamp']
                hourly_trades += 1
            
            # 检查现有仓位
            for pos in self.positions[:]:
                pos['bars_open'] += 1
                current_price = current_row['close']
                
                # 更新最高价和移动止损
                pos['max_price'] = max(pos['max_price'], current_price)
                
                # 从0.2×ATR开始移动止损
                if current_price > pos['entry_price'] + self.trailing_stop_start * pos['atr']:
                    pos['trailing_sl_price'] = max(
                        pos['trailing_sl_price'], 
                        pos['max_price'] - self.trailing_stop_multiplier * pos['atr']
                    )
                
                # 第一目标止盈
                if not pos['partial_closed'] and current_price >= pos['tp1_price']:
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
                        'is_bull_market': pos['is_bull_market'],
                        'w1_netflow': pos['w1_netflow']
                    })
                    
                    pos['position_size'] *= (1 - self.partial_close_ratio)
                    pos['partial_closed'] = True
                
                # 第二目标止盈
                elif pos['partial_closed'] and not pos['tp2_closed'] and current_price >= pos['tp2_price']:
                    gross_profit = (pos['tp2_price'] - pos['entry_price']) * pos['position_size']
                    net_profit = gross_profit * (1 - self.fee_rate - self.slippage_rate)
                    self.capital += net_profit
                    
                    self.trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': current_row['timestamp'],
                        'entry_price': pos['entry_price'],
                        'exit_price': pos['tp2_price'],
                        'position_size': pos['position_size'],
                        'pnl': net_profit,
                        'type': 'tp2',
                        'signal_strength': pos['signal_strength'],
                        'is_bull_market': pos['is_bull_market'],
                        'w1_netflow': pos['w1_netflow']
                    })
                    
                    self.positions.remove(pos)
                    pos['tp2_closed'] = True
                
                # 移动止损或超时平仓
                elif (current_price <= pos['trailing_sl_price'] or 
                      pos['bars_open'] >= self.timeout_bars):
                    
                    if current_price <= pos['trailing_sl_price']:
                        exit_price = pos['trailing_sl_price']
                        exit_type = 'trailing_sl'
                    else:
                        exit_price = current_price
                        exit_type = 'timeout'
                    
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
                        'is_bull_market': pos['is_bull_market'],
                        'w1_netflow': pos['w1_netflow']
                    })
                    
                    self.positions.remove(pos)

    def diagnostics(self, df):
        """诊断各过滤器的通过率"""
        print("\n" + "="*50)
        print("REXKING ETH 9.1 DIAGNOSTICS")
        print("="*50)
        
        # 计算各条件
        df['cond1'] = df['volume_spike']
        df['cond2'] = df['rsi'] < np.where(df['is_bull_market'], 55, 50)
        df['cond3'] = df['trend_30m']
        df['cond4'] = df['min_atr']
        df['cond5'] = (df['w1_netflow'] > 5000) & (df['w1_zscore'] > 0.05)
        
        # 计算信号分数
        df['signal_strength'] = df.apply(self.calculate_signal_strength, axis=1)
        
        # 输出通过率
        filters = {
            'Volume > 1.2×': df['cond1'].mean(),
            'RSI < 55/50': df['cond2'].mean(),
            '30min Trend': df['cond3'].mean(),
            'ATR > 0.002': df['cond4'].mean(),
            'W1 > 5000 & S1 > 0.05': df['cond5'].mean(),
            'Signal > 0.06': (df['signal_strength'] > self.signal_strength_threshold).mean()
        }
        
        for name, ratio in filters.items():
            print(f"{name:20s}: {ratio*100:5.2f}%")
        
        # 最终开仓条件（去掉4小时ATR）
        final_condition = (
            (df['signal_strength'] > self.signal_strength_threshold) &
            df['cond3'] &  # 30min trend
            df['cond4']    # min ATR
        )
        print(f"Final Condition:        {final_condition.mean()*100:5.2f}%")
        
        # 信号密度估算
        expected_trades = final_condition.sum()
        days = (df['timestamp'].max() - df['timestamp'].min()).days
        trades_per_day = expected_trades / max(1, days)
        
        print(f"\nExpected Trades: {expected_trades}")
        print(f"Trades per Day: {trades_per_day:.1f}")
        print("="*50)

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
            annual_return = (total_pnl / 10000) * (365 / days) if days > 0 else 0
        else:
            annual_return = 0
        
        # 信号密度
        signal_density = total_trades / max(1, (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days)
        
        print("\n" + "="*60)
        print("RexKing ETH 9.1 Strategy Results")
        print("="*60)
        print(f"Initial Capital: $10,000.00")
        print(f"Final Capital: ${self.capital:,.2f}")
        print(f"Total PnL: ${total_pnl:,.2f}")
        print(f"Total Return: {(self.capital/10000 - 1)*100:.2f}%")
        print(f"Annualized Return: {annual_return*100:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate*100:.1f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Signal Density: {signal_density:.1f} trades/day")
        
        # 市场状态分析
        bull_trades = trades_df[trades_df['is_bull_market'] == True]
        range_trades = trades_df[trades_df['is_bull_market'] == False]
        
        if len(bull_trades) > 0:
            bull_win_rate = len(bull_trades[bull_trades['pnl'] > 0]) / len(bull_trades)
            bull_pnl = bull_trades['pnl'].sum()
            print(f"\nBull Market Trades: {len(bull_trades)}")
            print(f"Bull Market Win Rate: {bull_win_rate*100:.1f}%")
            print(f"Bull Market PnL: ${bull_pnl:.2f}")
        
        if len(range_trades) > 0:
            range_win_rate = len(range_trades[range_trades['pnl'] > 0]) / len(range_trades)
            range_pnl = range_trades['pnl'].sum()
            print(f"Range Market Trades: {len(range_trades)}")
            print(f"Range Market Win Rate: {range_win_rate*100:.1f}%")
            print(f"Range Market PnL: ${range_pnl:.2f}")
        
        return trades_df

def main():
    # 初始化策略
    strategy = RexKingETH91Strategy(capital=10000)
    
    # 加载4月15分钟数据
    df = strategy.load_data('/Users/qiutianyu/Downloads/ETHUSDT-15m-2025-04.csv')
    
    # 计算指标
    df = strategy.calculate_indicators(df)
    
    # 回测
    strategy.backtest(df)
    
    # 分析结果
    trades_df = strategy.analyze_results()
    
    # 保存交易记录
    if trades_df is not None and len(trades_df) > 0:
        trades_df.to_csv('rexking_eth_9_1_trades_2025_04.csv', index=False)
        print(f"\nTrades saved to rexking_eth_9_1_trades_2025_04.csv")

if __name__ == "__main__":
    main() 