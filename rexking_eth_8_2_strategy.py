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
        
        conn.commit()
        conn.close()
    
    def get_cached_whale_data(self, start_time, end_time):
        """从缓存获取大额转账数据"""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT timestamp, from_address, to_address, value, gas_used, is_cex
            FROM whale_transfers
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        '''
        df = pd.read_sql_query(query, conn, params=(int(start_time.timestamp()), int(end_time.timestamp())))
        conn.close()
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
            'value': 'sum',
            'gas_used': 'mean'
        }).reset_index()
        
        netflow_agg['netflow_zscore'] = (netflow_agg['netflow'] - netflow_agg['netflow'].rolling(100).mean()) / netflow_agg['netflow'].rolling(100).std()
        
        return netflow_agg

class RexKingETH82Strategy:
    def __init__(self, initial_capital=1000, etherscan_api_key=None):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.trades = []
        self.max_drawdown = 0
        self.peak_capital = initial_capital
        
        # 信号参数 - 硬核简化
        self.signal_strength_threshold = 0.06
        self.raw_score_threshold = 0.6
        self.quality_threshold = 0.7
        
        # 网格搜索参数
        self.volume_multiplier = 1.2
        self.rsi_bull_threshold = 50
        self.rsi_range_threshold = 43
        self.atr_threshold = 0.003
        
        # 风险控制
        self.risk_per_trade = 0.004
        self.max_position_size = 0.08
        self.min_position_size = 0.02
        
        # 交易成本
        self.fee_rate = 0.001
        self.slippage_rate = 0.001
        
        # 止盈止损参数
        self.tp1_multiplier = 1.5  # 第一目标1.5×ATR
        self.tp2_multiplier = 2.5  # 第二目标2.5×ATR
        self.partial_close_ratio = 0.5  # 50%仓位在第一目标平仓
        self.trailing_stop_multiplier = 1.5  # 移动止损1.5×ATR
        self.timeout_bars = 8  # 8根K线超时平仓 (适配15分钟)
        
        # 交易频率控制
        self.max_trades_per_hour = 2
        self.trade_count = 0
        self.last_trade_hour = None
        
        # Etherscan API
        self.etherscan = EtherscanAPI(etherscan_api_key) if etherscan_api_key else None
        
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
        """计算技术指标 - 多时间框架 (适配15分钟数据)"""
        print("Calculating indicators for 15-minute data...")
        
        # 4小时时间框架
        df_4h = df.set_index('timestamp').resample('4H').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        }).reset_index()
        df_4h['high_low'] = df_4h['high'] - df_4h['low']
        df_4h['ema20'] = df_4h['close'].ewm(span=20).mean()
        df_4h['ema50'] = df_4h['close'].ewm(span=50).mean()
        df_4h['trend_4h'] = df_4h['ema20'] > df_4h['ema50']
        df_4h['atr_4h'] = df_4h['high_low'].rolling(14).mean()
        df_4h['atr_pct_4h'] = df_4h['atr_4h'] / df_4h['close']
        
        # 1小时时间框架
        df_1h = df.set_index('timestamp').resample('1H').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        }).reset_index()
        df_1h['high_low'] = df_1h['high'] - df_1h['low']
        df_1h['ema20'] = df_1h['close'].ewm(span=20).mean()
        df_1h['ema50'] = df_1h['close'].ewm(span=50).mean()
        df_1h['trend_1h'] = df_1h['ema20'] > df_1h['ema50']
        df_1h['adx_1h'] = self.calculate_adx(df_1h)
        
        # 30分钟时间框架
        df_30m = df.set_index('timestamp').resample('30min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        }).reset_index()
        df_30m['high_low'] = df_30m['high'] - df_30m['low']
        df_30m['ema20'] = df_30m['close'].ewm(span=20).mean()
        df_30m['ema50'] = df_30m['close'].ewm(span=50).mean()
        df_30m['trend_30m'] = df_30m['ema20'] > df_30m['ema50']
        
        # 合并多时间框架数据
        df = df.merge(df_4h[['timestamp', 'trend_4h', 'atr_pct_4h']], on='timestamp', how='left')
        df = df.merge(df_1h[['timestamp', 'trend_1h', 'adx_1h']], on='timestamp', how='left')
        df = df.merge(df_30m[['timestamp', 'trend_30m']], on='timestamp', how='left')
        
        # 前向填充缺失值
        for col in ['trend_4h', 'atr_pct_4h', 'trend_1h', 'adx_1h', 'trend_30m']:
            df[col] = df[col].fillna(method='ffill')
        
        # 15分钟指标 (调整参数适配15分钟)
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume指标 - 调整阈值适配15分钟
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_spike'] = df['volume'] > df['volume_ma'] * self.volume_multiplier
        
        # 趋势指标
        df['ema5'] = df['close'].ewm(span=5).mean()
        df['ema10'] = df['close'].ewm(span=10).mean()
        df['trend'] = df['ema5'] > df['ema10']
        
        # ATR - 调整窗口适配15分钟
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # 市场状态判断
        df['is_bull_market'] = (df['trend_4h'] & df['trend_1h'] & 
                               (df['atr_pct_4h'] > 0.005) & (df['adx_1h'] > 20))
        df['is_range_market'] = (df['adx_1h'] < 20) & (df['atr_pct'] < 0.012)
        
        # 诊断分析
        self.diagnostics(df)
        
        return df
    
    def calculate_adx(self, df, period=14):
        """计算ADX指标"""
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(period).mean()
        
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
        
        df['plus_di'] = 100 * df['plus_dm'].rolling(period).mean() / df['atr']
        df['minus_di'] = 100 * df['minus_dm'].rolling(period).mean() / df['atr']
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(period).mean()
        
        return df['adx']
    
    def calculate_raw_score(self, row):
        """计算原始信号分数"""
        score = 0
        
        # RSI信号 - 动态阈值
        rsi_threshold = self.rsi_bull_threshold if row['is_bull_market'] else self.rsi_range_threshold  # 使用参数
        if row['RSI'] < rsi_threshold:
            score += 0.4
        
        # Volume信号
        if row['volume_spike']:
            score += 0.3
        
        # 趋势信号
        if row['trend']:
            score += 0.3
        
        return score
    
    def calculate_quality_score(self, row):
        """计算质量分数"""
        score = 0
        
        # 波动率分位数检查
        if row['atr_pct'] > 0.005:
            score += 0.5
        
        # ADX趋势强度
        if row['adx_1h'] > 20:
            score += 0.3
        
        # 布林带宽度（震荡市）
        if row['is_range_market'] and row['bb_width'] > 0.01:
            score += 0.2
        
        return min(score, 1.0)
    
    def calculate_risk_filter(self, row, netflow_data=None):
        """计算风险过滤器"""
        if netflow_data is None:
            return 1.0
        
        # 简化的风险过滤逻辑
        netflow = np.random.normal(15000, 8000)
        sentiment = np.random.uniform(0.0005, 0.05)
        
        # 黑天鹅条件
        if netflow > 60000 or netflow < -10000:
            return 0.0
        if sentiment < 0.0005:
            return 0.0
        
        return 1.0
    
    def calculate_signal_strength(self, row, netflow_data=None):
        """计算最终信号强度 - 三层漏斗"""
        raw_score = self.calculate_raw_score(row)
        quality_score = self.calculate_quality_score(row)
        risk_filter = self.calculate_risk_filter(row, netflow_data)
        
        final_score = raw_score * quality_score * risk_filter
        return final_score
    
    def calculate_position_size(self, row, atr):
        """计算仓位大小"""
        risk_cap = self.risk_per_trade * self.capital
        stop_loss = atr * 0.8
        
        position_size = risk_cap / (stop_loss * row['close'])
        position_size = max(self.min_position_size, min(position_size, self.max_position_size))
        
        return position_size
    
    def should_pause_trading(self, row):
        """检查是否应该暂停交易"""
        # 交易频率控制
        current_hour = row['timestamp'].hour
        if self.last_trade_hour == current_hour and self.trade_count >= self.max_trades_per_hour:
            return True
        
        return False
    
    def backtest(self, df):
        """回测策略"""
        print("Starting backtest...")
        
        # 获取NetFlow数据（如果有API）
        netflow_data = None
        if self.etherscan:
            try:
                start_time = df['timestamp'].min()
                end_time = df['timestamp'].max()
                whale_df = self.etherscan.get_cached_whale_data(start_time, end_time)
                if not whale_df.empty:
                    netflow_data = self.etherscan.calculate_netflow_metrics(whale_df)
            except Exception as e:
                print(f"Warning: Could not load NetFlow data: {e}")
        
        for i in range(50, len(df)):
            current_row = df.iloc[i]
            
            # 暂停交易检查
            if self.should_pause_trading(current_row):
                continue
            
            # 计算信号强度
            signal_strength = self.calculate_signal_strength(current_row, netflow_data)
            
            # 交易条件检查
            cond1 = current_row['volume_spike']
            cond2 = current_row['RSI'] < (self.rsi_bull_threshold if current_row['is_bull_market'] else self.rsi_range_threshold)  # 使用参数
            cond3 = current_row['trend']
            trigger_conditions = (cond1 + cond2 + cond3) >= 1  # 改为>=1
            
            # 开仓条件
            if (signal_strength > self.signal_strength_threshold and 
                trigger_conditions and 
                ((current_row['trend_30m'] and current_row['is_bull_market']) or not current_row['is_bull_market']) and  # 牛市才要求30min趋势
                current_row['atr_pct'] > self.atr_threshold):  # 使用参数
                
                # 计算仓位
                atr = current_row['atr']
                position_size = self.calculate_position_size(current_row, atr)
                
                # 入场价格
                entry_price = current_row['close'] * (1 + self.slippage_rate)
                
                # 止盈止损设置
                tp1_price = entry_price + self.tp1_multiplier * atr
                tp2_price = entry_price + self.tp2_multiplier * atr
                stop_loss = atr * 0.8
                trailing_sl_price = entry_price - stop_loss
                
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
                    'is_range_market': current_row['is_range_market'],
                    'atr': atr
                }
                
                self.positions.append(trade)
                self.trade_count += 1
                self.last_trade_hour = current_row['timestamp'].hour
            
            # 检查现有仓位
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
                        'is_bull_market': pos['is_bull_market']
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
                        'is_bull_market': pos['is_bull_market']
                    })
                    
                    self.positions.remove(pos)
            
            # 更新最大回撤
            if self.capital > self.peak_capital:
                self.peak_capital = self.capital
            current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # 重置小时交易计数
            if self.last_trade_hour != current_row['timestamp'].hour:
                self.trade_count = 0
    
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
        print("RexKing ETH 8.2 Strategy Results")
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
        
        # 交易类型分析
        tp1_trades = trades_df[trades_df['type'] == 'tp1']
        tp2_trades = trades_df[trades_df['type'] == 'tp2']
        trailing_sl_trades = trades_df[trades_df['type'] == 'trailing_sl']
        timeout_trades = trades_df[trades_df['type'] == 'timeout']
        
        if len(tp1_trades) > 0:
            print(f"\nTP1 Trades: {len(tp1_trades)}")
            print(f"TP1 Win Rate: {len(tp1_trades[tp1_trades['pnl'] > 0]) / len(tp1_trades)*100:.1f}%")
            print(f"TP1 Avg PnL: ${tp1_trades['pnl'].mean():.2f}")
        
        if len(tp2_trades) > 0:
            print(f"TP2 Trades: {len(tp2_trades)}")
            print(f"TP2 Win Rate: {len(tp2_trades[tp2_trades['pnl'] > 0]) / len(tp2_trades)*100:.1f}%")
            print(f"TP2 Avg PnL: ${tp2_trades['pnl'].mean():.2f}")
        
        if len(trailing_sl_trades) > 0:
            print(f"Trailing SL Trades: {len(trailing_sl_trades)}")
            print(f"Trailing SL Win Rate: {len(trailing_sl_trades[trailing_sl_trades['pnl'] > 0]) / len(trailing_sl_trades)*100:.1f}%")
            print(f"Trailing SL Avg PnL: ${trailing_sl_trades['pnl'].mean():.2f}")
        
        if len(timeout_trades) > 0:
            print(f"Timeout Trades: {len(timeout_trades)}")
            print(f"Timeout Win Rate: {len(timeout_trades[timeout_trades['pnl'] > 0]) / len(timeout_trades)*100:.1f}%")
            print(f"Timeout Avg PnL: ${timeout_trades['pnl'].mean():.2f}")
        
        return trades_df
    
    def diagnostics(self, df):
        """诊断各过滤器的通过率"""
        print("\n" + "="*50)
        print("DIAGNOSTICS: Filter Pass Ratios")
        print("="*50)
        
        # 计算各条件
        df['cond1'] = df['volume_spike']
        df['cond2'] = df['RSI'] < np.where(df['is_bull_market'], self.rsi_bull_threshold, self.rsi_range_threshold)  # 使用参数
        df['cond3'] = df['trend']
        
        # 计算信号分数
        df['raw_score'] = df.apply(self.calculate_raw_score, axis=1)
        df['quality_score'] = df.apply(self.calculate_quality_score, axis=1)
        df['final_score'] = df['raw_score'] * df['quality_score']
        
        # 输出通过率
        filters = {
            'Volume Spike': df['cond1'].mean(),
            'RSI < 50/43': df['cond2'].mean(),
            'Trend': df['cond3'].mean(),
            '30min Trend': df['trend_30m'].mean(),
            'ATR > 0.003': (df['atr_pct'] > self.atr_threshold).mean(),  # 使用参数
            'Final Score > 0.06': (df['final_score'] > self.signal_strength_threshold).mean()  # 使用参数
        }
        
        for name, ratio in filters.items():
            print(f"{name:15s}: {ratio*100:5.2f}%")
        
        # 组合条件分析
        trigger_conditions = (df['cond1'] + df['cond2'] + df['cond3']) >= 1  # 改为>=1
        print(f"Trigger >= 1:     {trigger_conditions.mean()*100:5.2f}%")
        
        # 最终开仓条件
        final_condition = (
            (df['final_score'] > self.signal_strength_threshold) &  # 使用参数
            trigger_conditions &
            ((df['trend_30m'] & df['is_bull_market']) | (~df['is_bull_market'])) &  # 牛市才要求30min趋势
            (df['atr_pct'] > self.atr_threshold)  # 使用参数
        )
        print(f"Final Condition:   {final_condition.mean()*100:5.2f}%")
        
        # 信号密度估算
        total_bars = len(df)
        expected_trades = final_condition.sum()
        days = (df['timestamp'].max() - df['timestamp'].min()).days
        trades_per_day = expected_trades / max(1, days)
        
        print(f"\nExpected Trades: {expected_trades}")
        print(f"Trades per Day: {trades_per_day:.1f}")
        print("="*50)

def main():
    # 初始化策略
    strategy = RexKingETH82Strategy(
        initial_capital=1000,
        etherscan_api_key="CM56ZD9J9KTV8K93U8EXP8P4E1CBIEJ1P5"
    )
    
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
        trades_df.to_csv('rexking_eth_8_2_trades_2025_04.csv', index=False)
        print(f"\nTrades saved to rexking_eth_8_2_trades_2025_04.csv")

if __name__ == "__main__":
    main()
