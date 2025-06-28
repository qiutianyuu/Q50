import pandas as pd
import numpy as np
import sqlite3
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EtherscanAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key or "YourApiKeyToken"  # 需要真实API key
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
    
    def get_whale_data(self, start_time, end_time):
        """获取大额转账数据（Mock版本 - 后续替换为真实API）"""
        # 真实API调用（需要API key）
        # url = f"{self.base_url}?module=account&action=txlist&address=0x28C6c06298d514Db089934071355E5743bf21d60&startblock=0&endblock=99999999&page=1&offset=100&sort=desc&apikey={self.api_key}"
        
        # Mock数据 - 模拟真实的大额转账（W1>10000）
        np.random.seed(42)
        timestamps = pd.date_range(start_time, end_time, freq='10min')
        values = np.random.normal(8000, 4000, len(timestamps))
        values = np.where(values < 0, 0, values)  # 确保非负
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values,
            'is_cex': np.random.choice([0, 1], len(timestamps), p=[0.7, 0.3])
        })
        
        return df
    
    def calculate_netflow_metrics(self, df, window_minutes=10):
        """计算NetFlow指标"""
        if df.empty:
            return pd.DataFrame()
        
        df['netflow'] = df.apply(lambda row: 
            row['value'] if row['is_cex'] == 0 else -row['value'], axis=1)
        
        df['time_window'] = pd.to_datetime(df['timestamp']).dt.floor(f'{window_minutes}min')
        netflow_agg = df.groupby('time_window').agg({
            'netflow': 'sum',
            'value': 'sum'
        }).reset_index()
        
        netflow_agg['w1_zscore'] = (netflow_agg['value'] - netflow_agg['value'].rolling(100).mean()) / netflow_agg['value'].rolling(100).std()
        netflow_agg['w1_signal'] = (netflow_agg['value'] > 10000) & (netflow_agg['w1_zscore'] > 2)
        
        return netflow_agg

class RexKingETH110Strategy:
    def __init__(self, capital=10000, etherscan_api_key=None):
        self.initial_capital = capital
        self.capital = capital
        self.positions = []
        self.trades = []
        self.max_drawdown = 0
        self.peak_capital = capital
        
        # 信号参数 - 11.0优化
        self.volume_multiplier = 1.5
        self.bb_width_threshold = 0.04  # BB宽度 > 4%
        self.w1_threshold = 10000  # W1 > 10000
        
        # 风险控制 - 动态Kelly仓位
        self.kelly_fraction = 1/3  # Kelly 1/3
        self.base_risk = 0.1  # 初始10%
        self.min_risk = 0.05  # 最低5%
        self.recent_trades = []  # 最近交易记录
        
        # 交易成本
        self.fee_rate = 0.00075  # 0.075%手续费（Binance VIP0）
        self.slippage_rate = 0.0005  # 0.05%滑点
        self.total_cost_rate = 0.00125  # 总成本0.125%
        
        # 止盈止损参数 - 11.0优化
        self.be_multiplier = 1.0  # +1×ATR移到BE
        self.partial_tp_multiplier = 2.0  # +2×ATR出50%
        self.stop_loss_multiplier = 1.0  # -1×ATR止损
        self.timeout_bars = 16  # 16根K线超时（4小时）
        
        # 交易频率控制
        self.max_trades_per_day = 1  # 每日最多1笔
        self.min_hold_bars = 16  # 最少持仓4小时
        self.daily_trade_count = 0
        self.last_trade_date = None
        
        # 风控参数
        self.max_drawdown_limit = 0.08  # 最大回撤8%
        self.crash_threshold = 0.03  # 5分钟跌3%
        self.cooling_period_hours = 48  # 冷静期48小时
        self.last_crash_time = None
        
        # Etherscan API
        self.etherscan = EtherscanAPI(etherscan_api_key)
        
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
        """计算技术指标 - 4H信号+15m触发"""
        print("Calculating 4H framework indicators...")
        
        # 4小时时间框架
        df_4h = df.set_index('timestamp').resample('4H').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        }).reset_index()
        
        # 4H EMA
        df_4h['ema20'] = df_4h['close'].ewm(span=20).mean()
        df_4h['ema60'] = df_4h['close'].ewm(span=60).mean()
        df_4h['trend_4h'] = df_4h['ema20'] > df_4h['ema60']
        
        # 4H OBV
        df_4h['obv'] = (np.where(df_4h['close'] > df_4h['close'].shift(1), 
                                  df_4h['volume'], -df_4h['volume'])).cumsum()
        df_4h['obv_ma'] = df_4h['obv'].rolling(14).mean()
        
        # 4H ATR
        df_4h['high_low'] = df_4h['high'] - df_4h['low']
        df_4h['atr_4h'] = df_4h['high_low'].rolling(14).mean()
        df_4h['atr_pct_4h'] = df_4h['atr_4h'] / df_4h['close']
        
        # 4H 布林带宽度
        df_4h['bb_upper'] = df_4h['close'].rolling(20).mean() + 2 * df_4h['close'].rolling(20).std()
        df_4h['bb_lower'] = df_4h['close'].rolling(20).mean() - 2 * df_4h['close'].rolling(20).std()
        df_4h['bb_width'] = (df_4h['bb_upper'] - df_4h['bb_lower']) / df_4h['close']
        df_4h['bb_volatile'] = df_4h['bb_width'] > self.bb_width_threshold
        
        # 4H 核心信号
        df_4h['signal_4h'] = (df_4h['ema20'] > df_4h['ema60']) & (df_4h['obv'] > df_4h['obv_ma']) & (df_4h['bb_volatile'])
        
        # 合并4H数据到15m
        df_4h_merge = df_4h[['timestamp', 'signal_4h', 'atr_pct_4h', 'low']].copy()
        df_4h_merge = df_4h_merge.rename(columns={'low': 'low_4h'})
        df = df.merge(df_4h_merge, on='timestamp', how='left')
        
        # 前向填充缺失值
        for col in ['signal_4h', 'atr_pct_4h', 'low_4h']:
            df[col] = df[col].fillna(method='ffill')
        
        # 15分钟触发信号
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_spike'] = df['volume'] > df['volume_ma'] * self.volume_multiplier
        df['k_break'] = (df['close'] > df['high'].shift(1)) & (df['volume'] > df['volume_ma'] * self.volume_multiplier)
        
        # 黑天鹅检测
        df['crash'] = df['close'].pct_change() < -self.crash_threshold
        
        # Etherscan数据
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        whale_data = self.etherscan.get_whale_data(start_time, end_time)
        netflow_data = self.etherscan.calculate_netflow_metrics(whale_data)
        
        # 合并Etherscan数据
        df = df.merge(netflow_data[['time_window', 'w1_signal']], 
                     left_on='timestamp', right_on='time_window', how='left')
        df['w1_signal'] = df['w1_signal'].fillna(False)
        
        return df
    
    def recent_win_rate(self, lookback=20):
        """计算最近N笔交易的胜率"""
        if len(self.recent_trades) < lookback:
            return 0.6  # 默认胜率60%
        
        recent = self.recent_trades[-lookback:]
        wins = sum(1 for trade in recent if trade['pnl'] > 0)
        return wins / len(recent)
    
    def calculate_position_size(self, row, atr_4h):
        """计算动态Kelly仓位"""
        # 计算胜率
        win_rate = self.recent_win_rate(20)
        
        # 根据胜率调整风险
        if win_rate > 0.45:
            risk_cap = self.capital * self.base_risk  # 10%
        else:
            risk_cap = self.capital * self.min_risk   # 5%
        
        stop_loss = atr_4h * self.stop_loss_multiplier * row['close']
        position_size = risk_cap / stop_loss
        position_size = min(position_size, 0.15)  # 最大15%
        
        return position_size
    
    def should_pause_trading(self, row):
        """交易频率控制 + 风控"""
        current_date = row['timestamp'].date()
        
        # 重置每日计数
        if self.last_trade_date != current_date:
            self.daily_trade_count = 0
            self.last_trade_date = current_date
        
        # 每日限制
        if self.daily_trade_count >= self.max_trades_per_day:
            return True
        
        # 黑天鹅检测
        if row['crash']:
            self.last_crash_time = row['timestamp']
            self.positions.clear()  # 一键平仓
            return True
        
        # 回撤控制
        current_drawdown = (self.capital - self.peak_capital) / self.peak_capital
        if current_drawdown < -self.max_drawdown_limit:
            return True
        
        # 冷静期
        if (self.last_crash_time and 
            (row['timestamp'] - self.last_crash_time).total_seconds() < self.cooling_period_hours * 3600):
            return True
        
        return False
    
    def backtest(self, df):
        """回测 - 11.0优化"""
        print("Starting RexKing ETH 11.0 backtest...")
        
        for i, row in df.iterrows():
            # 更新峰值资金
            if self.capital > self.peak_capital:
                self.peak_capital = self.capital
            
            # 更新持仓
            self.update_positions(row)
            
            # 频率控制 + 风控
            if self.should_pause_trading(row):
                continue
            
            # 开仓条件
            if len(self.positions) == 0:
                # 4H信号 + 15m触发 + W1信号
                if (row['signal_4h'] and row['k_break'] and row['w1_signal']):
                    
                    # 计算仓位
                    position_size = self.calculate_position_size(row, row['atr_pct_4h'])
                    
                    # 开仓
                    position = {
                        'entry_time': row['timestamp'],
                        'entry_price': row['close'],
                        'position_size': position_size,
                        'atr_4h': row['atr_pct_4h'],
                        'signal_strength': 1.0,  # 简化信号强度
                        'max_price': row['close'],
                        'bars_held': 0,
                        'partial_closed': False,  # 是否已部分平仓
                        'trailing_sl': row['close'] * (1 - self.stop_loss_multiplier * row['atr_pct_4h'])
                    }
                    
                    self.positions.append(position)
                    self.daily_trade_count += 1
        
        # 平仓所有持仓
        for pos in self.positions:
            self.close_position(pos, df.iloc[-1]['close'], 'end_of_data')
        
        self.analyze_results()
    
    def update_positions(self, row):
        """更新持仓 - 11.0优化"""
        positions_to_close = []
        
        for i, pos in enumerate(self.positions):
            pos['bars_held'] += 1
            pos['max_price'] = max(pos['max_price'], row['high'])
            
            # 分级锁盈逻辑
            current_price = row['close']
            entry_price = pos['entry_price']
            atr_4h = pos['atr_4h']
            
            # +1×ATR移到BE
            if current_price >= entry_price * (1 + self.be_multiplier * atr_4h):
                pos['trailing_sl'] = entry_price
            
            # +2×ATR出50%
            if (current_price >= entry_price * (1 + self.partial_tp_multiplier * atr_4h) and 
                not pos['partial_closed']):
                
                # 部分平仓
                partial_profit = (current_price - entry_price) * pos['position_size'] * 0.5 * (1 - self.total_cost_rate)
                self.capital += partial_profit
                pos['position_size'] *= 0.5
                pos['partial_closed'] = True
                
                print(f"Partial close: PNL={partial_profit:.4f}, Remaining size={pos['position_size']:.4f}")
            
            # Higher-Low追踪
            if pos['partial_closed']:
                # 使用4H低点作为追踪止损
                pos['trailing_sl'] = max(pos['trailing_sl'], row['low_4h'])
            
            # 检查平仓条件
            exit_type = None
            exit_price = current_price
            
            # 止损
            if row['low'] <= pos['trailing_sl']:
                exit_type = 'stop_loss'
                exit_price = pos['trailing_sl']
            
            # Timeout
            elif pos['bars_held'] >= self.timeout_bars:
                exit_type = 'timeout'
                exit_price = current_price
            
            if exit_type:
                self.close_position(pos, exit_price, exit_type)
                positions_to_close.append(i)
        
        # 移除已平仓的持仓
        for i in reversed(positions_to_close):
            self.positions.pop(i)
    
    def close_position(self, pos, exit_price, exit_type):
        """平仓 - 11.0优化"""
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
            'signal_strength': pos['signal_strength']
        }
        
        self.trades.append(trade)
        self.recent_trades.append(trade)
        
        # 保持最近100笔交易
        if len(self.recent_trades) > 100:
            self.recent_trades.pop(0)
        
        # 更新资金
        self.capital += profit
        
        # 打印交易详情
        print(f"Trade closed: PNL={profit:.4f}, Type={exit_type}, Size={pos['position_size']:.4f}, Exit={exit_price:.2f}")
    
    def analyze_results(self):
        """分析结果 - 11.0优化"""
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
        total_return = total_pnl / self.initial_capital
        
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
        
        # 输出结果
        print("\n" + "="*60)
        print("RexKing ETH 11.0 Strategy Results")
        print("="*60)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
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
        
        # 保存交易记录
        df_trades.to_csv('rexking_eth_11_0_trades.csv', index=False)
        print(f"\nTrades saved to rexking_eth_11_0_trades.csv")

def main():
    # 初始化策略
    strategy = RexKingETH110Strategy(capital=10000)
    
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
    strategy = RexKingETH110Strategy(capital=10000)
    
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