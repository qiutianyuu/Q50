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
        """获取大额转账数据（Mock版本）"""
        # 真实API调用（需要API key）
        # url = f"{self.base_url}?module=account&action=txlist&address=0x28C6c06298d514Db089934071355E5743bf21d60&startblock=0&endblock=99999999&page=1&offset=100&sort=desc&apikey={self.api_key}"
        
        # Mock数据 - 模拟真实的大额转账
        np.random.seed(42)
        timestamps = pd.date_range(start_time, end_time, freq='10min')
        values = np.random.normal(5000, 3000, len(timestamps))
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
        netflow_agg['w1_signal'] = (netflow_agg['value'] > 8000) & (netflow_agg['w1_zscore'] > 2)
        
        return netflow_agg

class RexKingETH100Strategy:
    def __init__(self, capital=10000, etherscan_api_key=None):
        self.initial_capital = capital
        self.capital = capital
        self.positions = []
        self.trades = []
        self.max_drawdown = 0
        self.peak_capital = capital
        
        # 信号参数 - 10.0优化
        self.signal_strength_threshold = 0.08  # 提升到0.08
        self.rsi_bull_threshold = 50
        self.rsi_range_threshold = 45
        self.volume_multiplier = 1.5
        self.atr_threshold_1h = 0.006  # 1H ATR > 0.6%
        self.bb_width_threshold = 0.04  # BB宽度 > 4%
        
        # 风险控制 - 动态仓位
        self.risk_per_trade_bull = 0.005  # 牛市0.5%
        self.risk_per_trade_range = 0.003  # 震荡市0.3%
        self.max_position_size = 0.15  # 最大15%仓位
        self.fee_rate = 0.00075  # 0.075%手续费（Binance VIP0）
        self.slippage_rate = 0.0005  # 0.05%滑点
        self.total_cost_rate = 0.00125  # 总成本0.125%
        
        # 止盈止损参数 - 10.0优化
        self.tp1_multiplier_bull = 2.0  # 牛市TP1 2×ATR
        self.tp1_multiplier_range = 1.5  # 震荡市TP1 1.5×ATR
        self.tp2_multiplier = 4.0  # TP2 4×ATR
        self.partial_close_ratio = 0.4  # 40%仓位在第一目标平仓
        self.trailing_stop_multiplier = 4.0  # 移动止损4×ATR
        self.stop_loss_multiplier = 0.8  # 止损0.8×ATR
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
        """计算技术指标 - 1H/4H框架 + 15m触发"""
        print("Calculating 1H/4H framework indicators...")
        
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
        df_1h['atr_1h'] = df_1h['high_low'].rolling(14).mean()
        df_1h['atr_pct_1h'] = df_1h['atr_1h'] / df_1h['close']
        
        # 布林带宽度
        df_1h['bb_upper'] = df_1h['close'].rolling(20).mean() + 2 * df_1h['close'].rolling(20).std()
        df_1h['bb_lower'] = df_1h['close'].rolling(20).mean() - 2 * df_1h['close'].rolling(20).std()
        df_1h['bb_width'] = (df_1h['bb_upper'] - df_1h['bb_lower']) / df_1h['close']
        df_1h['bb_volatile'] = df_1h['bb_width'] > self.bb_width_threshold
        
        # 合并多时间框架数据
        df = df.merge(df_4h[['timestamp', 'trend_4h', 'atr_pct_4h']], on='timestamp', how='left')
        df = df.merge(df_1h[['timestamp', 'trend_1h', 'atr_pct_1h', 'bb_volatile']], on='timestamp', how='left')
        
        # 前向填充缺失值
        for col in ['trend_4h', 'atr_pct_4h', 'trend_1h', 'atr_pct_1h', 'bb_volatile']:
            df[col] = df[col].fillna(method='ffill')
        
        # 15分钟指标 - 触发信号
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume指标
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_spike'] = df['volume'] > df['volume_ma'] * self.volume_multiplier
        
        # K线突破
        df['k_break'] = (df['close'] > df['high'].shift(1)) & (df['volume'] > df['volume_ma'] * self.volume_multiplier)
        
        # 趋势指标
        df['ema5'] = df['close'].ewm(span=5).mean()
        df['ema10'] = df['close'].ewm(span=10).mean()
        df['trend_15m'] = df['ema5'] > df['ema10']
        
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
        
        # 市场状态检测
        df['market_state'] = 'range'
        df.loc[df['trend_1h'] & (df['RSI'] > 50), 'market_state'] = 'bull'
        df.loc[~df['trend_1h'] & (df['RSI'] < 45), 'market_state'] = 'bear'
        
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
        
        # Funding Rate（Mock）
        np.random.seed(42)
        df['funding_rate'] = np.random.normal(0.01, 0.02, len(df))
        df['funding_signal'] = df['funding_rate'] < 0.01
        
        return df
    
    def calculate_signal_strength(self, row):
        """计算信号强度 - 10.0增强"""
        score = 0
        
        # 1H/4H框架条件
        if row['trend_1h'] and row['trend_4h']:
            score += 0.3  # 多时间框架趋势一致
        
        if row['atr_pct_1h'] > self.atr_threshold_1h:
            score += 0.2  # 1H ATR > 0.6%
        
        if row['bb_volatile']:
            score += 0.1  # BB宽度 > 4%
        
        # 15m触发条件
        if row['k_break']:
            score += 0.2  # K线突破 + volume spike
        
        if row['RSI'] < self.rsi_bull_threshold:
            score += 0.1  # RSI < 50
        
        # Etherscan信号
        if row['w1_signal']:
            score += 0.1  # W1 > 8000 + Z-score > 2
        
        return score
    
    def calculate_position_size(self, row, atr_1h):
        """计算仓位大小 - 动态仓位"""
        # 根据市场状态调整风险
        if row['market_state'] == 'bull':
            risk_cap = self.capital * self.risk_per_trade_bull
        else:
            risk_cap = self.capital * self.risk_per_trade_range
        
        stop_loss = atr_1h * self.stop_loss_multiplier * row['close']
        position_size = risk_cap / stop_loss
        position_size = min(position_size, self.max_position_size)
        
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
        """回测 - 10.0优化"""
        print("Starting RexKing ETH 10.0 backtest...")
        
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
                signal_strength = self.calculate_signal_strength(row)
                
                if (signal_strength > self.signal_strength_threshold and 
                    row['atr_pct_1h'] > self.atr_threshold_1h):
                    
                    # 计算仓位
                    position_size = self.calculate_position_size(row, row['atr_pct_1h'])
                    
                    # 开仓
                    position = {
                        'entry_time': row['timestamp'],
                        'entry_price': row['close'],
                        'position_size': position_size,
                        'atr_1h': row['atr_pct_1h'],
                        'market_state': row['market_state'],
                        'signal_strength': signal_strength,
                        'max_price': row['close'],
                        'bars_held': 0
                    }
                    
                    # 根据市场状态设置止盈止损
                    if row['market_state'] == 'bull':
                        # 牛市：TP1 2×ATR，TP2 4×ATR，trailing SL 4×ATR
                        position['tp1'] = row['close'] * (1 + self.tp1_multiplier_bull * row['atr_pct_1h'])
                        position['tp2'] = row['close'] * (1 + self.tp2_multiplier * row['atr_pct_1h'])
                        position['trailing_sl'] = row['close'] * (1 - self.stop_loss_multiplier * row['atr_pct_1h'])
                    else:
                        # 震荡市：TP1 1.5×ATR，SL 0.8×ATR
                        position['tp1'] = row['close'] * (1 + self.tp1_multiplier_range * row['atr_pct_1h'])
                        position['tp2'] = row['close'] * (1 + self.tp2_multiplier * row['atr_pct_1h'])
                        position['trailing_sl'] = row['close'] * (1 - self.stop_loss_multiplier * row['atr_pct_1h'])
                    
                    self.positions.append(position)
                    self.daily_trade_count += 1
        
        # 平仓所有持仓
        for pos in self.positions:
            self.close_position(pos, df.iloc[-1]['close'], 'end_of_data')
        
        self.analyze_results()
    
    def update_positions(self, row):
        """更新持仓 - 10.0优化"""
        positions_to_close = []
        
        for i, pos in enumerate(self.positions):
            pos['bars_held'] += 1
            pos['max_price'] = max(pos['max_price'], row['high'])
            
            # 更新移动止损
            new_trailing_sl = pos['max_price'] * (1 - self.trailing_stop_multiplier * pos['atr_1h'])
            pos['trailing_sl'] = max(pos['trailing_sl'], new_trailing_sl)
            
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
        """平仓 - 10.0优化"""
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
        """分析结果 - 10.0优化"""
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
        
        # 按市场状态分析
        market_analysis = df_trades.groupby('market_state').agg({
            'pnl': ['count', 'mean', 'sum'],
            'exit_type': lambda x: (x == 'tp1').sum() + (x == 'tp2').sum()
        }).round(4)
        
        # 输出结果
        print("\n" + "="*60)
        print("RexKing ETH 10.0 Strategy Results")
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
        
        print(f"\nMarket State Analysis:")
        print(market_analysis)
        
        # 保存交易记录
        df_trades.to_csv('rexking_eth_10_0_trades.csv', index=False)
        print(f"\nTrades saved to rexking_eth_10_0_trades.csv")

def main():
    # 初始化策略
    strategy = RexKingETH100Strategy(capital=10000)
    
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
    strategy = RexKingETH100Strategy(capital=10000)
    
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