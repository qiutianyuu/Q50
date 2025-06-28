import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RexKingETH50Strategy:
    def __init__(self, initial_capital=1000, base_position_size=0.1):
        """
        RexKing ETH 5.0 Strategy - 基于第一性原理优化
        核心优化：胜率提升、回撤控制、信号密度、盈亏比优化
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.base_position_size = base_position_size
        self.trades = []
        self.positions = []
        self.daily_pnl = 0
        self.daily_trades = 0
        self.last_trade_date = None
        
        # 优化后的信号阈值参数
        self.bull_market_rsi_threshold = 45  # 牛市RSI阈值（放宽）
        self.oscillating_rsi_threshold = 35   # 震荡市RSI阈值（放宽）
        self.volume_multiplier = 1.5          # 成交量倍数（降低）
        self.signal_strength_threshold = 0.15 # 信号强度阈值
        
        # 优化后的风险控制参数
        self.netflow_threshold = 25000        # NetFlow阈值
        self.sentiment_threshold = 0.005      # 情绪阈值
        self.max_daily_loss = -200            # 日最大亏损（放宽）
        self.max_daily_trades = 80            # 日最大交易数（提高）
        
        # 优化后的止损止盈参数
        self.tp_multiplier = 2.5              # 止盈倍数（提高）
        self.sl_multiplier_bull = 1.2         # 牛市止损倍数
        self.sl_multiplier_osc = 1.5          # 震荡市止损倍数
        
        # 市场状态检测参数
        self.market_regime = 'oscillating'
        self.volatility_threshold = 0.015     # 波动率阈值
        self.trend_strength_threshold = 0.6   # 趋势强度阈值
        
        # 暂停机制
        self.pause_until = None
        self.pause_reason = None
        
        # 半Kelly仓位管理
        self.kelly_fraction = 0.5
        self.min_position_size = 0.05
        self.max_position_size = 0.15
        
    def calculate_indicators(self, df):
        """计算核心指标 - 5.0优化版"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        df['macd_golden_cross'] = (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))
        
        # 成交量指标 - 优化
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > df['volume_ma'] * self.volume_multiplier
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 动量指标
        df['momentum'] = df['close'].pct_change(3)
        df['momentum_ma'] = df['momentum'].rolling(window=10).mean()
        
        # ATR - 优化计算
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # 波动率 - 优化
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        df['volatility_ma'] = df['volatility'].rolling(window=10).mean()
        
        # 趋势强度 - 优化
        df['ema5'] = df['close'].ewm(span=5).mean()
        df['ema10'] = df['close'].ewm(span=10).mean()
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['trend_strength'] = (df['ema10'] - df['ema50']) / df['ema50']
        
        # 价格位置
        df['price_position'] = (df['close'] - df['close'].rolling(window=20).min()) / \
                              (df['close'].rolling(window=20).max() - df['close'].rolling(window=20).min())
        
        # 模拟NetFlow和Sentiment数据 - 更真实
        np.random.seed(42)
        # NetFlow: 更真实的分布
        df['NetFlow'] = np.random.normal(8000, 5000, len(df))
        df['NetFlow'] = np.where(df['NetFlow'] < 0, df['NetFlow'] * 0.5, df['NetFlow'])
        
        # Sentiment: 更真实的分布
        df['Sentiment'] = np.random.beta(2, 5, len(df)) * 0.15 + 0.01
        
        return df
    
    def calculate_15m_trend(self, df_5m, df_15m):
        """计算15分钟趋势确认 - 5.0优化版"""
        # 15分钟EMA - 优化为EMA10>EMA50
        df_15m['ema10'] = df_15m['close'].ewm(span=10).mean()
        df_15m['ema50'] = df_15m['close'].ewm(span=50).mean()
        df_15m['trend'] = df_15m['ema10'] > df_15m['ema50']
        df_15m['trend_strength'] = (df_15m['ema10'] - df_15m['ema50']) / df_15m['ema50']
        
        # 将15分钟趋势映射到5分钟数据
        trend_map = {}
        strength_map = {}
        for i, row in df_15m.iterrows():
            start_time = row.name
            end_time = start_time + timedelta(minutes=15)
            trend_map[start_time] = row['trend']
            strength_map[start_time] = row['trend_strength']
        
        df_5m['trend_15m'] = False
        df_5m['trend_strength_15m'] = 0.0
        for i, row in df_5m.iterrows():
            # 找到对应的15分钟趋势
            for start_time, trend in trend_map.items():
                if start_time <= i < start_time + timedelta(minutes=15):
                    df_5m.loc[i, 'trend_15m'] = trend
                    df_5m.loc[i, 'trend_strength_15m'] = strength_map[start_time]
                    break
        
        return df_5m
    
    def detect_market_regime(self, df):
        """检测市场状态 - 5.0优化版"""
        if len(df) < 20:
            return 'oscillating'
        
        recent_data = df.iloc[-20:]
        
        # 多因子市场状态判断 - 优化
        volatility_score = recent_data['volatility'].mean()
        trend_score = recent_data['trend_strength'].mean()
        momentum_score = recent_data['momentum_ma'].mean()
        price_position_score = recent_data['price_position'].mean()
        macd_score = recent_data['macd_golden_cross'].sum() / len(recent_data)
        
        # 综合评分 - 优化权重
        bull_score = 0
        if volatility_score > self.volatility_threshold:
            bull_score += 1
        if trend_score > 0.005:  # 上升趋势（降低阈值）
            bull_score += 1
        if momentum_score > 0.0005:  # 正动量（降低阈值）
            bull_score += 1
        if price_position_score > 0.5:  # 价格在中高位（降低阈值）
            bull_score += 1
        if macd_score > 0.1:  # MACD金叉频率
            bull_score += 1
        
        # 判断市场状态 - 降低阈值
        if bull_score >= 3:
            self.market_regime = 'bull'
            return 'bull'
        else:
            self.market_regime = 'oscillating'
            return 'oscillating'
    
    def calculate_signal_strength(self, row):
        """计算信号强度 - 5.0优化版"""
        # RSI权重 - 动态调整
        rsi_score = 0
        if self.market_regime == 'bull':
            if row['RSI'] < self.bull_market_rsi_threshold:
                rsi_score = (self.bull_market_rsi_threshold - row['RSI']) / self.bull_market_rsi_threshold
        else:
            if row['RSI'] < self.oscillating_rsi_threshold:
                rsi_score = (self.oscillating_rsi_threshold - row['RSI']) / self.oscillating_rsi_threshold
        
        # 成交量权重 - 优化
        volume_score = 0
        if row['volume_spike']:
            volume_score = min((row['volume_ratio'] - 1), 3) / 3
        
        # MACD权重 - 新增
        macd_score = 0
        if row['macd_golden_cross']:
            macd_score = 1.0
        elif row['MACD'] > row['MACD_signal']:
            macd_score = 0.5
        
        # 趋势确认权重 - 优化
        trend_score = 0
        if row['trend_15m']:
            trend_score = min(abs(row['trend_strength_15m']) * 10, 1.0)
        
        # 动量权重
        momentum_score = 0
        if row['momentum'] > 0:
            momentum_score = min(row['momentum'] * 100, 1.0)
        
        # 综合信号强度 - 优化权重分配
        signal_strength = (
            rsi_score * 0.25 + 
            volume_score * 0.20 + 
            macd_score * 0.20 + 
            trend_score * 0.20 + 
            momentum_score * 0.15
        )
        
        return signal_strength
    
    def calculate_kelly_position_size(self, win_rate, avg_win, avg_loss):
        """计算半Kelly仓位大小"""
        if avg_loss == 0:
            return self.base_position_size
        
        # Kelly公式
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # 半Kelly + 限制
        kelly_fraction = max(0, kelly_fraction * self.kelly_fraction)
        
        # 转换为仓位大小
        position_size = self.base_position_size * (1 + kelly_fraction)
        
        # 限制在合理范围内
        position_size = max(self.min_position_size, min(self.max_position_size, position_size))
        
        return position_size
    
    def check_risk_conditions(self, row, current_time):
        """检查风险条件 - 5.0优化版"""
        # 暂停机制检查
        if self.pause_until and current_time < self.pause_until:
            return False, f"Paused until {self.pause_until}: {self.pause_reason}"
        
        # NetFlow检查 - 动态阈值
        netflow_threshold = self.netflow_threshold
        if self.market_regime == 'bull':
            netflow_threshold *= 1.5  # 牛市放宽阈值
        
        if row['NetFlow'] > netflow_threshold:
            self.pause_until = current_time + timedelta(hours=24)
            self.pause_reason = f"NetFlow too high: {row['NetFlow']:.0f}"
            return False, f"NetFlow too high: {row['NetFlow']:.0f}"
        
        # Sentiment检查 - 动态阈值
        sentiment_threshold = self.sentiment_threshold
        if self.market_regime == 'bull':
            sentiment_threshold *= 0.8  # 牛市放宽阈值
        
        if row['Sentiment'] < sentiment_threshold:
            self.pause_until = current_time + timedelta(hours=6)
            self.pause_reason = f"Sentiment too low: {row['Sentiment']:.4f}"
            return False, f"Sentiment too low: {row['Sentiment']:.4f}"
        
        # 日亏损检查
        if self.daily_pnl < self.max_daily_loss:
            return False, f"Daily loss limit reached: {self.daily_pnl:.2f}"
        
        # 日交易数检查
        if self.daily_trades >= self.max_daily_trades:
            return False, f"Daily trade limit reached: {self.daily_trades}"
        
        return True, "Risk check passed"
    
    def should_trade(self, df, i):
        """判断是否应该交易 - 5.0优化版"""
        if i < 50:  # 需要足够的历史数据（增加）
            return False, "Insufficient data"
        
        row = df.iloc[i]
        current_time = row.name
        
        # 风险检查
        risk_ok, risk_msg = self.check_risk_conditions(row, current_time)
        if not risk_ok:
            return False, risk_msg
        
        # 检测市场状态
        self.detect_market_regime(df.iloc[:i+1])
        
        # 计算信号强度
        signal_strength = self.calculate_signal_strength(row)
        
        # 核心交易条件 - 5.0优化
        rsi_condition = False
        if self.market_regime == 'bull':
            rsi_condition = row['RSI'] < self.bull_market_rsi_threshold
        else:
            rsi_condition = row['RSI'] < self.oscillating_rsi_threshold
        
        # 信号条件 - 优化逻辑
        volume_condition = row['volume_spike']
        macd_condition = row['macd_golden_cross'] or row['MACD'] > row['MACD_signal']
        trend_condition = row['trend_15m']
        strength_condition = signal_strength > self.signal_strength_threshold
        
        # 价格位置条件
        price_condition = row['price_position'] < 0.85  # 不在极高位
        
        # 牛市：放宽条件，仅需RSI+（成交量或MACD）
        if self.market_regime == 'bull':
            signal_condition = (volume_condition or macd_condition)
        else:
            # 震荡市：需要更严格的条件
            signal_condition = (volume_condition or macd_condition) and trend_condition
        
        if (rsi_condition and signal_condition and strength_condition and price_condition):
            return True, f"Signal triggered - RSI: {row['RSI']:.1f}, Volume: {volume_condition}, MACD: {macd_condition}, Trend: {trend_condition}, Strength: {signal_strength:.3f}"
        
        return False, f"No signal - RSI: {row['RSI']:.1f}, Volume: {volume_condition}, MACD: {macd_condition}, Trend: {trend_condition}, Strength: {signal_strength:.3f}"
    
    def execute_trade(self, df, i, trade_type='long'):
        """执行交易 - 5.0优化版"""
        row = df.iloc[i]
        entry_price = row['close']
        
        # 计算止损止盈 - 动态调整
        atr = row['ATR']
        if pd.isna(atr) or atr == 0:
            atr = entry_price * 0.01  # 默认1%
        
        # 根据市场状态动态调整
        if self.market_regime == 'bull':
            tp_distance = atr * self.tp_multiplier * 1.3  # 牛市增加止盈
            sl_distance = atr * self.sl_multiplier_bull  # 牛市止损
        else:
            tp_distance = atr * self.tp_multiplier
            sl_distance = atr * self.sl_multiplier_osc  # 震荡市止损
        
        # 确保最小盈亏比
        if tp_distance / sl_distance < 2.0:
            tp_distance = sl_distance * 2.0
        
        tp_price = entry_price + tp_distance
        sl_price = entry_price - sl_distance
        
        # 计算动态仓位大小 - 半Kelly
        if len(self.trades) >= 10:
            # 计算历史胜率和盈亏
            recent_trades = self.trades[-10:]
            winning_trades = [t for t in recent_trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(recent_trades)
            
            if winning_trades:
                avg_win = np.mean([t['pnl'] for t in winning_trades])
                losing_trades = [t for t in recent_trades if t['pnl'] < 0]
                avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 1.0
                
                position_size = self.calculate_kelly_position_size(win_rate, avg_win, avg_loss)
            else:
                position_size = self.base_position_size
        else:
            position_size = self.base_position_size
        
        # 根据市场状态调整仓位
        if self.market_regime == 'bull':
            position_size = min(position_size, self.capital * 0.35 / entry_price)  # 牛市增加仓位
        else:
            position_size = min(position_size, self.capital * 0.25 / entry_price)
        
        # 记录交易
        trade = {
            'entry_time': row.name,
            'entry_price': entry_price,
            'position_size': position_size,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'market_regime': self.market_regime,
            'signal_strength': self.calculate_signal_strength(row),
            'rsi': row['RSI'],
            'volume_ratio': row['volume_ratio'],
            'atr': atr,
            'trend_strength': row['trend_strength_15m'],
            'macd_signal': row['macd_golden_cross']
        }
        
        self.positions.append(trade)
        self.daily_trades += 1
        
        print(f"🔄 Trade opened: {trade_type.upper()} {position_size:.3f} ETH @ ${entry_price:.2f}")
        print(f"   TP: ${tp_price:.2f} (+{tp_distance/entry_price*100:.1f}%), SL: ${sl_price:.2f} (-{sl_distance/entry_price*100:.1f}%)")
        print(f"   Market: {self.market_regime}, Strength: {trade['signal_strength']:.3f}")
        
        return trade
    
    def check_exit_conditions(self, df, i):
        """检查退出条件 - 5.0优化版"""
        current_price = df.iloc[i]['close']
        current_time = df.iloc[i].name
        
        closed_positions = []
        
        for pos in self.positions[:]:  # 复制列表避免修改迭代
            # 检查止损止盈
            if current_price >= pos['tp_price']:
                # 止盈
                pnl = (pos['tp_price'] - pos['entry_price']) * pos['position_size']
                self.capital += pnl
                self.daily_pnl += pnl
                
                trade_result = {
                    'entry_time': pos['entry_time'],
                    'exit_time': current_time,
                    'entry_price': pos['entry_price'],
                    'exit_price': pos['tp_price'],
                    'position_size': pos['position_size'],
                    'pnl': pnl,
                    'exit_type': 'take_profit',
                    'duration': (current_time - pos['entry_time']).total_seconds() / 3600,  # 小时
                    'market_regime': pos['market_regime']
                }
                
                self.trades.append(trade_result)
                self.positions.remove(pos)
                closed_positions.append(trade_result)
                
                print(f"✅ TP hit: +${pnl:.2f} ({pnl/pos['entry_price']/pos['position_size']*100:.1f}%)")
                
            elif current_price <= pos['sl_price']:
                # 止损
                pnl = (pos['sl_price'] - pos['entry_price']) * pos['position_size']
                self.capital += pnl
                self.daily_pnl += pnl
                
                trade_result = {
                    'entry_time': pos['entry_time'],
                    'exit_time': current_time,
                    'entry_price': pos['entry_price'],
                    'exit_price': pos['sl_price'],
                    'position_size': pos['position_size'],
                    'pnl': pnl,
                    'exit_type': 'stop_loss',
                    'duration': (current_time - pos['entry_time']).total_seconds() / 3600,
                    'market_regime': pos['market_regime']
                }
                
                self.trades.append(trade_result)
                self.positions.remove(pos)
                closed_positions.append(trade_result)
                
                print(f"❌ SL hit: ${pnl:.2f} ({pnl/pos['entry_price']/pos['position_size']*100:.1f}%)")
        
        return closed_positions
    
    def reset_daily_stats(self, current_date):
        """重置日统计"""
        if self.last_trade_date != current_date:
            self.daily_pnl = 0
            self.daily_trades = 0
            self.last_trade_date = current_date
            # 重置暂停状态
            if self.pause_until and current_date > self.pause_until.date():
                self.pause_until = None
                self.pause_reason = None
    
    def backtest(self, df_5m, df_15m):
        """回测策略 - 5.0优化版"""
        print("🚀 Starting RexKing ETH 5.0 Backtest...")
        print(f"Initial Capital: ${self.initial_capital}")
        print(f"Base Position Size: {self.base_position_size} ETH")
        print("=" * 60)
        
        # 计算指标
        df_5m = self.calculate_indicators(df_5m)
        df_5m = self.calculate_15m_trend(df_5m, df_15m)
        
        signals = 0
        trades_executed = 0
        
        for i in range(len(df_5m)):
            current_date = df_5m.index[i].date()
            self.reset_daily_stats(current_date)
            
            # 检查退出条件
            self.check_exit_conditions(df_5m, i)
            
            # 检查是否应该开仓
            should_trade, reason = self.should_trade(df_5m, i)
            
            if should_trade:
                signals += 1
                # 检查是否有足够资金
                if self.capital > 0:
                    self.execute_trade(df_5m, i, 'long')
                    trades_executed += 1
                else:
                    print(f"⚠️  Insufficient capital: ${self.capital:.2f}")
            
            # 每1000个数据点打印进度
            if i % 1000 == 0 and i > 0:
                print(f"Progress: {i}/{len(df_5m)} ({i/len(df_5m)*100:.1f}%)")
        
        # 强制平仓剩余仓位
        final_price = df_5m.iloc[-1]['close']
        for pos in self.positions:
            pnl = (final_price - pos['entry_price']) * pos['position_size']
            self.capital += pnl
            
            trade_result = {
                'entry_time': pos['entry_time'],
                'exit_time': df_5m.index[-1],
                'entry_price': pos['entry_price'],
                'exit_price': final_price,
                'position_size': pos['position_size'],
                'pnl': pnl,
                'exit_type': 'force_close',
                'duration': (df_5m.index[-1] - pos['entry_time']).total_seconds() / 3600,
                'market_regime': pos['market_regime']
            }
            self.trades.append(trade_result)
        
        # 计算回测结果
        self.calculate_results(signals, trades_executed)
        
        return self.trades
    
    def calculate_results(self, signals, trades_executed):
        """计算回测结果 - 5.0优化版"""
        if not self.trades:
            print("❌ No trades executed!")
            return
        
        df_trades = pd.DataFrame(self.trades)
        
        # 基础统计
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # 盈亏统计
        total_pnl = df_trades['pnl'].sum()
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
        
        # 最大回撤
        cumulative_pnl = df_trades['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # 年化收益率
        if len(df_trades) > 1:
            first_trade = df_trades['entry_time'].min()
            last_trade = df_trades['exit_time'].max()
            days = (last_trade - first_trade).days
            if days > 0:
                annual_return = (total_pnl / self.initial_capital) * (365 / days) * 100
            else:
                annual_return = 0
        else:
            annual_return = 0
        
        # 信号密度统计
        total_days = (df_trades['exit_time'].max() - df_trades['entry_time'].min()).days
        daily_signals = signals / total_days if total_days > 0 else 0
        signal_conversion_rate = trades_executed / signals * 100 if signals > 0 else 0
        
        # 打印结果
        print("\n" + "=" * 60)
        print("📊 REXKING ETH 5.0 BACKTEST RESULTS")
        print("=" * 60)
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"Final Capital: ${self.capital:.2f}")
        print(f"Total P&L: ${total_pnl:.2f} ({total_pnl/self.initial_capital*100:.2f}%)")
        print(f"Annual Return: {annual_return:.2f}%")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print("-" * 60)
        print(f"Total Signals: {signals}")
        print(f"Trades Executed: {trades_executed}")
        print(f"Signal Conversion Rate: {signal_conversion_rate:.1f}%")
        print(f"Daily Signal Density: {daily_signals:.1f} signals/day")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print("-" * 60)
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Risk-Reward Ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "Risk-Reward Ratio: N/A")
        print("-" * 60)
        
        # 按退出类型统计
        exit_stats = df_trades.groupby('exit_type').agg({
            'pnl': ['count', 'sum', 'mean'],
            'duration': 'mean'
        }).round(2)
        print("Exit Type Statistics:")
        print(exit_stats)
        print("-" * 60)
        
        # 按市场状态统计
        regime_stats = df_trades.groupby('market_regime').agg({
            'pnl': ['count', 'sum', 'mean'],
            'duration': 'mean'
        }).round(2)
        print("Market Regime Statistics:")
        print(regime_stats)
        print("=" * 60)
        
        # 保存交易记录
        df_trades.to_csv('rexking_eth_5_0_trades.csv', index=False)
        print("💾 Trade records saved to 'rexking_eth_5_0_trades.csv'")

def load_and_prepare_data():
    """加载和准备数据"""
    print("📊 Loading data...")
    
    # Binance 5m无表头数据列名
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    df_5m = pd.read_csv('/Users/qiutianyu/ETHUSDT-5m-2025-05/ETHUSDT-5m-2025-05.csv', names=columns)
    df_5m['open_time'] = pd.to_datetime(df_5m['open_time'] // 1000000, unit='s')
    df_5m.set_index('open_time', inplace=True)
    for col in ['open','high','low','close','volume']:
        df_5m[col] = df_5m[col].astype(float)
    
    # 创建15分钟数据（通过重采样）
    df_15m = df_5m.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    print(f"5m data: {len(df_5m)} records, {df_5m.index[0]} to {df_5m.index[-1]}")
    print(f"15m data: {len(df_15m)} records, {df_15m.index[0]} to {df_15m.index[-1]}")
    
    return df_5m, df_15m

if __name__ == "__main__":
    # 加载数据
    df_5m, df_15m = load_and_prepare_data()
    
    # 创建策略实例
    strategy = RexKingETH50Strategy(
        initial_capital=1000,
        base_position_size=0.1
    )
    
    # 运行回测
    trades = strategy.backtest(df_5m, df_15m) 