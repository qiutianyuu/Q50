import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RexKingMVPOptimized:
    def __init__(self):
        # 优化参数 - 基于分析结果调整
        self.rsi_period = 14
        self.rsi_oversold = 25  # 更严格的超卖
        self.rsi_overbought = 75  # 更严格的超买
        
        self.vwap_period = 20
        
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # 优化信号阈值 - 更严格
        self.long_threshold = 0.4  # 提高阈值，减少假信号
        self.short_threshold = -0.4
        
        # 优化风险管理
        self.stop_loss_pct = 0.015  # 1.5%止损，更严格
        self.take_profit_pct = 0.03  # 3%止盈，更保守
        self.max_position_size = 0.15  # 增加仓位到15%
        
        # 时间过滤 - 基于分析结果
        self.best_hours = [12, 13, 14, 15, 2, 3, 23]  # 最佳交易时段
        
        # 交易状态
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
        
    def calculate_indicators(self, df):
        """计算核心指标"""
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        
        # VWAP
        df['vwap'] = talib.SMA(df['close'], timeperiod=self.vwap_period)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], 
            fastperiod=self.macd_fast, 
            slowperiod=self.macd_slow, 
            signalperiod=self.macd_signal
        )
        
        # 价格动量
        df['price_momentum'] = df['close'].pct_change(3)
        
        # 成交量指标
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
        
        # 新增：价格波动率
        df['volatility'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()
        
        # 新增：趋势强度
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['trend_strength'] = (df['ema_20'] - df['ema_50']) / df['ema_50']
        
        return df
    
    def calculate_signal_score(self, row):
        """计算综合信号分数 - 优化版本"""
        score = 0
        
        # RSI信号 (权重: 0.25)
        if row['rsi'] < self.rsi_oversold:
            score += 0.25
        elif row['rsi'] > self.rsi_overbought:
            score -= 0.25
        
        # VWAP信号 (权重: 0.25)
        vwap_diff = (row['close'] - row['vwap']) / row['vwap']
        if vwap_diff > 0.005:  # 价格显著高于VWAP
            score += 0.25
        elif vwap_diff < -0.005:  # 价格显著低于VWAP
            score -= 0.25
        
        # MACD信号 (权重: 0.2)
        if row['macd'] > row['macd_signal'] and row['macd_hist'] > 0:
            score += 0.2
        elif row['macd'] < row['macd_signal'] and row['macd_hist'] < 0:
            score -= 0.2
        
        # 价格动量 (权重: 0.15)
        if row['price_momentum'] > 0.01:  # 1%以上动量
            score += 0.15
        elif row['price_momentum'] < -0.01:
            score -= 0.15
        
        # 成交量确认 (权重: 0.1)
        if row['volume_ratio'] > 1.3:
            score += 0.1
        elif row['volume_ratio'] < 0.7:
            score -= 0.1
        
        # 新增：趋势强度 (权重: 0.05)
        if abs(row['trend_strength']) > 0.02:  # 强趋势
            if row['trend_strength'] > 0:
                score += 0.05
            else:
                score -= 0.05
        
        return score
    
    def is_good_trading_hour(self, current_time):
        """检查是否在最佳交易时段"""
        hour = current_time.hour
        return hour in self.best_hours
    
    def should_exit(self, current_price, current_time):
        """检查是否应该平仓"""
        if self.position == 0:
            return False, ""
        
        # 止损检查
        if self.position == 1:  # 多头
            loss_pct = (current_price - self.entry_price) / self.entry_price
            if loss_pct <= -self.stop_loss_pct:
                return True, "止损"
            if loss_pct >= self.take_profit_pct:
                return True, "止盈"
        
        elif self.position == -1:  # 空头
            loss_pct = (self.entry_price - current_price) / self.entry_price
            if loss_pct <= -self.stop_loss_pct:
                return True, "止损"
            if loss_pct >= self.take_profit_pct:
                return True, "止盈"
        
        # 时间止损 (3小时，更短)
        if self.entry_time and (current_time - self.entry_time).total_seconds() > 3 * 3600:
            return True, "时间止损"
        
        return False, ""
    
    def backtest(self, data_file):
        """回测策略"""
        print(f"开始优化MVP策略回测: {data_file}")
        
        # 读取数据
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 计算指标
        df = self.calculate_indicators(df)
        
        # 初始化结果
        trades = []
        equity_curve = []
        initial_capital = 50000
        current_capital = initial_capital
        
        print(f"数据范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
        print(f"总数据点: {len(df)}")
        
        signal_count = 0
        trade_count = 0
        
        for i, row in df.iterrows():
            current_price = row['close']
            current_time = row['timestamp']
            
            # 检查平仓条件
            should_exit, exit_reason = self.should_exit(current_price, current_time)
            
            if should_exit and self.position != 0:
                # 计算收益
                if self.position == 1:  # 多头平仓
                    pnl = (current_price - self.entry_price) / self.entry_price
                else:  # 空头平仓
                    pnl = (self.entry_price - current_price) / self.entry_price
                
                trade_pnl = pnl * current_capital * self.max_position_size
                current_capital += trade_pnl
                
                trades.append({
                    'entry_time': self.entry_time,
                    'exit_time': current_time,
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'position': self.position,
                    'pnl': trade_pnl,
                    'pnl_pct': pnl,
                    'exit_reason': exit_reason
                })
                
                trade_count += 1
                print(f"平仓 {trade_count}: {exit_reason} | 价格: {current_price:.2f} | PnL: {trade_pnl:.2f} | 收益率: {pnl*100:.2f}%")
                
                self.position = 0
                self.entry_price = 0
                self.entry_time = None
            
            # 如果无仓位，检查开仓信号
            if self.position == 0 and i > 50:  # 确保有足够的历史数据
                # 时间过滤
                if not self.is_good_trading_hour(current_time):
                    continue
                
                signal_score = self.calculate_signal_score(row)
                
                # 记录信号用于调试
                if abs(signal_score) > 0.3:
                    signal_count += 1
                    if signal_count <= 5:  # 只打印前5个信号
                        print(f"信号 {signal_count}: 时间={current_time}, 分数={signal_score:.3f}, "
                              f"RSI={row['rsi']:.1f}, VWAP_diff={((row['close']-row['vwap'])/row['vwap']*100):.2f}%, "
                              f"MACD_diff={((row['macd']-row['macd_signal'])*100):.3f}%")
                
                # 开仓条件
                if signal_score > self.long_threshold:
                    self.position = 1
                    self.entry_price = current_price
                    self.entry_time = current_time
                    trade_count += 1
                    print(f"开多仓 {trade_count}: 价格={current_price:.2f}, 信号分数={signal_score:.3f}")
                
                elif signal_score < self.short_threshold:
                    self.position = -1
                    self.entry_price = current_price
                    self.entry_time = current_time
                    trade_count += 1
                    print(f"开空仓 {trade_count}: 价格={current_price:.2f}, 信号分数={signal_score:.3f}")
            
            # 记录权益曲线
            equity_curve.append({
                'timestamp': current_time,
                'equity': current_capital,
                'position': self.position
            })
        
        # 如果最后还有持仓，强制平仓
        if self.position != 0:
            final_price = df.iloc[-1]['close']
            if self.position == 1:
                pnl = (final_price - self.entry_price) / self.entry_price
            else:
                pnl = (self.entry_price - final_price) / self.entry_price
            
            trade_pnl = pnl * current_capital * self.max_position_size
            current_capital += trade_pnl
            
            trades.append({
                'entry_time': self.entry_time,
                'exit_time': df.iloc[-1]['timestamp'],
                'entry_price': self.entry_price,
                'exit_price': final_price,
                'position': self.position,
                'pnl': trade_pnl,
                'pnl_pct': pnl,
                'exit_reason': '强制平仓'
            })
        
        # 计算统计结果
        total_return = (current_capital - initial_capital) / initial_capital
        total_trades = len(trades)
        
        if total_trades > 0:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / total_trades
            
            total_pnl = sum(t['pnl'] for t in trades)
            avg_pnl = total_pnl / total_trades
            
            print(f"\n=== 优化MVP策略回测结果 ===")
            print(f"初始资金: ${initial_capital:,.2f}")
            print(f"最终资金: ${current_capital:,.2f}")
            print(f"总收益: ${current_capital - initial_capital:,.2f}")
            print(f"总收益率: {total_return*100:.2f}%")
            print(f"总交易次数: {total_trades}")
            print(f"胜率: {win_rate*100:.1f}%")
            print(f"平均每笔收益: ${avg_pnl:.2f}")
            print(f"信号触发次数: {signal_count}")
            
            # 保存交易记录
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv('rexking_mvp_optimized_trades.csv', index=False)
            print(f"交易记录已保存到: rexking_mvp_optimized_trades.csv")
            
        else:
            print(f"\n=== 优化MVP策略回测结果 ===")
            print(f"无交易产生")
            print(f"信号触发次数: {signal_count}")
        
        return trades, equity_curve

if __name__ == "__main__":
    strategy = RexKingMVPOptimized()
    
    # 回测4月数据
    trades, equity = strategy.backtest('data/ETHUSDT-1h-2025-04.csv') 