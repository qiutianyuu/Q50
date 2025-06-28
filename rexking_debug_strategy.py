import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 交易成本配置
FEE_RATE = 0.0004  # Binance taker fee 0.04%
SLIPPAGE = 0.0002  # 0.02% 滑点
TOTAL_COST = FEE_RATE * 2 + SLIPPAGE  # 双向交易总成本

class RexKingDebugStrategy:
    def __init__(self):
        self.name = "RexKing Debug Strategy"
        self.position = 0
        self.trades = []
        self.current_price = None
        self.entry_price = None
        self.entry_time = None
        
    def load_data(self, month, year):
        """加载指定月份的数据"""
        # 5分钟数据路径（特殊格式）
        data_path = f"/Users/qiutianyu/ETHUSDT-5m-{year}-{month:02d}/ETHUSDT-5m-{year}-{month:02d}.csv"
        
        try:
            # 定义列名
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                      'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                      'taker_buy_quote', 'ignore']
            
            df = pd.read_csv(data_path, header=None, names=columns)
            print(f"成功加载数据: {data_path}")
            print(f"数据形状: {df.shape}")
            print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
            return df
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None
    
    def calculate_indicators(self, df):
        """计算技术指标"""
        # 基础价格数据
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # 1. RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 2. MACD (12, 26, 9)
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 3. Bollinger Bands (20, 2)
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # 4. Stochastic (14, 3)
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # 5. ATR (14)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # 6. Volume OBV
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # 7. 动态活力指标 (Dynamic Vitality)
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['vitality'] = df['price_change'] * df['volume_change']
        df['vitality_ma'] = df['vitality'].rolling(window=10).mean()
        
        # 8. 价格动量
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)
        
        # 9. 波动率
        df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
        
        # 10. 趋势强度
        df['trend_strength'] = abs(df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
        
        return df
    
    def detect_regime(self, df):
        """检测市场状态"""
        # 简化的状态检测
        df['regime'] = 'normal'
        
        # 高波动状态
        high_vol_mask = df['volatility'] > df['volatility'].rolling(window=50).quantile(0.8)
        df.loc[high_vol_mask, 'regime'] = 'high_volatility'
        
        # 趋势状态
        trend_mask = (df['trend_strength'] > 1.5) & (df['momentum_20'] > 0.02)
        df.loc[trend_mask, 'regime'] = 'trending'
        
        # 震荡状态
        range_mask = (df['bb_width'] < df['bb_width'].rolling(window=50).quantile(0.3)) & (df['volatility'] < df['volatility'].rolling(window=50).quantile(0.3))
        df.loc[range_mask, 'regime'] = 'ranging'
        
        return df
    
    def generate_signals(self, df):
        """生成交易信号"""
        df['signal'] = 0
        # 预先计算rolling均值序列
        vol_ma = df['volume'].rolling(window=20).mean()
        vitality_ma = df['vitality'].rolling(window=10).mean()
        for i in range(50, len(df)):
            row = df.iloc[i]
            # 多头信号条件
            long_conditions = [
                row['rsi'] < 30,  # RSI超卖
                row['macd'] > row['macd_signal'],  # MACD金叉
                row['close'] < row['bb_lower'],  # 价格触及布林带下轨
                row['stoch_k'] < 20,  # 随机指标超卖
                row['vitality'] > vitality_ma.iloc[i],  # 活力指标上升
                row['momentum_5'] > 0,  # 短期动量为正
                row['volume'] > vol_ma.iloc[i]  # 成交量放大
            ]
            # 空头信号条件
            short_conditions = [
                row['rsi'] > 70,  # RSI超买
                row['macd'] < row['macd_signal'],  # MACD死叉
                row['close'] > row['bb_upper'],  # 价格触及布林带上轨
                row['stoch_k'] > 80,  # 随机指标超买
                row['vitality'] < vitality_ma.iloc[i],  # 活力指标下降
                row['momentum_5'] < 0,  # 短期动量为负
                row['volume'] > vol_ma.iloc[i]  # 成交量放大
            ]
            long_score = sum(long_conditions)
            short_score = sum(short_conditions)
            if long_score >= 4:
                df.loc[df.index[i], 'signal'] = 1
                print(f"多头信号 - 时间: {row['timestamp']}, 价格: {row['close']:.2f}, 信号强度: {long_score}")
                print(f"  条件详情: RSI<30:{long_conditions[0]}, MACD金叉:{long_conditions[1]}, BB下轨:{long_conditions[2]}, Stoch<20:{long_conditions[3]}, 活力上升:{long_conditions[4]}, 动量>0:{long_conditions[5]}, 量放大:{long_conditions[6]}")
            elif short_score >= 4:
                df.loc[df.index[i], 'signal'] = -1
                print(f"空头信号 - 时间: {row['timestamp']}, 价格: {row['close']:.2f}, 信号强度: {short_score}")
                print(f"  条件详情: RSI>70:{short_conditions[0]}, MACD死叉:{short_conditions[1]}, BB上轨:{short_conditions[2]}, Stoch>80:{short_conditions[3]}, 活力下降:{short_conditions[4]}, 动量<0:{short_conditions[5]}, 量放大:{short_conditions[6]}")
        return df
    
    def backtest(self, df, initial_capital=50000):
        """回测策略"""
        print(f"\n开始回测 {self.name}")
        print(f"初始资金: ${initial_capital:,.2f}")
        print(f"数据点数量: {len(df)}")
        
        capital = initial_capital
        position = 0
        entry_price = 0
        entry_time = None
        trades = []
        
        # 过滤掉NaN值
        df = df.dropna()
        print(f"过滤后数据点数量: {len(df)}")
        
        signal_count = 0
        
        for i, row in df.iterrows():
            if pd.isna(row['signal']) or row['signal'] == 0:
                continue
                
            signal_count += 1
            current_price = row['close']
            current_time = row['timestamp']
            
            # 开仓逻辑
            if position == 0:
                if row['signal'] == 1:  # 多头信号
                    position = 1
                    entry_price = current_price
                    entry_time = current_time
                    print(f"开多仓 - 时间: {current_time}, 价格: ${current_price:.2f}")
                    
                elif row['signal'] == -1:  # 空头信号
                    position = -1
                    entry_price = current_price
                    entry_time = current_time
                    print(f"开空仓 - 时间: {current_time}, 价格: ${current_price:.2f}")
            
            # 平仓逻辑
            elif position != 0:
                # 多头平仓条件
                if position == 1 and row['signal'] == -1:
                    pnl = (current_price - entry_price) / entry_price
                    capital *= (1 + pnl)
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': 'long',
                        'pnl': pnl,
                        'capital': capital
                    })
                    print(f"平多仓 - 时间: {current_time}, 价格: ${current_price:.2f}, 收益率: {pnl:.4f}, 资金: ${capital:,.2f}")
                    position = 0
                    entry_price = 0
                    entry_time = None
                
                # 空头平仓条件
                elif position == -1 and row['signal'] == 1:
                    pnl = (entry_price - current_price) / entry_price
                    capital *= (1 + pnl)
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': 'short',
                        'pnl': pnl,
                        'capital': capital
                    })
                    print(f"平空仓 - 时间: {current_time}, 价格: ${current_price:.2f}, 收益率: {pnl:.4f}, 资金: ${capital:,.2f}")
                    position = 0
                    entry_price = 0
                    entry_time = None
        
        # 计算统计指标
        total_return = (capital - initial_capital) / initial_capital
        num_trades = len(trades)
        
        if num_trades > 0:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / num_trades
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in trades) else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
        
        print(f"\n=== 回测结果 ===")
        print(f"信号数量: {signal_count}")
        print(f"交易次数: {num_trades}")
        print(f"最终资金: ${capital:,.2f}")
        print(f"总收益率: {total_return:.4f} ({total_return*100:.2f}%)")
        print(f"胜率: {win_rate:.4f} ({win_rate*100:.2f}%)")
        if num_trades > 0:
            print(f"平均盈利: {avg_win:.4f} ({avg_win*100:.2f}%)")
            print(f"平均亏损: {avg_loss:.4f} ({avg_loss*100:.2f}%)")
        
        return {
            'trades': trades,
            'final_capital': capital,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }

def main():
    strategy = RexKingDebugStrategy()
    
    # 测试5月份数据
    print("=== 测试5月份数据 ===")
    df_may = strategy.load_data(5, 2025)
    if df_may is not None:
        df_may = strategy.calculate_indicators(df_may)
        df_may = strategy.detect_regime(df_may)
        df_may = strategy.generate_signals(df_may)
        results_may = strategy.backtest(df_may)
        
        # 保存交易记录
        if results_may['trades']:
            trades_df = pd.DataFrame(results_may['trades'])
            trades_df.to_csv('rexking_debug_trades_2025_05.csv', index=False)
            print(f"交易记录已保存到: rexking_debug_trades_2025_05.csv")

if __name__ == "__main__":
    main() 