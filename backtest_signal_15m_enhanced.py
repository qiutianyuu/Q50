#!/usr/bin/env python3
"""
RexKing – Enhanced 15m Backtest with Adaptive Position Sizing & Trailing TP

增强版回测脚本，包含：
1. 自适应仓位大小（基于概率强度）
2. 动态止损止盈
3. 趋势过滤
4. 风险控制
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ---------- 配置 ----------
DATA_DIR = Path("/Users/qiutianyu/data/processed")
FEATURES_FILE = DATA_DIR / "features_15m_2023_2025.parquet"  # 使用原始特征文件
MODEL_FILE = Path("xgb_15m_optimized.bin")

# 回测参数
LONG_THRESHOLD = 0.60
SHORT_THRESHOLD = 0.40
TREND_FILTER = "4h"  # "1h", "4h", "both", "none"
STOP_LOSS = -0.01  # -1%
TRAILING_TP = 0.005  # 0.5%
MAX_POSITION_SIZE = 3.0  # 最大仓位倍数
MIN_POSITION_SIZE = 0.5  # 最小仓位倍数

# 风险控制
MAX_DAILY_LOSS = -0.05  # -5%
MAX_DRAWDOWN = -0.15  # -15%
MAX_POSITIONS = 3  # 最大同时持仓数

class EnhancedBacktest:
    def __init__(self, df, model, params):
        self.df = df.copy()
        self.model = model
        self.params = params
        self.positions = []
        self.trades = []
        self.daily_pnl = {}
        self.max_equity = 10000
        self.current_equity = 10000
        
    def calculate_position_size(self, prob):
        """基于概率强度计算仓位大小"""
        if prob > 0.8:
            return MAX_POSITION_SIZE
        elif prob > 0.7:
            return 2.0
        elif prob > 0.65:
            return 1.5
        elif prob > 0.6:
            return 1.0
        else:
            return MIN_POSITION_SIZE
    
    def check_trend_filter(self, row):
        """检查趋势过滤条件"""
        if TREND_FILTER == "none":
            return True
        elif TREND_FILTER == "1h":
            return row['trend_1h'] == 1
        elif TREND_FILTER == "4h":
            return row['trend_4h'] == 1
        elif TREND_FILTER == "both":
            return row['trend_1h'] == 1 and row['trend_4h'] == 1
        return True
    
    def check_risk_limits(self, new_position_size):
        """检查风险限制"""
        # 检查日损失限制
        today = pd.Timestamp.now().date()
        if today in self.daily_pnl and self.daily_pnl[today] < MAX_DAILY_LOSS:
            return False
        
        # 检查最大回撤
        current_drawdown = (self.current_equity - self.max_equity) / self.max_equity
        if current_drawdown < MAX_DRAWDOWN:
            return False
        
        # 检查最大持仓数
        active_positions = len([p for p in self.positions if p['status'] == 'open'])
        if active_positions >= MAX_POSITIONS:
            return False
        
        return True
    
    def update_trailing_stop(self, position, current_price):
        """更新追踪止盈"""
        if position['direction'] == 'long':
            if current_price > position['entry_price'] * (1 + TRAILING_TP):
                # 更新追踪止盈价格
                new_tp = current_price * (1 - TRAILING_TP * 0.5)
                if new_tp > position.get('trailing_tp', 0):
                    position['trailing_tp'] = new_tp
        else:  # short
            if current_price < position['entry_price'] * (1 - TRAILING_TP):
                # 更新追踪止盈价格
                new_tp = current_price * (1 + TRAILING_TP * 0.5)
                if new_tp < position.get('trailing_tp', float('inf')):
                    position['trailing_tp'] = new_tp
    
    def check_exit_conditions(self, position, current_price, current_time):
        """检查退出条件"""
        # 止损检查
        if position['direction'] == 'long':
            if current_price <= position['entry_price'] * (1 + STOP_LOSS):
                return 'stop_loss'
        else:  # short
            if current_price >= position['entry_price'] * (1 - STOP_LOSS):
                return 'stop_loss'
        
        # 追踪止盈检查
        if 'trailing_tp' in position:
            if position['direction'] == 'long' and current_price <= position['trailing_tp']:
                return 'trailing_tp'
            elif position['direction'] == 'short' and current_price >= position['trailing_tp']:
                return 'trailing_tp'
        
        # 时间止损（4小时后）
        if (current_time - position['entry_time']).total_seconds() > 4 * 3600:
            return 'time_stop'
        
        return None
    
    def close_position(self, position, exit_price, exit_time, exit_reason):
        """关闭持仓"""
        # 计算收益
        if position['direction'] == 'long':
            pnl = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl = (position['entry_price'] - exit_price) / position['entry_price']
        
        # 减去手续费（0.1%）
        pnl -= 0.002  # 0.1% * 2 (开仓+平仓)
        
        # 计算实际收益
        actual_pnl = pnl * position['size'] * position['entry_price']
        
        # 记录交易
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'size': position['size'],
            'pnl': pnl,
            'actual_pnl': actual_pnl,
            'exit_reason': exit_reason
        }
        self.trades.append(trade)
        
        # 更新权益
        self.current_equity += actual_pnl
        self.max_equity = max(self.max_equity, self.current_equity)
        
        # 更新日收益
        exit_date = exit_time.date()
        if exit_date not in self.daily_pnl:
            self.daily_pnl[exit_date] = 0
        self.daily_pnl[exit_date] += actual_pnl
        
        position['status'] = 'closed'
    
    def run_backtest(self):
        """运行回测"""
        print(f"🚀 开始增强回测...")
        print(f"参数: 多空阈值={LONG_THRESHOLD}/{SHORT_THRESHOLD}, 趋势过滤={TREND_FILTER}")
        print(f"止损={STOP_LOSS*100}%, 追踪止盈={TRAILING_TP*100}%")
        
        for i, row in self.df.iterrows():
            current_time = row['timestamp']
            current_price = row['close']
            
            # 1. 检查现有持仓的退出条件
            for position in self.positions[:]:  # 复制列表避免修改迭代
                if position['status'] == 'open':
                    # 更新追踪止盈
                    self.update_trailing_stop(position, current_price)
                    
                    # 检查退出条件
                    exit_reason = self.check_exit_conditions(position, current_price, current_time)
                    if exit_reason:
                        self.close_position(position, current_price, current_time, exit_reason)
            
            # 2. 生成信号 - 使用与训练时相同的特征集
            exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'trend_1h', 'trend_4h']
            feature_cols = [col for col in row.index if col not in exclude_cols and pd.api.types.is_numeric_dtype(row[col])]
            features = row[feature_cols].values
            
            # Debug: 检查特征数量
            if i == 0:
                print(f"特征数量: {len(feature_cols)}")
                print(f"特征列: {feature_cols[:10]}...")  # 显示前10个特征
            
            prob = self.model.predict_proba(features.reshape(1, -1))[0][1]
            
            # 3. 检查开仓条件
            if prob > LONG_THRESHOLD and self.check_trend_filter(row):
                # 多头信号
                position_size = self.calculate_position_size(prob)
                if self.check_risk_limits(position_size):
                    position = {
                        'entry_time': current_time,
                        'entry_price': current_price,
                        'direction': 'long',
                        'size': position_size,
                        'status': 'open'
                    }
                    self.positions.append(position)
            
            elif prob < SHORT_THRESHOLD and self.check_trend_filter(row):
                # 空头信号
                position_size = self.calculate_position_size(1 - prob)
                if self.check_risk_limits(position_size):
                    position = {
                        'entry_time': current_time,
                        'entry_price': current_price,
                        'direction': 'short',
                        'size': position_size,
                        'status': 'open'
                    }
                    self.positions.append(position)
        
        # 4. 强制平仓所有剩余持仓
        for position in self.positions:
            if position['status'] == 'open':
                self.close_position(position, self.df.iloc[-1]['close'], 
                                 self.df.iloc[-1]['timestamp'], 'force_close')
        
        return self.analyze_results()
    
    def analyze_results(self):
        """分析回测结果"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_trade_return': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # 基础统计
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades
        
        # 收益统计
        total_return = (self.current_equity - 10000) / 10000
        avg_trade_return = trades_df['pnl'].mean()
        
        # 计算日收益序列
        daily_returns = pd.Series(self.daily_pnl).pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # 计算最大回撤
        equity_curve = [10000]
        for trade in self.trades:
            equity_curve.append(equity_curve[-1] + trade['actual_pnl'])
        
        max_drawdown = 0
        peak = equity_curve[0]
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (equity - peak) / peak
            max_drawdown = min(max_drawdown, drawdown)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_trade_return': avg_trade_return,
            'final_equity': self.current_equity,
            'trades': self.trades
        }

def main():
    print("=== RexKing Enhanced 15m Backtest ===")
    
    # 加载数据
    print("📥 加载特征数据...")
    df = pd.read_parquet(FEATURES_FILE)
    print(f"数据行数: {len(df)}")
    
    # 加载模型
    print("🤖 加载XGBoost模型...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    print("✅ 模型加载完成")
    
    # 运行回测
    backtest = EnhancedBacktest(df, model, {
        'long_threshold': LONG_THRESHOLD,
        'short_threshold': SHORT_THRESHOLD,
        'trend_filter': TREND_FILTER,
        'stop_loss': STOP_LOSS,
        'trailing_tp': TRAILING_TP
    })
    
    results = backtest.run_backtest()
    
    # 输出结果
    print(f"\n📊 回测结果:")
    print(f"总交易数: {results['total_trades']}")
    print(f"胜率: {results['win_rate']:.2%}")
    print(f"总收益: {results['total_return']:.2%}")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    print(f"平均单笔收益: {results['avg_trade_return']:.2%}")
    print(f"最终权益: ${results['final_equity']:,.2f}")
    
    # 保存交易记录
    if results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv('enhanced_trades_15m.csv', index=False)
        print(f"✅ 交易记录已保存到: enhanced_trades_15m.csv")
    
    return results

if __name__ == "__main__":
    main() 