#!/usr/bin/env python3
"""
RexKing – Enhanced 15m Backtest (Fixed Version)

修复版本，解决以下问题：
1. 资金占用计算错误
2. 日损失限制比较值错误
3. 实际收益计算夸大
4. 回撤计算不准确
5. 追踪止盈逻辑缺陷
6. 多空仓位不对称
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ---------- 配置 ----------
DATA_DIR = Path("/Users/qiutianyu/data/processed")
FEATURES_FILE = DATA_DIR / "features_15m_2023_2025.parquet"
MODEL_FILE = Path("xgb_15m_optimized.bin")

# 回测参数
LONG_THRESHOLD = 0.60
SHORT_THRESHOLD = 0.40
TREND_FILTER = "4h"  # "1h", "4h", "both", "none"
STOP_LOSS = -0.01  # -1%
TRAILING_TP = 0.005  # 0.5%
MAX_POSITION_SIZE = 3.0  # 最大仓位倍数
MIN_POSITION_SIZE = 0.5  # 最小仓位倍数

# 风险控制参数
MAX_CONCURRENT_POSITIONS = 3  # 最大同时持仓数
MAX_DAILY_LOSS = -0.05  # -5% 日损失限制
MAX_DRAWDOWN = -0.15  # -15% 最大回撤限制
MAX_POSITION_VALUE_RATIO = 0.1  # 单笔持仓最大占权益比例 10%

def calculate_position_size(prob, direction='long'):
    """基于概率强度计算仓位大小，修复多空不对称问题"""
    if direction == 'long':
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
    else:  # short
        short_prob = 1 - prob
        if short_prob > 0.8:
            return MAX_POSITION_SIZE
        elif short_prob > 0.7:
            return 2.0
        elif short_prob > 0.65:
            return 1.5
        elif short_prob > 0.6:
            return 1.0
        else:
            return MIN_POSITION_SIZE

def check_trend_filter(row):
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

def calculate_position_value(position_size, entry_price, equity):
    """计算持仓价值，考虑资金限制"""
    position_value = position_size * entry_price
    max_allowed_value = equity * MAX_POSITION_VALUE_RATIO
    return min(position_value, max_allowed_value)

def main():
    print("=== RexKing Enhanced 15m Backtest (Fixed) ===")
    
    # 加载数据
    print("📥 加载特征数据...")
    df = pd.read_parquet(FEATURES_FILE)
    print(f"数据行数: {len(df)}")
    
    # 加载模型
    print("🤖 加载XGBoost模型...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    print("✅ 模型加载完成")
    
    # 准备特征
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'trend_1h', 'trend_4h']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    print(f"特征数量: {len(feature_cols)}")
    
    # 初始化回测变量
    positions = []
    trades = []
    equity = 10000
    max_equity = 10000
    daily_pnl = {}
    equity_curve = [10000]  # 记录权益曲线用于计算回撤
    
    print(f"🚀 开始修复版回测...")
    print(f"参数: 多空阈值={LONG_THRESHOLD}/{SHORT_THRESHOLD}, 趋势过滤={TREND_FILTER}")
    print(f"止损={STOP_LOSS*100}%, 追踪止盈={TRAILING_TP*100}%")
    print(f"风险控制: 最大持仓={MAX_CONCURRENT_POSITIONS}, 日损失限制={MAX_DAILY_LOSS*100}%")
    print(f"单笔持仓限制: {MAX_POSITION_VALUE_RATIO*100}%")
    
    for i, row in df.iterrows():
        current_time = row['timestamp']
        current_price = row['close']
        current_date = current_time.date()
        
        # 计算当前持仓的浮动盈亏
        current_equity = equity
        for position in positions:
            if position['status'] == 'open':
                if position['direction'] == 'long':
                    unrealized_pnl = (current_price - position['entry_price']) / position['entry_price']
                else:
                    unrealized_pnl = (position['entry_price'] - current_price) / position['entry_price']
                
                # 减去手续费
                unrealized_pnl -= 0.002
                current_equity += unrealized_pnl * position['position_value']
        
        # 更新权益曲线
        equity_curve.append(current_equity)
        max_equity = max(max_equity, current_equity)
        
        # 1. 检查现有持仓的退出条件
        for position in positions[:]:
            if position['status'] == 'open':
                # 更新追踪止盈 - 修复：锁定盈利
                if position['direction'] == 'long':
                    if current_price > position['entry_price'] * (1 + TRAILING_TP):
                        new_tp = current_price * (1 - TRAILING_TP * 0.5)
                        # 修复：确保追踪止盈不低于入场价
                        new_tp = max(new_tp, position['entry_price'])
                        if new_tp > position.get('trailing_tp', 0):
                            position['trailing_tp'] = new_tp
                else:  # short
                    if current_price < position['entry_price'] * (1 - TRAILING_TP):
                        new_tp = current_price * (1 + TRAILING_TP * 0.5)
                        # 修复：确保追踪止盈不高于入场价
                        new_tp = min(new_tp, position['entry_price'])
                        if new_tp < position.get('trailing_tp', float('inf')):
                            position['trailing_tp'] = new_tp
                
                # 检查退出条件
                exit_reason = None
                
                # 止损检查
                if position['direction'] == 'long':
                    if current_price <= position['entry_price'] * (1 + STOP_LOSS):
                        exit_reason = 'stop_loss'
                else:  # short
                    if current_price >= position['entry_price'] * (1 - STOP_LOSS):
                        exit_reason = 'stop_loss'
                
                # 追踪止盈检查
                if not exit_reason and 'trailing_tp' in position:
                    if position['direction'] == 'long' and current_price <= position['trailing_tp']:
                        exit_reason = 'trailing_tp'
                    elif position['direction'] == 'short' and current_price >= position['trailing_tp']:
                        exit_reason = 'trailing_tp'
                
                # 时间止损（4小时后）
                if not exit_reason and (current_time - position['entry_time']).total_seconds() > 4 * 3600:
                    exit_reason = 'time_stop'
                
                if exit_reason:
                    # 关闭持仓
                    if position['direction'] == 'long':
                        pnl = (current_price - position['entry_price']) / position['entry_price']
                    else:
                        pnl = (position['entry_price'] - current_price) / position['entry_price']
                    
                    # 减去手续费（0.1%）
                    pnl -= 0.002
                    
                    # 修复：计算实际收益 - 基于持仓价值而非名义价值
                    actual_pnl = pnl * position['position_value']
                    
                    # 记录交易
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'size': position['size'],
                        'position_value': position['position_value'],
                        'pnl': pnl,
                        'actual_pnl': actual_pnl,
                        'exit_reason': exit_reason
                    }
                    trades.append(trade)
                    
                    # 更新权益
                    equity += actual_pnl
                    
                    # 更新日收益
                    if current_date not in daily_pnl:
                        daily_pnl[current_date] = 0
                    daily_pnl[current_date] += actual_pnl
                    
                    # 修复：移除已平仓的持仓
                    positions.remove(position)
        
        # 2. 风险控制检查
        # 检查最大回撤
        current_drawdown = (current_equity - max_equity) / max_equity
        if current_drawdown < MAX_DRAWDOWN:
            continue  # 跳过开仓
        
        # 修复：检查日损失限制 - 使用当前权益而非固定值
        if current_date in daily_pnl and daily_pnl[current_date] < MAX_DAILY_LOSS * equity:
            continue  # 跳过开仓
        
        # 检查最大持仓数
        active_positions = len([p for p in positions if p['status'] == 'open'])
        if active_positions >= MAX_CONCURRENT_POSITIONS:
            continue  # 跳过开仓
        
        # 3. 生成信号
        features = row[feature_cols].values
        prob = model.predict_proba(features.reshape(1, -1))[0][1]
        
        # 4. 检查开仓条件
        if prob > LONG_THRESHOLD and check_trend_filter(row):
            # 多头信号
            position_size = calculate_position_size(prob, 'long')
            position_value = calculate_position_value(position_size, current_price, equity)
            
            position = {
                'entry_time': current_time,
                'entry_price': current_price,
                'direction': 'long',
                'size': position_size,
                'position_value': position_value,
                'status': 'open'
            }
            positions.append(position)
        
        elif prob < SHORT_THRESHOLD and check_trend_filter(row):
            # 空头信号
            position_size = calculate_position_size(prob, 'short')
            position_value = calculate_position_value(position_size, current_price, equity)
            
            position = {
                'entry_time': current_time,
                'entry_price': current_price,
                'direction': 'short',
                'size': position_size,
                'position_value': position_value,
                'status': 'open'
            }
            positions.append(position)
    
    # 5. 强制平仓所有剩余持仓
    for position in positions[:]:
        if position['direction'] == 'long':
            pnl = (df.iloc[-1]['close'] - position['entry_price']) / position['entry_price']
        else:
            pnl = (position['entry_price'] - df.iloc[-1]['close']) / position['entry_price']
        
        pnl -= 0.002
        actual_pnl = pnl * position['position_value']
        
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': df.iloc[-1]['timestamp'],
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': df.iloc[-1]['close'],
            'size': position['size'],
            'position_value': position['position_value'],
            'pnl': pnl,
            'actual_pnl': actual_pnl,
            'exit_reason': 'force_close'
        }
        trades.append(trade)
        
        equity += actual_pnl
        positions.remove(position)
    
    # 分析结果
    if trades:
        trades_df = pd.DataFrame(trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades
        
        total_return = (equity - 10000) / 10000
        avg_trade_return = trades_df['pnl'].mean()
        
        # 修复：计算真实最大回撤
        max_drawdown = 0
        peak = equity_curve[0]
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            drawdown = (eq - peak) / peak
            max_drawdown = min(max_drawdown, drawdown)
        
        # 计算年化收益
        days = (df.iloc[-1]['timestamp'] - df.iloc[0]['timestamp']).days
        annual_return = ((equity / 10000) ** (365 / days) - 1) * 100 if days > 0 else 0
        
        # 计算夏普比率
        daily_returns = pd.Series(daily_pnl).pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 and daily_returns.std() > 0 else 0
        
        print(f"\n📊 修复版回测结果:")
        print(f"总交易数: {total_trades}")
        print(f"胜率: {win_rate:.2%}")
        print(f"总收益: {total_return:.2%}")
        print(f"年化收益: {annual_return:.2f}%")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"平均单笔收益: {avg_trade_return:.2%}")
        print(f"最终权益: ${equity:,.2f}")
        
        # 保存交易记录
        trades_df.to_csv('enhanced_trades_15m_fixed.csv', index=False)
        print(f"✅ 交易记录已保存到: enhanced_trades_15m_fixed.csv")
        
        # 保存权益曲线
        if len(equity_curve) == len(df):
            equity_df = pd.DataFrame({
                'timestamp': df['timestamp'],
                'equity': equity_curve
            })
        else:
            # 如果长度不匹配，只保存权益曲线
            equity_df = pd.DataFrame({
                'equity': equity_curve
            })
        equity_df.to_csv('equity_curve_15m_fixed.csv', index=False)
        print(f"✅ 权益曲线已保存到: equity_curve_15m_fixed.csv")
    else:
        print("❌ 没有产生任何交易信号")
    
    return trades

if __name__ == "__main__":
    main() 