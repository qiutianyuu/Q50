#!/usr/bin/env python3
"""
RexKing – Optimized 15m Strategy

优化版策略，专注于：
1. 动态阈值调整
2. 改进的仓位管理
3. 信号去重和合并
4. 更精细的风险控制
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

# 优化参数
BASE_LONG_THRESHOLD = 0.60
BASE_SHORT_THRESHOLD = 0.40
TREND_FILTER = "4h"
STOP_LOSS = -0.01
TRAILING_TP = 0.005
MAX_POSITION_SIZE = 2.0  # 降低最大仓位
MIN_POSITION_SIZE = 0.5

# 风险控制
MAX_CONCURRENT_POSITIONS = 2  # 降低最大持仓数
MAX_DAILY_LOSS = -0.03  # 收紧日损失限制
MAX_DRAWDOWN = -0.10  # 收紧最大回撤
MAX_POSITION_VALUE_RATIO = 0.08  # 降低单笔持仓比例

# 动态参数
VOLATILITY_LOOKBACK = 48  # 12小时
CONFIDENCE_LOOKBACK = 20  # 5小时

def calculate_dynamic_thresholds(df, current_idx):
    """基于市场状态动态调整阈值"""
    if current_idx < VOLATILITY_LOOKBACK:
        return BASE_LONG_THRESHOLD, BASE_SHORT_THRESHOLD
    
    # 计算近期波动率 - 使用volatility_24列
    recent_volatility = df['volatility_24'].iloc[current_idx-VOLATILITY_LOOKBACK:current_idx].mean()
    avg_volatility = df['volatility_24'].iloc[:current_idx].mean()
    
    # 波动率调整因子
    vol_ratio = recent_volatility / avg_volatility if avg_volatility > 0 else 1.0
    
    # 高波动时放宽阈值，低波动时收紧阈值
    if vol_ratio > 1.2:  # 高波动
        long_threshold = BASE_LONG_THRESHOLD - 0.05
        short_threshold = BASE_SHORT_THRESHOLD + 0.05
    elif vol_ratio < 0.8:  # 低波动
        long_threshold = BASE_LONG_THRESHOLD + 0.05
        short_threshold = BASE_SHORT_THRESHOLD - 0.05
    else:
        long_threshold = BASE_LONG_THRESHOLD
        short_threshold = BASE_SHORT_THRESHOLD
    
    return max(0.55, min(0.75, long_threshold)), max(0.25, min(0.45, short_threshold))

def calculate_adaptive_position_size(prob, direction, recent_confidence):
    """基于概率和近期置信度计算自适应仓位"""
    if direction == 'long':
        base_size = calculate_base_position_size(prob)
    else:
        base_size = calculate_base_position_size(1 - prob)
    
    # 根据近期置信度调整
    if recent_confidence > 0.7:
        size_multiplier = 1.2
    elif recent_confidence < 0.3:
        size_multiplier = 0.8
    else:
        size_multiplier = 1.0
    
    return min(MAX_POSITION_SIZE, base_size * size_multiplier)

def calculate_base_position_size(prob):
    """基础仓位计算"""
    if prob > 0.8:
        return MAX_POSITION_SIZE
    elif prob > 0.7:
        return 1.5
    elif prob > 0.65:
        return 1.0
    elif prob > 0.6:
        return 0.8
    else:
        return MIN_POSITION_SIZE

def check_signal_quality(row, recent_signals):
    """检查信号质量，避免重复信号"""
    if len(recent_signals) == 0:
        return True
    
    # 检查是否与最近信号方向相同且时间间隔太短
    last_signal = recent_signals[-1]
    time_diff = (row['timestamp'] - last_signal['time']).total_seconds() / 3600  # 小时
    
    # 如果时间间隔小于2小时且方向相同，拒绝信号
    if time_diff < 2 and last_signal['direction'] == row.get('signal_direction'):
        return False
    
    return True

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
    print("=== RexKing Optimized 15m Strategy ===")
    
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
    equity_curve = [10000]
    recent_signals = []  # 记录最近信号
    
    print(f"🚀 开始优化版回测...")
    print(f"基础参数: 多空阈值={BASE_LONG_THRESHOLD}/{BASE_SHORT_THRESHOLD}, 趋势过滤={TREND_FILTER}")
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
                
                unrealized_pnl -= 0.002
                current_equity += unrealized_pnl * position['position_value']
        
        equity_curve.append(current_equity)
        max_equity = max(max_equity, current_equity)
        
        # 1. 检查现有持仓的退出条件
        for position in positions[:]:
            if position['status'] == 'open':
                # 更新追踪止盈
                if position['direction'] == 'long':
                    if current_price > position['entry_price'] * (1 + TRAILING_TP):
                        new_tp = current_price * (1 - TRAILING_TP * 0.5)
                        new_tp = max(new_tp, position['entry_price'])
                        if new_tp > position.get('trailing_tp', 0):
                            position['trailing_tp'] = new_tp
                else:  # short
                    if current_price < position['entry_price'] * (1 - TRAILING_TP):
                        new_tp = current_price * (1 + TRAILING_TP * 0.5)
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
                
                # 时间止损（3小时后）
                if not exit_reason and (current_time - position['entry_time']).total_seconds() > 3 * 3600:
                    exit_reason = 'time_stop'
                
                if exit_reason:
                    # 关闭持仓
                    if position['direction'] == 'long':
                        pnl = (current_price - position['entry_price']) / position['entry_price']
                    else:
                        pnl = (position['entry_price'] - current_price) / position['entry_price']
                    
                    pnl -= 0.002
                    actual_pnl = pnl * position['position_value']
                    
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
                    
                    equity += actual_pnl
                    
                    if current_date not in daily_pnl:
                        daily_pnl[current_date] = 0
                    daily_pnl[current_date] += actual_pnl
                    
                    positions.remove(position)
        
        # 2. 风险控制检查
        current_drawdown = (current_equity - max_equity) / max_equity
        if current_drawdown < MAX_DRAWDOWN:
            continue
        
        if current_date in daily_pnl and daily_pnl[current_date] < MAX_DAILY_LOSS * equity:
            continue
        
        active_positions = len([p for p in positions if p['status'] == 'open'])
        if active_positions >= MAX_CONCURRENT_POSITIONS:
            continue
        
        # 3. 生成信号
        features = row[feature_cols].values
        prob = model.predict_proba(features.reshape(1, -1))[0][1]
        
        # 计算动态阈值
        long_threshold, short_threshold = calculate_dynamic_thresholds(df, i)
        
        # 计算近期置信度
        if i >= CONFIDENCE_LOOKBACK:
            recent_probs = [model.predict_proba(df.iloc[j][feature_cols].values.reshape(1, -1))[0][1] 
                           for j in range(i-CONFIDENCE_LOOKBACK, i)]
            recent_confidence = np.mean([max(p, 1-p) for p in recent_probs])
        else:
            recent_confidence = 0.5
        
        # 4. 检查开仓条件
        signal_direction = None
        
        if prob > long_threshold and check_trend_filter(row):
            signal_direction = 'long'
        elif prob < short_threshold and check_trend_filter(row):
            signal_direction = 'short'
        
        # 检查信号质量
        if signal_direction and check_signal_quality(row, recent_signals):
            # 记录信号
            recent_signals.append({
                'time': current_time,
                'direction': signal_direction,
                'prob': prob
            })
            
            # 保持最近20个信号
            if len(recent_signals) > 20:
                recent_signals.pop(0)
            
            # 开仓
            if signal_direction == 'long':
                position_size = calculate_adaptive_position_size(prob, 'long', recent_confidence)
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
            
            elif signal_direction == 'short':
                position_size = calculate_adaptive_position_size(prob, 'short', recent_confidence)
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
    
    # 5. 强制平仓
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
        
        # 计算最大回撤
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
        
        print(f"\n📊 优化版回测结果:")
        print(f"总交易数: {total_trades}")
        print(f"胜率: {win_rate:.2%}")
        print(f"总收益: {total_return:.2%}")
        print(f"年化收益: {annual_return:.2f}%")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"平均单笔收益: {avg_trade_return:.2%}")
        print(f"最终权益: ${equity:,.2f}")
        
        # 保存结果
        trades_df.to_csv('optimized_trades_15m.csv', index=False)
        print(f"✅ 交易记录已保存到: optimized_trades_15m.csv")
        
        # 保存权益曲线
        equity_df = pd.DataFrame({'equity': equity_curve})
        equity_df.to_csv('optimized_equity_curve_15m.csv', index=False)
        print(f"✅ 权益曲线已保存到: optimized_equity_curve_15m.csv")
    else:
        print("❌ 没有产生任何交易信号")
    
    return trades

if __name__ == "__main__":
    main() 