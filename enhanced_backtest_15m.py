#!/usr/bin/env python3
"""
RexKing – Enhanced 15m Backtest with Adaptive Position Sizing & Trailing TP

基于现有回测脚本，添加增强功能：
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

def calculate_position_size(prob):
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
    
    print(f"🚀 开始增强回测...")
    print(f"参数: 多空阈值={LONG_THRESHOLD}/{SHORT_THRESHOLD}, 趋势过滤={TREND_FILTER}")
    print(f"止损={STOP_LOSS*100}%, 追踪止盈={TRAILING_TP*100}%")
    print(f"风险控制: 最大持仓={MAX_CONCURRENT_POSITIONS}, 日损失限制={MAX_DAILY_LOSS*100}%")
    
    for i, row in df.iterrows():
        current_time = row['timestamp']
        current_price = row['close']
        current_date = current_time.date()
        
        # 1. 检查现有持仓的退出条件
        for position in positions[:]:
            if position['status'] == 'open':
                # 更新追踪止盈
                if position['direction'] == 'long':
                    if current_price > position['entry_price'] * (1 + TRAILING_TP):
                        new_tp = current_price * (1 - TRAILING_TP * 0.5)
                        if new_tp > position.get('trailing_tp', 0):
                            position['trailing_tp'] = new_tp
                else:  # short
                    if current_price < position['entry_price'] * (1 - TRAILING_TP):
                        new_tp = current_price * (1 + TRAILING_TP * 0.5)
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
                    
                    # 计算实际收益
                    actual_pnl = pnl * position['size'] * position['entry_price']
                    
                    # 记录交易
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'size': position['size'],
                        'pnl': pnl,
                        'actual_pnl': actual_pnl,
                        'exit_reason': exit_reason
                    }
                    trades.append(trade)
                    
                    # 更新权益
                    equity += actual_pnl
                    max_equity = max(max_equity, equity)
                    
                    # 更新日收益
                    if current_date not in daily_pnl:
                        daily_pnl[current_date] = 0
                    daily_pnl[current_date] += actual_pnl
                    
                    position['status'] = 'closed'
        
        # 2. 风险控制检查
        # 检查最大回撤
        current_drawdown = (equity - max_equity) / max_equity
        if current_drawdown < MAX_DRAWDOWN:
            continue  # 跳过开仓
        
        # 检查日损失限制
        if current_date in daily_pnl and daily_pnl[current_date] < MAX_DAILY_LOSS * 10000:
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
            position_size = calculate_position_size(prob)
            position = {
                'entry_time': current_time,
                'entry_price': current_price,
                'direction': 'long',
                'size': position_size,
                'status': 'open'
            }
            positions.append(position)
        
        elif prob < SHORT_THRESHOLD and check_trend_filter(row):
            # 空头信号
            position_size = calculate_position_size(1 - prob)
            position = {
                'entry_time': current_time,
                'entry_price': current_price,
                'direction': 'short',
                'size': position_size,
                'status': 'open'
            }
            positions.append(position)
    
    # 4. 强制平仓所有剩余持仓
    for position in positions:
        if position['status'] == 'open':
            if position['direction'] == 'long':
                pnl = (df.iloc[-1]['close'] - position['entry_price']) / position['entry_price']
            else:
                pnl = (position['entry_price'] - df.iloc[-1]['close']) / position['entry_price']
            
            pnl -= 0.002
            actual_pnl = pnl * position['size'] * position['entry_price']
            
            trade = {
                'entry_time': position['entry_time'],
                'exit_time': df.iloc[-1]['timestamp'],
                'direction': position['direction'],
                'entry_price': position['entry_price'],
                'exit_price': df.iloc[-1]['close'],
                'size': position['size'],
                'pnl': pnl,
                'actual_pnl': actual_pnl,
                'exit_reason': 'force_close'
            }
            trades.append(trade)
            
            equity += actual_pnl
    
    # 分析结果
    if trades:
        trades_df = pd.DataFrame(trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades
        
        total_return = (equity - 10000) / 10000
        avg_trade_return = trades_df['pnl'].mean()
        
        # 计算最大回撤
        equity_curve = [10000]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade['actual_pnl'])
        
        max_drawdown = 0
        peak = equity_curve[0]
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            drawdown = (eq - peak) / peak
            max_drawdown = min(max_drawdown, drawdown)
        
        print(f"\n📊 回测结果:")
        print(f"总交易数: {total_trades}")
        print(f"胜率: {win_rate:.2%}")
        print(f"总收益: {total_return:.2%}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"平均单笔收益: {avg_trade_return:.2%}")
        print(f"最终权益: ${equity:,.2f}")
        
        # 保存交易记录
        trades_df.to_csv('enhanced_trades_15m.csv', index=False)
        print(f"✅ 交易记录已保存到: enhanced_trades_15m.csv")
    else:
        print("❌ 没有产生任何交易信号")
    
    return trades

if __name__ == "__main__":
    main() 