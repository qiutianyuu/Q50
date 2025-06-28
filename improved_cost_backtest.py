import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import xgboost as xgb
warnings.filterwarnings('ignore')

# 成本模型参数
MAKER_FEE = 0.0001  # 0.01% maker fee
TAKER_FEE = 0.0005  # 0.05% taker fee
SLIPPAGE_BASIS = 0.0002  # 2 bps base slippage
SLIPPAGE_VOLATILITY = 0.0001  # 1 bps per volatility unit
FUNDING_RATE = 0.0001  # 0.01% per 8h funding period (假设平均)

def load_data():
    """加载特征数据和模型"""
    try:
        # 加载特征数据
        features = pd.read_parquet('/Users/qiutianyu/data/processed/features_15m_2023_2025.parquet')
        print(f"Loaded features: {features.shape}")
        
        # 加载训练好的模型 (使用XGBoost .bin格式)
        model = xgb.Booster()
        model.load_model('xgb_15m_optimized.bin')  # 使用匹配特征的模型
        print("Loaded XGBoost model: xgb_15m_optimized.bin")
        
        return features, model
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def calculate_costs(entry_price, exit_price, position_size, volatility, is_long, is_maker=True):
    """计算交易成本 - 修复后的版本"""
    # 手续费 - 直接基于position_size计算，不需要乘以价格
    fee_rate = MAKER_FEE if is_maker else TAKER_FEE
    entry_fee = position_size * fee_rate
    exit_fee = position_size * fee_rate
    
    # 滑点 - 直接基于position_size计算
    slippage_rate = SLIPPAGE_BASIS + volatility * SLIPPAGE_VOLATILITY
    entry_slippage = position_size * slippage_rate
    exit_slippage = position_size * slippage_rate
    
    # 资金费率 (仅对多头)
    if is_long:
        # 假设平均持仓时间4个15分钟bar = 1小时
        funding_periods = 1/8  # 1小时 = 1/8个资金费率周期
        funding_cost = position_size * FUNDING_RATE * funding_periods
    else:
        funding_cost = 0
    
    total_cost = entry_fee + exit_fee + entry_slippage + exit_slippage + funding_cost
    return total_cost

def improved_backtest(features, model, long_threshold=0.7, short_threshold=0.3, 
                     holding_period=4, initial_capital=10000, position_size=1000):
    """改进的成本感知回测 - 使用更合理的阈值"""
    print(f"Running improved backtest with thresholds: {long_threshold}/{short_threshold}")
    print(f"Holding period: {holding_period} bars, Initial capital: ${initial_capital:,.0f}")
    print(f"Position size: ${position_size:,.0f}")
    
    # 准备特征数据 - 只使用模型训练时的特征
    model_features = model.feature_names
    X = features[model_features].copy()
    
    # 转换为DMatrix格式用于XGBoost预测
    dmatrix = xgb.DMatrix(X)
    probabilities = model.predict(dmatrix)
    
    # 生成信号
    signals = pd.DataFrame({
        'timestamp': features['timestamp'],
        'probability': probabilities,
        'volatility': features['volatility_10'] if 'volatility_10' in features.columns else 0.02,
        'signal': 0
    })
    
    # 应用更严格的阈值
    signals.loc[signals['probability'] > long_threshold, 'signal'] = 1
    signals.loc[signals['probability'] < short_threshold, 'signal'] = -1
    
    # 统计信号分布
    long_signals = (signals['signal'] == 1).sum()
    short_signals = (signals['signal'] == -1).sum()
    total_signals = long_signals + short_signals
    
    print(f"Signal distribution:")
    print(f"  Long signals: {long_signals} ({100*long_signals/len(signals):.1f}%)")
    print(f"  Short signals: {short_signals} ({100*short_signals/len(signals):.1f}%)")
    print(f"  Total signals: {total_signals} ({100*total_signals/len(signals):.1f}%)")
    
    # 回测逻辑
    trades = []
    position = 0
    entry_i = None
    entry_price = None
    entry_prob = None
    entry_vol = None
    capital = initial_capital
    
    for i in range(len(signals)):
        current_signal = signals.iloc[i]['signal']
        current_price = features.iloc[i]['close'] if 'close' in features.columns else 1000
        current_time = signals.iloc[i]['timestamp']
        current_vol = signals.iloc[i]['volatility']
        
        # 平仓逻辑
        if position != 0:
            bars_held = i - entry_i
            if bars_held >= holding_period:
                exit_price = current_price
                
                # 计算成本
                costs = calculate_costs(
                    entry_price, exit_price, position_size, entry_vol, 
                    position > 0, is_maker=True
                )
                
                # 计算收益
                if position > 0:  # 多头
                    gross_return = (exit_price - entry_price) / entry_price
                else:  # 空头
                    gross_return = (entry_price - exit_price) / entry_price
                
                net_return = gross_return - costs / position_size
                pnl = position_size * net_return
                capital += pnl
                
                # 确保capital不会变成负数
                if capital < 0:
                    capital = 0
                
                # 记录交易
                trades.append({
                    'entry_time': signals.iloc[entry_i]['timestamp'],
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': position,
                    'entry_prob': entry_prob,
                    'exit_prob': signals.iloc[i]['probability'],
                    'bars_held': bars_held,
                    'gross_return': gross_return,
                    'costs': costs,
                    'net_return': net_return,
                    'pnl': pnl,
                    'capital': capital
                })
                
                position = 0
                entry_i = None
                entry_price = None
                entry_prob = None
                entry_vol = None
        
        # 开仓逻辑
        if position == 0 and current_signal != 0:
            position = current_signal
            entry_i = i
            entry_price = current_price
            entry_prob = signals.iloc[i]['probability']
            entry_vol = current_vol
    
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        print("No trades generated!")
        return None, None
    
    # 计算指标
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    total_return = (capital - initial_capital) / initial_capital
    total_pnl = capital - initial_capital
    
    # 计算年化收益率
    if len(trades_df) > 0:
        first_trade = trades_df['entry_time'].min()
        last_trade = trades_df['exit_time'].max()
        days = (last_trade - first_trade).days
        annual_return = total_return * (365 / days) if days > 0 else 0
    else:
        annual_return = 0
    
    # 计算夏普比率
    if len(trades_df) > 0:
        returns = trades_df['net_return'].values
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 96) if np.std(returns) > 0 else 0
    else:
        sharpe = 0
    
    # 计算最大回撤
    if len(trades_df) > 0:
        cumulative = trades_df['capital'].values
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_drawdown = np.min(drawdown)
    else:
        max_drawdown = 0
    
    print(f"\n=== Improved Backtest Results ===")
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.1%}")
    print(f"Total return: {total_return:.1%}")
    print(f"Annual return: {annual_return:.1%}")
    print(f"Total PnL: ${total_pnl:,.2f}")
    print(f"Final capital: ${capital:,.2f}")
    print(f"Sharpe ratio: {sharpe:.2f}")
    print(f"Max drawdown: {max_drawdown:.1%}")
    
    # 成本分析
    if len(trades_df) > 0:
        avg_costs = trades_df['costs'].mean()
        avg_cost_pct = (trades_df['costs'] / trades_df['pnl'].abs()).mean()
        print(f"Average costs per trade: ${avg_costs:.2f}")
        print(f"Average costs as % of PnL: {avg_cost_pct:.1%}")
        
        # 理论成本验证
        theoretical_cost = position_size * (MAKER_FEE + MAKER_FEE + SLIPPAGE_BASIS + SLIPPAGE_BASIS + FUNDING_RATE/8)
        print(f"Theoretical cost per trade: ${theoretical_cost:.2f}")
        print(f"Cost ratio (actual/theoretical): {avg_costs/theoretical_cost:.1f}x")
    
    return trades_df, {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'annual_return': annual_return,
        'total_pnl': total_pnl,
        'final_capital': capital,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'long_signals': long_signals,
        'short_signals': short_signals
    }

def main():
    """主函数"""
    print("Starting improved cost-aware backtest...")
    
    # 加载数据
    features, model = load_data()
    if features is None or model is None:
        print("Failed to load data or model")
        return
    
    # 运行改进的回测 - 使用更严格的阈值
    trades, metrics = improved_backtest(
        features, model,
        long_threshold=0.7,    # 更严格的多头阈值
        short_threshold=0.3,   # 更严格的空头阈值
        holding_period=4,
        initial_capital=10000,
        position_size=1000
    )
    
    if trades is not None:
        # 保存结果
        trades.to_csv('improved_cost_trades.csv', index=False)
        print(f"\nTrades saved to improved_cost_trades.csv")
        
        # 保存指标
        pd.DataFrame([metrics]).to_csv('improved_cost_metrics.csv', index=False)
        print(f"Metrics saved to improved_cost_metrics.csv")

if __name__ == "__main__":
    main()
