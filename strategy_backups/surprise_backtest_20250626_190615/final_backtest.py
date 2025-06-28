#!/usr/bin/env python3
"""
最终的回测系统 - 使用固定资金分配和更严格的信号过滤
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 数据路径
FEATURES_FILE = "/Users/qiutianyu/features_offline_15m.parquet"
MODEL_FILE = "xgb_final_model.bin"

# 策略参数 - 更保守的设置
STRATEGY_PARAMS = {
    'conservative': {
        'long_threshold': 0.85,
        'short_threshold': 0.85,
        'hold_period': 12,  # 3小时
        'min_interval': 8,  # 2小时最小间隔
        'name': 'Conservative'
    },
    'aggressive': {
        'long_threshold': 0.9,
        'short_threshold': 0.9,
        'hold_period': 8,   # 2小时
        'min_interval': 6,  # 1.5小时最小间隔
        'name': 'Aggressive'
    }
}

def load_and_prepare_data():
    """加载和准备数据"""
    print("📥 加载特征数据...")
    df = pd.read_parquet(FEATURES_FILE)
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    
    # 排除不需要的列
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_1h', 'close_4h', 'label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"标签分布: {df['label'].value_counts().to_dict()}")
    
    return df, feature_cols

def train_final_model(df, feature_cols, test_size=0.2):
    """训练最终模型"""
    print("🔄 训练最终模型...")
    
    # 准备数据
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    # 计算类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    weight_dict = dict(zip(np.unique(y), class_weights))
    print(f"类别权重: {weight_dict}")
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 训练模型
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5,
        random_state=42,
        eval_metric='logloss'
    )
    
    # 设置类别权重
    sample_weights = np.array([weight_dict[label] for label in y_train])
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # 评估模型
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    print("\n📊 分类报告:")
    print(classification_report(y_test, y_pred))
    
    print("\n📊 混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
    # 检查概率分布
    print("\n📊 概率分布统计:")
    for i, label in enumerate(['下跌', '上涨', '横盘']):
        print(f"{label}: 平均概率={y_proba[:, i].mean():.3f}, 最大概率={y_proba[:, i].max():.3f}")
    
    return model, X_test, y_test, y_pred, y_proba

def generate_final_signals(df, model, feature_cols, strategy_params):
    """生成最终信号"""
    print(f"⚡ 生成{strategy_params['name']}策略信号...")
    print(f"阈值: ({strategy_params['long_threshold']}, {strategy_params['short_threshold']})")
    
    # 准备特征
    X = df[feature_cols].fillna(0)
    
    # 预测概率
    proba = model.predict_proba(X)
    
    # 创建信号DataFrame
    signals = pd.DataFrame({
        'timestamp': df['timestamp'],
        'close': df['close'],
        'prob_down': proba[:, 0],  # 下跌概率 (标签0)
        'prob_up': proba[:, 1],    # 上涨概率 (标签1) 
        'prob_flat': proba[:, 2],  # 横盘概率 (标签2)
        'prediction': model.predict(X)
    })
    
    # 生成原始信号 - 非常严格的逻辑
    long_th = strategy_params['long_threshold']
    short_th = strategy_params['short_threshold']
    
    signals['signal'] = 0
    
    # 做多：上涨概率 > 阈值 且 上涨概率 > 下跌概率 且 差值足够大
    long_condition = (
        (signals['prob_up'] > long_th) & 
        (signals['prob_up'] > signals['prob_down']) & 
        (signals['prob_up'] - signals['prob_down'] > 0.3)  # 更大的差值要求
    )
    signals.loc[long_condition, 'signal'] = 1
    
    # 做空：下跌概率 > 阈值 且 下跌概率 > 上涨概率 且 差值足够大
    short_condition = (
        (signals['prob_down'] > short_th) & 
        (signals['prob_down'] > signals['prob_up']) & 
        (signals['prob_down'] - signals['prob_up'] > 0.3)  # 更大的差值要求
    )
    signals.loc[short_condition, 'signal'] = -1
    
    # 信号去重和频率限制
    signals['signal_filtered'] = 0
    
    position = 0  # 0=空仓, 1=做多, -1=做空
    bars_in_trade = 0
    bars_since_last_signal = 0
    hold_period = strategy_params['hold_period']
    min_interval = strategy_params['min_interval']
    
    for i in range(len(signals)):
        current_signal = signals.iloc[i]['signal']
        
        # 如果当前有持仓
        if position != 0:
            bars_in_trade += 1
            bars_since_last_signal += 1
            
            # 持仓期满，平仓
            if bars_in_trade >= hold_period:
                position = 0
                bars_in_trade = 0
        
        # 如果空仓且有新信号，且满足最小间隔要求
        if position == 0 and current_signal != 0 and bars_since_last_signal >= min_interval:
            position = current_signal
            bars_in_trade = 0
            bars_since_last_signal = 0
            signals.iloc[i, signals.columns.get_loc('signal_filtered')] = current_signal
        elif position == 0:
            bars_since_last_signal += 1
    
    # 计算信号统计
    original_signals = len(signals[signals['signal'] != 0])
    filtered_signals = len(signals[signals['signal_filtered'] != 0])
    long_signals = len(signals[signals['signal_filtered'] == 1])
    short_signals = len(signals[signals['signal_filtered'] == -1])
    
    print(f"📊 信号统计:")
    print(f"原始信号数: {original_signals}")
    print(f"过滤后信号数: {filtered_signals}")
    print(f"做多信号: {long_signals}")
    print(f"做空信号: {short_signals}")
    print(f"信号减少比例: {(1 - filtered_signals/original_signals)*100:.1f}%")
    print(f"日均信号数: {filtered_signals / ((signals['timestamp'].max() - signals['timestamp'].min()).days):.1f}")
    
    return signals

def final_backtest(signals, strategy_params, fee_rate=0.0004, position_size=0.1):
    """最终回测 - 使用固定资金分配"""
    print(f"📈 回测{strategy_params['name']}策略...")
    print(f"每次交易使用资金比例: {position_size*100}%")
    
    # 只对有过滤信号的样本进行回测
    signal_samples = signals[signals['signal_filtered'] != 0].copy()
    
    if len(signal_samples) == 0:
        print("⚠️ 没有生成任何有效信号")
        return signals, {
            'win_rate': 0,
            'total_ret': 0,
            'annual_ret': 0,
            'max_dd': 0,
            'sharpe': 0,
            'signal_count': 0,
            'trades': []
        }
    
    # 初始化回测变量
    position = 0  # 0=空仓, 1=做多, -1=做空
    entry_price = 0
    entry_time = None
    bars_in_trade = 0
    hold_period = strategy_params['hold_period']
    
    total_capital = 10000  # 总资金
    available_capital = total_capital  # 可用资金
    equity_curve = []
    trades = []
    
    # 按时间顺序处理
    for i, row in signals.iterrows():
        current_time = row['timestamp']
        current_price = row['close']
        current_signal = row['signal_filtered']
        
        # 记录权益
        equity_curve.append({
            'timestamp': current_time,
            'total_capital': total_capital,
            'available_capital': available_capital,
            'position': position,
            'price': current_price
        })
        
        # 如果当前有持仓
        if position != 0:
            bars_in_trade += 1
            
            # 持仓期满，平仓
            if bars_in_trade >= hold_period:
                # 计算收益
                if position == 1:  # 做多
                    ret = (current_price - entry_price) / entry_price
                else:  # 做空
                    ret = (entry_price - current_price) / entry_price
                
                # 扣除手续费
                ret_net = ret - fee_rate * 2
                
                # 计算本次交易的资金变化
                trade_capital = total_capital * position_size
                capital_change = trade_capital * ret_net
                
                # 更新资金
                total_capital += capital_change
                available_capital += trade_capital  # 资金回到可用状态
                
                # 记录交易
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'return': ret,
                    'return_net': ret_net,
                    'trade_capital': trade_capital,
                    'capital_change': capital_change,
                    'total_capital': total_capital
                })
                
                # 重置持仓
                position = 0
                bars_in_trade = 0
        
        # 如果空仓且有新信号，开仓
        if position == 0 and current_signal != 0:
            position = current_signal
            entry_price = current_price
            entry_time = current_time
            bars_in_trade = 0
            available_capital -= total_capital * position_size  # 锁定资金
    
    # 处理最后可能未平仓的持仓
    if position != 0:
        last_price = signals.iloc[-1]['close']
        last_time = signals.iloc[-1]['timestamp']
        
        if position == 1:  # 做多
            ret = (last_price - entry_price) / entry_price
        else:  # 做空
            ret = (entry_price - last_price) / entry_price
        
        ret_net = ret - fee_rate * 2
        trade_capital = total_capital * position_size
        capital_change = trade_capital * ret_net
        total_capital += capital_change
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': last_time,
            'position': position,
            'entry_price': entry_price,
            'exit_price': last_price,
            'return': ret,
            'return_net': ret_net,
            'trade_capital': trade_capital,
            'capital_change': capital_change,
            'total_capital': total_capital
        })
    
    # 计算统计指标
    if len(trades) == 0:
        print("⚠️ 没有完成任何交易")
        return signals, {
            'win_rate': 0,
            'total_ret': 0,
            'annual_ret': 0,
            'max_dd': 0,
            'sharpe': 0,
            'signal_count': 0,
            'trades': []
        }
    
    trades_df = pd.DataFrame(trades)
    win_rate = (trades_df['return_net'] > 0).mean()
    avg_ret = trades_df['return_net'].mean()
    total_ret = (total_capital - 10000) / 10000
    
    # 计算最大回撤
    equity_df = pd.DataFrame(equity_curve)
    equity_df['cummax'] = equity_df['total_capital'].cummax()
    equity_df['drawdown'] = (equity_df['cummax'] - equity_df['total_capital']) / equity_df['cummax']
    max_dd = equity_df['drawdown'].max()
    
    # 计算年化收益和夏普比
    days = (signals['timestamp'].max() - signals['timestamp'].min()).days
    annual_ret = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0
    sharpe = avg_ret / (trades_df['return_net'].std() + 1e-8) * np.sqrt(252 * 96)  # 96个15分钟/天
    
    print(f"📊 {strategy_params['name']}策略回测结果:")
    print(f"总交易数: {len(trades)}")
    print(f"胜率: {win_rate:.2%}")
    print(f"平均单笔收益: {avg_ret:.4%}")
    print(f"累计收益: {total_ret:.2%}")
    print(f"年化收益: {annual_ret:.2%}")
    print(f"最大回撤: {max_dd:.2%}")
    print(f"夏普比: {sharpe:.2f}")
    print(f"最终资金: ${total_capital:,.2f}")
    print(f"信号区间: {signals['timestamp'].min()} ~ {signals['timestamp'].max()}")
    
    return signals, {
        'win_rate': win_rate,
        'total_ret': total_ret,
        'annual_ret': annual_ret,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'signal_count': len(signal_samples),
        'trades': trades,
        'equity_curve': equity_curve
    }

def main():
    """主函数"""
    print("🚀 开始最终回测...")
    
    # 加载数据
    df, feature_cols = load_and_prepare_data()
    
    # 训练最终模型
    model, X_test, y_test, y_pred, y_proba = train_final_model(df, feature_cols)
    
    # 测试两种策略
    results = {}
    
    for strategy_name, params in STRATEGY_PARAMS.items():
        print(f"\n{'='*50}")
        print(f"测试 {params['name']} 策略")
        print(f"{'='*50}")
        
        # 生成最终信号
        signals = generate_final_signals(df, model, feature_cols, params)
        
        # 回测信号
        backtest_signals, stats = final_backtest(signals, params)
        
        # 保存结果
        signals.to_csv(f"{strategy_name}_final_signals.csv", index=False)
        print(f"💾 {params['name']}信号已保存到: {strategy_name}_final_signals.csv")
        
        # 保存交易记录
        if stats['trades']:
            trades_df = pd.DataFrame(stats['trades'])
            trades_df.to_csv(f"{strategy_name}_final_trades.csv", index=False)
            print(f"💾 交易记录已保存到: {strategy_name}_final_trades.csv")
        
        results[strategy_name] = stats
    
    # 保存模型
    model.save_model(MODEL_FILE)
    print(f"💾 最终模型已保存到: {MODEL_FILE}")
    
    # 策略对比
    print(f"\n{'='*60}")
    print("📊 策略对比")
    print(f"{'='*60}")
    
    comparison_data = {}
    for name, stats in results.items():
        comparison_data[name] = {
            'win_rate': stats['win_rate'],
            'total_ret': stats['total_ret'],
            'annual_ret': stats['annual_ret'],
            'max_dd': stats['max_dd'],
            'sharpe': stats['sharpe'],
            'signal_count': stats['signal_count']
        }
    
    comparison_df = pd.DataFrame(comparison_data).T
    print(comparison_df.round(4))
    
    # 推荐策略
    if results:
        best_strategy = max(results.keys(), key=lambda x: results[x]['sharpe'])
        print(f"\n🏆 推荐策略: {STRATEGY_PARAMS[best_strategy]['name']}")
        print(f"夏普比: {results[best_strategy]['sharpe']:.2f}")
        print(f"年化收益: {results[best_strategy]['annual_ret']:.2%}")
        print(f"最大回撤: {results[best_strategy]['max_dd']:.2%}")
    
    # 绘制权益曲线对比
    plt.figure(figsize=(15, 8))
    
    for strategy_name, params in STRATEGY_PARAMS.items():
        if results[strategy_name]['equity_curve']:
            equity_df = pd.DataFrame(results[strategy_name]['equity_curve'])
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            
            # 计算相对收益
            initial_capital = equity_df['total_capital'].iloc[0]
            equity_df['relative_return'] = (equity_df['total_capital'] - initial_capital) / initial_capital
            
            plt.plot(equity_df['timestamp'], equity_df['relative_return'], 
                    label=f"{params['name']} (Sharpe: {results[strategy_name]['sharpe']:.2f})")
    
    plt.title('Final Strategy Comparison - Relative Returns')
    plt.xlabel('Time')
    plt.ylabel('Relative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('final_strategy_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 策略对比图已保存为: final_strategy_comparison.png")

if __name__ == "__main__":
    main() 