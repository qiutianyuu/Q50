#!/usr/bin/env python3
"""
使用Walk-Forward优化的最佳参数重新训练模型
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 数据路径
FEATURES_FILE = "/Users/qiutianyu/features_offline_15m.parquet"
MODEL_FILE = "xgb_optimized_model.bin"
SIGNALS_FILE = "optimized_signals.csv"

# Walk-Forward优化的最佳参数
BEST_PARAMS = {
    'conservative': {
        'long_threshold': 0.7,    # 提高做多阈值
        'short_threshold': 0.7,   # 提高做空阈值
        'hold_period': 4,
        'name': 'Conservative'
    },
    'aggressive': {
        'long_threshold': 0.8,
        'short_threshold': 0.8,   # 提高做空阈值
        'hold_period': 5,
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

def train_optimized_model(df, feature_cols, test_size=0.2):
    """训练优化模型"""
    print("🔄 训练优化模型...")
    
    # 准备数据
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 训练模型（使用更保守的参数）
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2,  # 增加正则化
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    print("\n📊 分类报告:")
    print(classification_report(y_test, y_pred))
    
    print("\n📊 混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 特征重要性 (前15):")
    print(feature_importance.head(15))
    
    return model, X_test, y_test, y_pred, y_proba, feature_importance

def generate_optimized_signals(df, model, feature_cols, strategy_params):
    """生成优化信号"""
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
    
    # 生成信号 - 修复逻辑
    long_th = strategy_params['long_threshold']
    short_th = strategy_params['short_threshold']
    
    signals['signal'] = 0
    # 做多：上涨概率 > 阈值 且 上涨概率 > 下跌概率
    long_condition = (signals['prob_up'] > long_th) & (signals['prob_up'] > signals['prob_down'])
    signals.loc[long_condition, 'signal'] = 1
    
    # 做空：下跌概率 > 阈值 且 下跌概率 > 上涨概率  
    short_condition = (signals['prob_down'] > short_th) & (signals['prob_down'] > signals['prob_up'])
    signals.loc[short_condition, 'signal'] = -1
    
    # 计算信号统计
    signal_count = len(signals[signals['signal'] != 0])
    long_signals = len(signals[signals['signal'] == 1])
    short_signals = len(signals[signals['signal'] == -1])
    
    print(f"📊 信号统计:")
    print(f"总信号数: {signal_count}")
    print(f"做多信号: {long_signals}")
    print(f"做空信号: {short_signals}")
    
    return signals

def backtest_optimized_signals(signals, strategy_params, fee_rate=0.0004):
    """回测优化信号"""
    print(f"📈 回测{strategy_params['name']}策略...")
    
    hold_period = strategy_params['hold_period']
    
    # 只对有信号的样本进行回测
    signal_samples = signals[signals['signal'] != 0].copy()
    
    if len(signal_samples) == 0:
        print("⚠️ 没有生成任何信号")
        return signals, {
            'win_rate': 0,
            'total_ret': 0,
            'annual_ret': 0,
            'max_dd': 0,
            'sharpe': 0,
            'signal_count': 0
        }
    
    # 计算每笔信号的开平仓价格
    signal_samples['entry_price'] = signal_samples['close']
    signal_samples['exit_price'] = signal_samples['close'].shift(-hold_period)
    signal_samples['exit_time'] = signal_samples['timestamp'].shift(-hold_period)
    
    # 移除没有平仓价格的信号
    signal_samples = signal_samples.dropna(subset=['exit_price'])
    
    if len(signal_samples) == 0:
        print("⚠️ 所有信号都没有完整的平仓价格")
        return signals, {
            'win_rate': 0,
            'total_ret': 0,
            'annual_ret': 0,
            'max_dd': 0,
            'sharpe': 0,
            'signal_count': 0
        }
    
    # 计算收益
    signal_samples['ret'] = (signal_samples['exit_price'] - signal_samples['entry_price']) / signal_samples['entry_price'] * signal_samples['signal']
    signal_samples['ret_net'] = signal_samples['ret'] - fee_rate * 2  # 开平各收一次手续费
    
    # 统计
    win_rate = (signal_samples['ret_net'] > 0).mean()
    avg_ret = signal_samples['ret_net'].mean()
    
    # 计算累计收益（使用复利计算）
    signal_samples['cum_ret'] = (1 + signal_samples['ret_net']).cumprod() - 1
    total_ret = signal_samples['cum_ret'].iloc[-1]
    max_dd = (signal_samples['cum_ret'].cummax() - signal_samples['cum_ret']).max()
    
    # 计算年化收益和夏普比
    days = (signal_samples['timestamp'].max() - signal_samples['timestamp'].min()).days
    annual_ret = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0
    sharpe = avg_ret / (signal_samples['ret_net'].std() + 1e-8) * np.sqrt(252 * 96)  # 96个15分钟/天
    
    print(f"📊 {strategy_params['name']}策略回测结果:")
    print(f"有效信号数: {len(signal_samples)}")
    print(f"胜率: {win_rate:.2%}")
    print(f"平均单笔收益: {avg_ret:.4%}")
    print(f"累计收益: {total_ret:.2%}")
    print(f"年化收益: {annual_ret:.2%}")
    print(f"最大回撤: {max_dd:.2%}")
    print(f"夏普比: {sharpe:.2f}")
    print(f"信号区间: {signal_samples['timestamp'].min()} ~ {signal_samples['timestamp'].max()}")
    
    return signal_samples, {
        'win_rate': win_rate,
        'total_ret': total_ret,
        'annual_ret': annual_ret,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'signal_count': len(signal_samples)
    }

def main():
    """主函数"""
    print("🚀 开始优化模型训练...")
    
    # 加载数据
    df, feature_cols = load_and_prepare_data()
    
    # 训练模型
    model, X_test, y_test, y_pred, y_proba, feature_importance = train_optimized_model(df, feature_cols)
    
    # 测试两种策略
    results = {}
    
    for strategy_name, params in BEST_PARAMS.items():
        print(f"\n{'='*50}")
        print(f"测试 {params['name']} 策略")
        print(f"{'='*50}")
        
        # 生成信号
        signals = generate_optimized_signals(df, model, feature_cols, params)
        
        # 回测信号
        backtest_signals, stats = backtest_optimized_signals(signals, params)
        
        # 保存结果
        signals.to_csv(f"{strategy_name}_signals.csv", index=False)
        print(f"💾 {params['name']}信号已保存到: {strategy_name}_signals.csv")
        
        results[strategy_name] = stats
    
    # 保存模型和特征重要性
    model.save_model(MODEL_FILE)
    print(f"💾 优化模型已保存到: {MODEL_FILE}")
    
    feature_importance.to_csv("optimized_feature_importance.csv", index=False)
    print(f"💾 特征重要性已保存到: optimized_feature_importance.csv")
    
    # 策略对比
    print(f"\n{'='*60}")
    print("📊 策略对比")
    print(f"{'='*60}")
    
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.round(4))
    
    # 推荐策略
    best_strategy = max(results.keys(), key=lambda x: results[x]['sharpe'])
    print(f"\n🏆 推荐策略: {BEST_PARAMS[best_strategy]['name']}")
    print(f"夏普比: {results[best_strategy]['sharpe']:.2f}")
    print(f"年化收益: {results[best_strategy]['annual_ret']:.2%}")
    print(f"最大回撤: {results[best_strategy]['max_dd']:.2%}")
    
    # 绘制收益曲线对比
    plt.figure(figsize=(15, 8))
    
    for strategy_name, params in BEST_PARAMS.items():
        signals = pd.read_csv(f"{strategy_name}_signals.csv")
        signals['timestamp'] = pd.to_datetime(signals['timestamp'])
        
        # 计算累计收益
        hold_period = params['hold_period']
        signals['entry_price'] = signals['close']
        signals['exit_price'] = signals['close'].shift(-hold_period)
        signals = signals.dropna(subset=['exit_price'])
        
        signals['ret'] = (signals['exit_price'] - signals['entry_price']) / signals['entry_price'] * signals['signal']
        signals['ret_net'] = signals['ret'] - 0.0004 * 2
        signals['cum_ret'] = (1 + signals['ret_net']).cumprod() - 1
        
        plt.plot(signals['timestamp'], signals['cum_ret'], label=f"{params['name']} (Sharpe: {results[strategy_name]['sharpe']:.2f})")
    
    plt.title('Strategy Comparison - Cumulative Returns')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 策略对比图已保存为: strategy_comparison.png")

if __name__ == "__main__":
    main() 