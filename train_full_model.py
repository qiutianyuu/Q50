#!/usr/bin/env python3
"""
基于离线特征进行全量训练，生成信号并回测
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
MODEL_FILE = "xgb_full_model.bin"
SIGNALS_FILE = "full_signals.csv"

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

def train_model(df, feature_cols, test_size=0.2):
    """训练模型"""
    print("🔄 训练模型...")
    
    # 准备数据
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 训练模型
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1,
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

def generate_signals(df, model, feature_cols, thresholds=(0.6, 0.4)):
    """生成交易信号"""
    print(f"⚡ 生成交易信号，阈值: {thresholds}")
    
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
        'prob_flat': proba[:, 2] if proba.shape[1] == 3 else 0,  # 横盘概率 (标签2)
        'prediction': model.predict(X)
    })
    
    # 生成信号 (0=下跌, 1=上涨, 2=横盘)
    long_th, short_th = thresholds
    signals['signal'] = 0
    signals.loc[signals['prob_up'] > long_th, 'signal'] = 1  # 做多信号
    signals.loc[signals['prob_down'] > short_th, 'signal'] = -1  # 做空信号
    
    # 计算信号统计
    signal_count = len(signals[signals['signal'] != 0])
    long_signals = len(signals[signals['signal'] == 1])
    short_signals = len(signals[signals['signal'] == -1])
    
    print(f"📊 信号统计:")
    print(f"总信号数: {signal_count}")
    print(f"做多信号: {long_signals}")
    print(f"做空信号: {short_signals}")
    
    return signals

def backtest_signals(signals, hold_period=3, fee_rate=0.0004):
    """回测信号"""
    print(f"📈 回测信号，持仓期: {hold_period}根K线")
    
    # 计算每笔信号的开平仓价格
    signals['entry_price'] = signals['close']
    signals['exit_price'] = signals['close'].shift(-hold_period)
    signals['exit_time'] = signals['timestamp'].shift(-hold_period)
    
    # 移除没有平仓价格的信号
    signals = signals.dropna(subset=['exit_price'])
    
    # 计算收益
    signals['ret'] = (signals['exit_price'] - signals['entry_price']) / signals['entry_price'] * signals['signal']
    signals['ret_net'] = signals['ret'] - fee_rate * 2  # 开平各收一次手续费
    signals['cum_ret'] = (1 + signals['ret_net']).cumprod() - 1
    
    # 统计
    win_rate = (signals['ret_net'] > 0).mean()
    avg_ret = signals['ret_net'].mean()
    total_ret = signals['cum_ret'].iloc[-1]
    max_dd = (signals['cum_ret'].cummax() - signals['cum_ret']).max()
    
    print(f"📊 回测结果:")
    print(f"胜率: {win_rate:.2%}")
    print(f"平均单笔收益: {avg_ret:.4%}")
    print(f"累计收益: {total_ret:.2%}")
    print(f"最大回撤: {max_dd:.2%}")
    print(f"信号区间: {signals['timestamp'].min()} ~ {signals['timestamp'].max()}")
    
    return signals

def main():
    """主函数"""
    print("🚀 开始全量训练...")
    
    # 加载数据
    df, feature_cols = load_and_prepare_data()
    
    # 训练模型
    model, X_test, y_test, y_pred, y_proba, feature_importance = train_model(df, feature_cols)
    
    # 生成信号
    signals = generate_signals(df, model, feature_cols, thresholds=(0.6, 0.4))
    
    # 回测信号
    backtest_results = backtest_signals(signals, hold_period=3, fee_rate=0.0004)
    
    # 保存结果
    model.save_model(MODEL_FILE)
    print(f"💾 模型已保存到: {MODEL_FILE}")
    
    signals.to_csv(SIGNALS_FILE, index=False)
    print(f"💾 信号已保存到: {SIGNALS_FILE}")
    
    feature_importance.to_csv("full_feature_importance.csv", index=False)
    print(f"💾 特征重要性已保存到: full_feature_importance.csv")
    
    # 显示信号样本
    print("\n📊 信号样本:")
    print(signals[signals['signal'] != 0].head(10))
    
    # 绘制收益曲线
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_results['timestamp'], backtest_results['cum_ret'], label='Cumulative Return')
    plt.title('Full Model Backtest Results')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig('full_backtest_curve.png')
    print("📊 收益曲线已保存为: full_backtest_curve.png")

if __name__ == "__main__":
    main() 