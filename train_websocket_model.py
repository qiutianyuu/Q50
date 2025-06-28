#!/usr/bin/env python3
"""
基于websocket特征训练短期预测模型
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def create_labels(df, horizon=4, threshold=0.001):
    """
    创建短期价格变动标签
    horizon: 预测未来几个15分钟窗口
    threshold: 价格变动阈值
    """
    print(f"创建标签: horizon={horizon}, threshold={threshold}")
    
    # 计算未来价格变动
    df['future_price'] = df['ws_mid_price'].shift(-horizon)
    df['price_change_future'] = (df['future_price'] - df['ws_mid_price']) / df['ws_mid_price']
    
    # 创建标签
    df['label'] = 0
    df.loc[df['price_change_future'] > threshold, 'label'] = 1  # 上涨
    df.loc[df['price_change_future'] < -threshold, 'label'] = 2  # 下跌 (改为2而不是-1)
    
    # 移除最后几行（没有未来数据）
    df = df.dropna(subset=['price_change_future'])
    
    print(f"标签分布: {df['label'].value_counts().to_dict()}")
    return df

def prepare_features(df):
    """准备特征"""
    print("准备特征...")
    
    # 选择websocket特征
    feature_columns = [col for col in df.columns if col.startswith('ws_') and col != 'ws_mid_price']
    
    # 添加技术指标
    df['ws_price_momentum'] = df['ws_mid_price'].pct_change(2)
    df['ws_volume_momentum'] = df['ws_total_volume'].pct_change(2)
    df['ws_liquidity_momentum'] = df['ws_total_liquidity'].pct_change(2)
    
    feature_columns.extend(['ws_price_momentum', 'ws_volume_momentum', 'ws_liquidity_momentum'])
    
    # 添加交互特征
    df['ws_volume_price_ratio'] = df['ws_total_volume'] / df['ws_mid_price']
    df['ws_liquidity_price_ratio'] = df['ws_total_liquidity'] / df['ws_mid_price']
    df['ws_spread_ratio'] = df['ws_spread'] / df['ws_mid_price']
    
    feature_columns.extend(['ws_volume_price_ratio', 'ws_liquidity_price_ratio', 'ws_spread_ratio'])
    
    # 移除包含NaN的特征并确保唯一性
    feature_columns = [col for col in feature_columns if col in df.columns]
    feature_columns = list(dict.fromkeys(feature_columns))  # 保持顺序的去重
    
    print(f"使用 {len(feature_columns)} 个特征")
    print(f"特征列表: {feature_columns}")
    
    return df, feature_columns

def train_model(df, feature_columns, test_size=0.3):
    """训练模型"""
    print("训练模型...")
    
    # 准备数据
    X = df[feature_columns].fillna(0)
    
    # 清理无穷大值和异常值
    X = X.replace([np.inf, -np.inf], 0)
    
    # 移除异常值 (超过3个标准差的值)
    for col in X.columns:
        mean_val = X[col].mean()
        std_val = X[col].std()
        if std_val > 0:
            X[col] = X[col].clip(mean_val - 3*std_val, mean_val + 3*std_val)
    
    y = df['label']
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 训练模型
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特征重要性 (前10):")
    print(feature_importance.head(10))
    
    return model, X_test, y_test, y_pred, y_proba

def generate_signals(df, model, feature_columns, threshold=0.6):
    """生成交易信号"""
    print(f"生成交易信号，概率阈值: {threshold}")
    
    # 准备特征
    X = df[feature_columns].fillna(0)
    
    # 预测概率
    proba = model.predict_proba(X)
    
    # 创建信号DataFrame
    signals = pd.DataFrame({
        'timestamp': df['timestamp'],
        'mid_price': df['ws_mid_price'],
        'prob_down': proba[:, 0],  # 下跌概率 (标签0)
        'prob_up': proba[:, 1],    # 上涨概率 (标签1)
        'prob_flat': proba[:, 2] if proba.shape[1] == 3 else 0,  # 横盘概率 (标签2)
        'prediction': model.predict(X)
    })
    
    # 生成信号 (0=下跌, 1=上涨, 2=横盘)
    signals['signal'] = 0
    signals.loc[signals['prob_up'] > threshold, 'signal'] = 1  # 做多信号
    signals.loc[signals['prob_down'] > threshold, 'signal'] = -1  # 做空信号
    
    # 计算信号统计
    signal_count = len(signals[signals['signal'] != 0])
    long_signals = len(signals[signals['signal'] == 1])
    short_signals = len(signals[signals['signal'] == -1])
    
    print(f"信号统计:")
    print(f"总信号数: {signal_count}")
    print(f"做多信号: {long_signals}")
    print(f"做空信号: {short_signals}")
    
    return signals

def main():
    """主函数"""
    print("开始训练websocket模型...")
    
    # 加载特征
    try:
        df = pd.read_parquet('data/features_15m_websocket_only.parquet')
        print(f"加载特征: {df.shape}")
    except FileNotFoundError:
        print("未找到特征文件")
        return
    
    # 创建标签
    df = create_labels(df, horizon=2, threshold=0.002)  # 预测30分钟，0.2%阈值
    
    # 准备特征
    df, feature_columns = prepare_features(df)
    
    # 训练模型
    model, X_test, y_test, y_pred, y_proba = train_model(df, feature_columns)
    
    # 生成信号
    signals = generate_signals(df, model, feature_columns, threshold=0.6)
    
    # 保存模型和信号
    model_file = 'xgb_websocket_model.bin'
    model.save_model(model_file)
    print(f"模型已保存到: {model_file}")
    
    signals_file = 'websocket_signals.csv'
    signals.to_csv(signals_file, index=False)
    print(f"信号已保存到: {signals_file}")
    
    # 显示信号样本
    print("\n信号样本:")
    print(signals[signals['signal'] != 0].head(10))
    
    # 保存特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_file = 'websocket_feature_importance.csv'
    feature_importance.to_csv(importance_file, index=False)
    print(f"特征重要性已保存到: {importance_file}")

if __name__ == "__main__":
    main() 