#!/usr/bin/env python3
"""
XGBoost二分类训练 - 使用平衡标签
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import IsotonicRegression
import warnings
from pathlib import Path
import argparse
warnings.filterwarnings('ignore')

def load_data(features_path, labels_path):
    """加载特征和标签数据"""
    print(f"📁 加载特征数据: {features_path}")
    features_df = pd.read_parquet(features_path)
    
    print(f"📁 加载标签数据: {labels_path}")
    labels_df = pd.read_csv(labels_path)
    
    # 确保时间戳格式一致
    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
    labels_df['timestamp'] = pd.to_datetime(labels_df['timestamp'])
    
    # 将标签时间戳转换为UTC时区
    if labels_df['timestamp'].dt.tz is None:
        labels_df['timestamp'] = labels_df['timestamp'].dt.tz_localize('UTC')
    
    print(f"特征时间戳格式: {features_df['timestamp'].dtype}")
    print(f"标签时间戳格式: {labels_df['timestamp'].dtype}")
    
    # 合并特征和标签
    merged_df = features_df.merge(labels_df[['timestamp', 'label']], on='timestamp', how='inner')
    
    print(f"📊 合并后数据形状: {merged_df.shape}")
    print(f"📅 时间范围: {merged_df['timestamp'].min()} 到 {merged_df['timestamp'].max()}")
    
    # 显示标签分布
    print(f"📊 标签分布:")
    print(merged_df['label'].value_counts())
    print(merged_df['label'].value_counts(normalize=True) * 100)
    
    return merged_df

def prepare_features(df, exclude_cols=None):
    """准备特征矩阵"""
    if exclude_cols is None:
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'date', 'label']
    
    # 排除不需要的列
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 只保留数值型特征
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    print(f"📈 使用特征数量: {len(numeric_cols)}")
    
    X = df[numeric_cols].fillna(0)
    y = df['label']
    
    return X, y, numeric_cols

def train_model(X_train, y_train, X_test, y_test, feature_cols):
    """训练模型"""
    print(f"🎯 训练XGBoost模型")
    print(f"训练样本: {len(X_train)}")
    print(f"测试样本: {len(X_test)}")
    
    # 只保留交易信号（多头和空头）
    train_trade_mask = y_train != 0
    test_trade_mask = y_test != 0
    
    X_train_trade = X_train[train_trade_mask]
    y_train_trade = y_train[train_trade_mask]
    X_test_trade = X_test[test_trade_mask]
    y_test_trade = y_test[test_trade_mask]
    
    # 将标签转换为二分类（1=多头，0=空头）
    y_train_binary = (y_train_trade == 1).astype(int)
    y_test_binary = (y_test_trade == 1).astype(int)
    
    print(f"交易信号训练样本: {len(X_train_trade)}")
    print(f"交易信号测试样本: {len(X_test_trade)}")
    print(f"多头/空头比例: {y_train_binary.mean():.2%}")
    
    # 训练模型
    model = xgb.XGBClassifier(
        max_depth=4,
        n_estimators=200,
        learning_rate=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        eval_metric='auc',
        early_stopping_rounds=50
    )
    
    # 训练
    model.fit(
        X_train_trade, y_train_binary,
        eval_set=[(X_test_trade, y_test_binary)],
        verbose=0
    )
    
    # 预测
    train_proba = model.predict_proba(X_train_trade)[:, 1]
    test_proba = model.predict_proba(X_test_trade)[:, 1]
    
    # 评估
    train_auc = roc_auc_score(y_train_binary, train_proba)
    test_auc = roc_auc_score(y_test_binary, test_proba)
    
    # 概率校准
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(train_proba, y_train_binary)
    test_proba_calibrated = calibrator.predict(test_proba)
    test_auc_calibrated = roc_auc_score(y_test_binary, test_proba_calibrated)
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"训练AUC: {train_auc:.4f}")
    print(f"测试AUC: {test_auc:.4f}")
    print(f"校准后AUC: {test_auc_calibrated:.4f}")
    print(f"过拟合程度: {train_auc - test_auc:.4f}")
    print(f"Top特征: {', '.join(feature_importance.head(3)['feature'].tolist())}")
    
    return model, calibrator, feature_importance, {
        'train_auc': train_auc,
        'test_auc': test_auc,
        'test_auc_calibrated': test_auc_calibrated,
        'overfitting': train_auc - test_auc
    }

def save_model(model, calibrator, feature_importance, output_path):
    """保存模型"""
    import joblib
    
    # 保存模型和校准器
    model_data = {
        'model': model,
        'calibrator': calibrator,
        'feature_importance': feature_importance
    }
    
    joblib.dump(model_data, output_path)
    print(f"✅ 模型已保存: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='XGBoost二分类训练')
    parser.add_argument('--features', type=str, required=True, help='特征文件路径')
    parser.add_argument('--labels', type=str, required=True, help='标签文件路径')
    parser.add_argument('--output', type=str, required=True, help='模型输出路径')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    
    args = parser.parse_args()
    
    print("🚀 XGBoost二分类训练")
    print(f"📁 特征文件: {args.features}")
    print(f"📁 标签文件: {args.labels}")
    print(f"📁 模型输出: {args.output}")
    print(f"📊 测试集比例: {args.test_size}")
    
    # 加载数据
    df = load_data(args.features, args.labels)
    
    # 准备特征
    X, y, feature_cols = prepare_features(df)
    
    # 时间序列分割
    split_idx = int(len(X) * (1 - args.test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"时间分割点: {df['timestamp'].iloc[split_idx]}")
    
    # 训练模型
    model, calibrator, feature_importance, metrics = train_model(
        X_train, y_train, X_test, y_test, feature_cols
    )
    
    # 保存模型
    save_model(model, calibrator, feature_importance, args.output)
    
    print("✅ 训练完成！")

if __name__ == "__main__":
    main() 