#!/usr/bin/env python3
"""
快速训练优化后的模型 - 使用经验性参数
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def train_optimized_15m():
    """训练优化后的15m模型"""
    print("🚀 训练优化后的15m模型...")
    
    # 读取数据
    df = pd.read_parquet('/Users/qiutianyu/data/processed/features_15m_2023_2025.parquet')
    
    # 准备特征
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'trend_1h', 'trend_4h']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"样本数量: {len(X)}")
    print(f"标签分布: {y.value_counts().to_dict()}")
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 优化参数（经验性）
    params = {
        'max_depth': 6,
        'n_estimators': 300,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'random_state': 42
    }
    
    # 训练模型
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=0)
    
    # 评估
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"测试集AUC: {auc:.4f}")
    
    # 高置信度预测
    high_conf_mask = (y_pred_proba > 0.8) | (y_pred_proba < 0.2)
    if high_conf_mask.sum() > 0:
        high_conf_auc = roc_auc_score(y_test[high_conf_mask], y_pred_proba[high_conf_mask])
        high_conf_acc = (y_pred[high_conf_mask] == y_test[high_conf_mask]).mean()
        print(f"高置信度样本AUC: {high_conf_auc:.4f}")
        print(f"高置信度样本准确率: {high_conf_acc:.4f}")
        print(f"高置信度样本数量: {high_conf_mask.sum()}")
    
    # 保存模型
    model.save_model('xgb_15m_optimized.bin')
    print("✅ 优化后的15m模型已保存: xgb_15m_optimized.bin")
    
    return model, auc

def train_optimized_5m():
    """训练优化后的5m模型"""
    print("\n🚀 训练优化后的5m模型...")
    
    # 读取数据
    df = pd.read_parquet('/Users/qiutianyu/data/processed/features_5m_2023_2025.parquet')
    
    # 准备特征
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'trend_1h', 'trend_4h']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"样本数量: {len(X)}")
    print(f"标签分布: {y.value_counts().to_dict()}")
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 优化参数（经验性）
    params = {
        'max_depth': 5,
        'n_estimators': 250,
        'learning_rate': 0.12,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'min_child_weight': 2,
        'reg_alpha': 0.05,
        'reg_lambda': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'random_state': 42
    }
    
    # 训练模型
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=0)
    
    # 评估
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"测试集AUC: {auc:.4f}")
    
    # 高置信度预测
    high_conf_mask = (y_pred_proba > 0.8) | (y_pred_proba < 0.2)
    if high_conf_mask.sum() > 0:
        high_conf_auc = roc_auc_score(y_test[high_conf_mask], y_pred_proba[high_conf_mask])
        high_conf_acc = (y_pred[high_conf_mask] == y_test[high_conf_mask]).mean()
        print(f"高置信度样本AUC: {high_conf_auc:.4f}")
        print(f"高置信度样本准确率: {high_conf_acc:.4f}")
        print(f"高置信度样本数量: {high_conf_mask.sum()}")
    
    # 保存模型
    model.save_model('xgb_5m_optimized.bin')
    print("✅ 优化后的5m模型已保存: xgb_5m_optimized.bin")
    
    return model, auc

def main():
    print("🎯 快速训练优化后的模型...")
    
    # 训练15m模型
    model_15m, auc_15m = train_optimized_15m()
    
    # 训练5m模型
    model_5m, auc_5m = train_optimized_5m()
    
    # 结果总结
    print("\n📊 训练结果总结:")
    print(f"15m模型AUC: {auc_15m:.4f}")
    print(f"5m模型AUC: {auc_5m:.4f}")
    
    print("\n🎉 模型训练完成!")

if __name__ == "__main__":
    main() 