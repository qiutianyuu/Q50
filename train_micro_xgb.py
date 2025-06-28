#!/usr/bin/env python3
"""
微观特征XGBoost训练脚本
使用最佳标签训练模型
"""

import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import json
import glob
import os
import argparse

def load_latest_features_with_labels():
    """加载带标签的特征文件"""
    files = glob.glob("data/realtime_features_with_labels_*.parquet")
    if not files:
        raise FileNotFoundError("No features with labels files found")
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading: {latest_file}")
    df = pd.read_parquet(latest_file)
    print(f"Loaded {len(df)} rows")
    return df, latest_file

def load_features_from_path(features_path):
    """从指定路径加载特征文件"""
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    print(f"Loading: {features_path}")
    df = pd.read_parquet(features_path)
    print(f"Loaded {len(df)} rows")
    return df

def prepare_features(df):
    """准备特征列"""
    # 排除非特征列
    exclude_cols = ['timestamp', 'label_h60_a0.3_maker', 'label_h120_a0.6_maker', 'label_h120_a0.3_maker']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"特征列数量: {len(feature_cols)}")
    return feature_cols

def train_model(X, y, label_name, test_size=0.2):
    """训练XGBoost模型"""
    print(f"\n=== 训练模型: {label_name} ===")
    
    # 移除中性标签
    mask = y != 0
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    if len(X_filtered) == 0:
        print("没有有效标签，跳过训练")
        return None, None
    
    # 转换为二分类 (1 vs -1)
    y_binary = (y_filtered == 1).astype(int)
    
    print(f"有效样本: {len(X_filtered)} (long: {y_binary.sum()}, short: {len(y_binary)-y_binary.sum()})")
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_binary, test_size=test_size, random_state=42, stratify=y_binary
    )
    
    # 训练模型
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='auc'
    )
    
    model.fit(X_train, y_train)
    
    # 评估
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"准确率: {accuracy:.3f}")
    print(f"AUC: {auc:.3f}")
    print(f"分类报告:\n{classification_report(y_test, y_pred)}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n前10重要特征:")
    print(feature_importance.head(10))
    
    return model, {
        'accuracy': accuracy,
        'auc': auc,
        'feature_importance': feature_importance.to_dict('records'),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

def main():
    parser = argparse.ArgumentParser(description='Train Micro XGBoost Model')
    parser.add_argument('--features', type=str, help='Path to features with labels parquet file')
    parser.add_argument('--model-out', type=str, default='xgb_micro_model.bin', help='Output model file path')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--label-col', type=str, default='label_h120_a0.3_maker', help='Label column to use')
    
    args = parser.parse_args()
    
    # 加载数据
    if args.features:
        df = load_features_from_path(args.features)
    else:
        df, _ = load_latest_features_with_labels()
    
    # 准备特征
    feature_cols = prepare_features(df)
    X = df[feature_cols]
    
    # 训练模型
    models = {}
    results = {}
    
    label_cols = [args.label_col] if args.label_col else ['label_h60_a0.3_maker', 'label_h120_a0.6_maker', 'label_h120_a0.3_maker']
    
    for label_col in label_cols:
        if label_col in df.columns:
            y = df[label_col]
            model, result = train_model(X, y, label_col)
            
            if model is not None:
                models[label_col] = model
                results[label_col] = result
    
    # 保存模型和结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型
    for label_name, model in models.items():
        if len(models) == 1:
            model_file = args.model_out
        else:
            model_file = f"xgb_micro_{label_name}_{timestamp}.bin"
        model.save_model(model_file)
        print(f"模型已保存: {model_file}")
    
    # 保存结果
    results_file = f"micro_model_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"结果已保存: {results_file}")
    
    return models, results

if __name__ == "__main__":
    main() 