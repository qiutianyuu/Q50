#!/usr/bin/env python3
"""
特征分析和筛选 - SHAP重要性分析
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import shap
import warnings
warnings.filterwarnings('ignore')

def analyze_features_15m():
    """分析15m特征"""
    print("🔍 分析15m特征...")
    
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
    
    # 训练简单模型
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = xgb.XGBClassifier(
        max_depth=4, n_estimators=100, learning_rate=0.1,
        random_state=42, eval_metric='auc'
    )
    model.fit(X_train, y_train, verbose=0)
    
    # 预测和评估
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"测试集AUC: {auc:.4f}")
    
    # SHAP分析
    print("📊 计算SHAP重要性...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.sample(min(1000, len(X_test)), random_state=42))
    
    # 计算特征重要性
    feature_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\n📈 Top-20 特征重要性:")
    print(importance_df.head(20).to_string(index=False))
    
    # 保存结果
    importance_df.to_csv('/Users/qiutianyu/data/processed/feature_importance_15m.csv', index=False)
    print(f"\n特征重要性已保存: /Users/qiutianyu/data/processed/feature_importance_15m.csv")
    
    # 筛选重要特征（重要性 > 1%）
    threshold = importance_df['importance'].max() * 0.01
    important_features = importance_df[importance_df['importance'] > threshold]['feature'].tolist()
    
    print(f"\n🎯 重要特征筛选 (阈值: {threshold:.6f}):")
    print(f"保留特征数: {len(important_features)} / {len(feature_cols)}")
    print(f"保留特征: {important_features}")
    
    # 保存重要特征列表
    with open('/Users/qiutianyu/data/processed/important_features_15m.txt', 'w') as f:
        f.write('\n'.join(important_features))
    
    return important_features, importance_df

def analyze_features_5m():
    """分析5m特征"""
    print("\n🔍 分析5m特征...")
    
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
    
    # 训练简单模型
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = xgb.XGBClassifier(
        max_depth=4, n_estimators=100, learning_rate=0.1,
        random_state=42, eval_metric='auc'
    )
    model.fit(X_train, y_train, verbose=0)
    
    # 预测和评估
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"测试集AUC: {auc:.4f}")
    
    # SHAP分析
    print("📊 计算SHAP重要性...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.sample(min(1000, len(X_test)), random_state=42))
    
    # 计算特征重要性
    feature_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\n📈 Top-20 特征重要性:")
    print(importance_df.head(20).to_string(index=False))
    
    # 保存结果
    importance_df.to_csv('/Users/qiutianyu/data/processed/feature_importance_5m.csv', index=False)
    print(f"\n特征重要性已保存: /Users/qiutianyu/data/processed/feature_importance_5m.csv")
    
    # 筛选重要特征（重要性 > 1%）
    threshold = importance_df['importance'].max() * 0.01
    important_features = importance_df[importance_df['importance'] > threshold]['feature'].tolist()
    
    print(f"\n🎯 重要特征筛选 (阈值: {threshold:.6f}):")
    print(f"保留特征数: {len(important_features)} / {len(feature_cols)}")
    print(f"保留特征: {important_features}")
    
    # 保存重要特征列表
    with open('/Users/qiutianyu/data/processed/important_features_5m.txt', 'w') as f:
        f.write('\n'.join(important_features))
    
    return important_features, importance_df

def main():
    print("🚀 开始特征分析...")
    
    # 分析15m特征
    important_15m, importance_15m = analyze_features_15m()
    
    # 分析5m特征
    important_5m, importance_5m = analyze_features_5m()
    
    # 对比分析
    print("\n📊 特征重要性对比:")
    print("15m Top-10特征:")
    print(importance_15m.head(10)[['feature', 'importance']].to_string(index=False))
    
    print("\n5m Top-10特征:")
    print(importance_5m.head(10)[['feature', 'importance']].to_string(index=False))
    
    print("\n🎉 特征分析完成!")

if __name__ == "__main__":
    main() 