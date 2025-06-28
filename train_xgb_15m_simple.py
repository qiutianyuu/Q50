#!/usr/bin/env python3
"""
简化版15m XGBoost训练 - 过拟合检测
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🔧 简化版15m XGBoost训练 - 过拟合检测")
    
    # 读取修正版特征数据
    df = pd.read_parquet('/Users/qiutianyu/data/processed/features_15m_fixed.parquet')
    
    # 准备特征
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'trend_1h', 'trend_4h']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"样本数量: {len(X)}")
    print(f"标签分布: {y.value_counts().to_dict()}")
    
    # 时间排序
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    X_sorted = X.loc[df_sorted.index]
    y_sorted = y.loc[df_sorted.index]
    
    # 简单的时序分割：前80%训练，后20%测试
    split_idx = int(len(df_sorted) * 0.8)
    
    X_train = X_sorted[:split_idx]
    y_train = y_sorted[:split_idx]
    X_test = X_sorted[split_idx:]
    y_test = y_sorted[split_idx:]
    
    print(f"训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    
    # 正则化参数（避免过拟合）
    params = {
        'max_depth': 4,           # 降低深度
        'n_estimators': 200,      # 减少树数量
        'learning_rate': 0.05,    # 降低学习率
        'subsample': 0.8,         # 子采样
        'colsample_bytree': 0.8,  # 特征子采样
        'min_child_weight': 5,    # 增加最小子节点权重
        'reg_alpha': 0.5,         # L1正则化
        'reg_lambda': 2.0,        # L2正则化
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'random_state': 42
    }
    
    # 训练模型
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=0)
    
    # 评估
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_pred_proba_train)
    test_auc = roc_auc_score(y_test, y_pred_proba_test)
    
    # 计算过拟合程度
    overfitting = train_auc - test_auc
    
    print(f"\n📊 训练结果:")
    print(f"训练AUC: {train_auc:.4f}")
    print(f"测试AUC: {test_auc:.4f}")
    print(f"过拟合程度: {overfitting:.4f}")
    
    # 过拟合判断
    if overfitting > 0.05:
        print(f"⚠️ 警告: 过拟合程度 {overfitting:.4f} > 0.05，模型可能过拟合")
    elif overfitting > 0.02:
        print(f"⚠️ 注意: 过拟合程度 {overfitting:.4f} > 0.02，需要关注")
    else:
        print(f"✅ 过拟合控制良好: {overfitting:.4f} ≤ 0.02")
    
    # 高置信度预测分析
    high_conf_mask_test = (y_pred_proba_test > 0.8) | (y_pred_proba_test < 0.2)
    
    if high_conf_mask_test.sum() > 0:
        high_conf_auc = roc_auc_score(y_test[high_conf_mask_test], y_pred_proba_test[high_conf_mask_test])
        high_conf_acc = ((y_pred_proba_test[high_conf_mask_test] > 0.5) == y_test[high_conf_mask_test]).mean()
        
        print(f"\n🎯 高置信度预测分析:")
        print(f"高置信度样本数: {high_conf_mask_test.sum()}")
        print(f"高置信度AUC: {high_conf_auc:.4f}")
        print(f"高置信度准确率: {high_conf_acc:.4f}")
    
    # 保存模型
    model.save_model('xgb_15m_simple.bin')
    print(f"\n✅ 模型已保存: xgb_15m_simple.bin")
    
    # 保存预测结果用于后续分析
    results_df = pd.DataFrame({
        'timestamp': df_sorted['timestamp'][split_idx:],
        'true_label': y_test,
        'pred_proba': y_pred_proba_test,
        'pred_label': (y_pred_proba_test > 0.5).astype(int)
    })
    
    results_df.to_csv('xgb_15m_simple_results.csv', index=False)
    print("✅ 预测结果已保存: xgb_15m_simple_results.csv")
    
    return model, test_auc, overfitting

if __name__ == "__main__":
    main() 