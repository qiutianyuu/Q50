#!/usr/bin/env python3
"""
实时XGBoost模型训练脚本
使用WebSocket收集的数据训练模型
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import glob
import os
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_websocket_data():
    """加载WebSocket收集的数据"""
    # 加载特征数据
    feature_files = glob.glob('data/analysis/micro_features_*.parquet')
    if not feature_files:
        logger.error("未找到特征文件")
        return None, None
    
    # 使用最新的特征文件
    latest_feature_file = max(feature_files, key=os.path.getctime)
    logger.info(f"加载特征文件: {latest_feature_file}")
    
    features_df = pd.read_parquet(latest_feature_file)
    logger.info(f"特征数据形状: {features_df.shape}")
    
    return features_df

def prepare_features_and_labels(features_df):
    """准备特征和标签"""
    # 移除包含NaN的行
    features_df = features_df.dropna()
    
    # 选择特征列（排除时间戳和标签相关列）
    exclude_cols = ['timestamp', 'datetime', 'label', 'target', 'price_change', 'price_change_pct', 
                   'future_price_change', 'label_binary', 'label_three']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    X = features_df[feature_cols]
    y = features_df['label_binary'] if 'label_binary' in features_df.columns else None
    
    if y is None:
        logger.error("未找到标签列")
        return None, None, None
    
    logger.info(f"特征数量: {len(feature_cols)}")
    logger.info(f"样本数量: {len(X)}")
    logger.info(f"标签分布: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols

def train_xgb_model(X, y, feature_cols):
    """训练XGBoost模型"""
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"训练集大小: {X_train.shape}")
    logger.info(f"测试集大小: {X_test.shape}")
    
    # 设置XGBoost参数
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'early_stopping_rounds': 10
    }
    
    # 训练模型
    logger.info("开始训练XGBoost模型...")
    model = xgb.XGBClassifier(**params)
    
    # 使用验证集进行早停
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=True
    )
    
    # 评估模型
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"测试集准确率: {accuracy:.4f}")
    logger.info(f"测试集AUC: {auc:.4f}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 特征重要性:")
    for i, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, feature_importance, (X_test, y_test, y_pred, y_pred_proba)

def save_model_and_results(model, feature_importance, results, feature_cols):
    """保存模型和结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型
    model_path = f"xgb_realtime_{timestamp}.bin"
    model.save_model(model_path)
    logger.info(f"模型已保存: {model_path}")
    
    # 保存特征重要性
    importance_path = f"feature_importance_realtime_{timestamp}.csv"
    feature_importance.to_csv(importance_path, index=False)
    logger.info(f"特征重要性已保存: {importance_path}")
    
    # 保存结果
    X_test, y_test, y_pred, y_pred_proba = results
    results_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    })
    results_path = f"model_results_realtime_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"预测结果已保存: {results_path}")
    
    # 保存特征列表
    feature_list_path = f"feature_list_realtime_{timestamp}.json"
    import json
    with open(feature_list_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    logger.info(f"特征列表已保存: {feature_list_path}")
    
    return model_path, feature_list_path

def main():
    """主函数"""
    logger.info("开始实时XGBoost模型训练...")
    
    # 1. 加载数据
    features_df = load_websocket_data()
    if features_df is None:
        return
    
    # 2. 准备特征和标签
    X, y, feature_cols = prepare_features_and_labels(features_df)
    if X is None:
        return
    
    # 3. 训练模型
    model, feature_importance, results = train_xgb_model(X, y, feature_cols)
    
    # 4. 保存结果
    model_path, feature_list_path = save_model_and_results(
        model, feature_importance, results, feature_cols
    )
    
    logger.info("实时XGBoost模型训练完成!")
    logger.info(f"模型文件: {model_path}")
    logger.info(f"特征列表: {feature_list_path}")

if __name__ == "__main__":
    main() 