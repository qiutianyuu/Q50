#!/usr/bin/env python3
"""
Optuna超参数优化 - 使用筛选后的特征优化XGBoost模型
"""

import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def load_data(features_path, labels_path):
    """加载数据"""
    print(f"📁 加载特征: {features_path}")
    features_df = pd.read_parquet(features_path)
    
    print(f"📁 加载标签: {labels_path}")
    labels_df = pd.read_csv(labels_path)
    
    # 确保timestamp列类型一致
    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
    labels_df['timestamp'] = pd.to_datetime(labels_df['timestamp'])
    
    # 移除时区信息以确保匹配
    features_df['timestamp'] = features_df['timestamp'].dt.tz_localize(None)
    labels_df['timestamp'] = labels_df['timestamp'].dt.tz_localize(None)
    
    # 合并数据
    df = features_df.merge(labels_df[['timestamp', 'label']], on='timestamp', how='inner')
    
    # 准备特征和标签
    feature_cols = [col for col in df.columns if col not in 
                   ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'date', 'label']]
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    # 转换标签为二进制格式 (-1 -> 0, 1 -> 1)
    y = (y == 1).astype(int)
    
    print(f"📊 数据形状: {X.shape}")
    print(f"📊 特征数量: {len(feature_cols)}")
    print(f"📊 标签分布: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols

def objective(trial, X, y, n_splits=5):
    """Optuna目标函数"""
    
    # 超参数搜索空间
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1,
        
        # 核心参数
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        
        # 正则化参数
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        
        # 其他参数
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0),
    }
    
    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 训练模型
        model = xgb.XGBClassifier(**params, early_stopping_rounds=50)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # 预测和评估
        y_pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred)
        scores.append(score)
    
    return np.mean(scores)

def optimize_hyperparameters(X, y, n_trials=100, timeout=3600):
    """优化超参数"""
    print(f"🔍 开始超参数优化...")
    print(f"📊 试验次数: {n_trials}")
    print(f"⏱️ 超时时间: {timeout}秒")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    print(f"✅ 优化完成!")
    print(f"🏆 最佳AUC: {study.best_value:.4f}")
    print(f"🎯 最佳参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study

def train_final_model(X, y, best_params, output_path):
    """训练最终模型"""
    print("🚀 训练最终模型...")
    
    # 使用最佳参数训练模型
    final_model = xgb.XGBClassifier(**best_params, random_state=42)
    final_model.fit(X, y)
    
    # 保存模型
    final_model.save_model(output_path)
    print(f"✅ 模型已保存: {output_path}")
    
    return final_model

def main():
    print("🚀 Optuna超参数优化")
    
    # 优化15m模型
    print("\n📊 优化15m模型...")
    X_15m, y_15m, feature_cols_15m = load_data(
        'data/features_15m_selected.parquet',
        'data/label_15m_cost.csv'
    )
    
    study_15m = optimize_hyperparameters(X_15m, y_15m, n_trials=50, timeout=1800)
    
    # 训练最终15m模型
    final_model_15m = train_final_model(
        X_15m, y_15m, 
        study_15m.best_params,
        'xgb_15m_optuna_optimized.bin'
    )
    
    # 保存优化结果
    optuna_results_15m = {
        'best_auc': study_15m.best_value,
        'best_params': study_15m.best_params,
        'n_trials': len(study_15m.trials),
        'feature_cols': feature_cols_15m
    }
    
    import json
    with open('optuna_results_15m.json', 'w') as f:
        json.dump(optuna_results_15m, f, indent=2, default=str)
    
    # 优化5m模型
    print("\n📊 优化5m模型...")
    X_5m, y_5m, feature_cols_5m = load_data(
        'data/features_5m_selected.parquet',
        'data/label_5m_cost.csv'
    )
    
    study_5m = optimize_hyperparameters(X_5m, y_5m, n_trials=50, timeout=1800)
    
    # 训练最终5m模型
    final_model_5m = train_final_model(
        X_5m, y_5m,
        study_5m.best_params,
        'xgb_5m_optuna_optimized.bin'
    )
    
    # 保存优化结果
    optuna_results_5m = {
        'best_auc': study_5m.best_value,
        'best_params': study_5m.best_params,
        'n_trials': len(study_5m.trials),
        'feature_cols': feature_cols_5m
    }
    
    with open('optuna_results_5m.json', 'w') as f:
        json.dump(optuna_results_5m, f, indent=2, default=str)
    
    print("\n✅ 超参数优化完成！")

if __name__ == "__main__":
    main() 