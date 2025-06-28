#!/usr/bin/env python3
"""
Optuna超参数优化 - XGBoost
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna
import warnings
warnings.filterwarnings('ignore')

def optimize_xgb_15m():
    """优化15m XGBoost超参数"""
    print("🔧 优化15m XGBoost超参数...")
    
    # 读取数据
    df = pd.read_parquet('/Users/qiutianyu/data/processed/features_15m_2023_2025.parquet')
    
    # 准备特征
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'trend_1h', 'trend_4h']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"样本数量: {len(X)}")
    
    def objective(trial):
        # 超参数搜索空间
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'random_state': 42
        }
        
        # 5折交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=0)
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            scores.append(auc)
        
        return np.mean(scores)
    
    # 创建study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    print(f"🎯 最佳AUC: {study.best_value:.4f}")
    print(f"最佳参数: {study.best_params}")
    
    # 保存结果
    best_params = study.best_params
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'random_state': 42
    })
    
    # 使用最佳参数训练最终模型
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X, y, verbose=0)
    
    # 保存模型
    final_model.save_model('xgb_15m_optimized.bin')
    print("✅ 优化后的15m模型已保存: xgb_15m_optimized.bin")
    
    return best_params, study.best_value

def optimize_xgb_5m():
    """优化5m XGBoost超参数"""
    print("\n🔧 优化5m XGBoost超参数...")
    
    # 读取数据
    df = pd.read_parquet('/Users/qiutianyu/data/processed/features_5m_2023_2025.parquet')
    
    # 准备特征
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'trend_1h', 'trend_4h']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"样本数量: {len(X)}")
    
    def objective(trial):
        # 超参数搜索空间
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'random_state': 42
        }
        
        # 5折交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=0)
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            scores.append(auc)
        
        return np.mean(scores)
    
    # 创建study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)  # 5m数据量大，减少trial数
    
    print(f"🎯 最佳AUC: {study.best_value:.4f}")
    print(f"最佳参数: {study.best_params}")
    
    # 保存结果
    best_params = study.best_params
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'random_state': 42
    })
    
    # 使用最佳参数训练最终模型
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X, y, verbose=0)
    
    # 保存模型
    final_model.save_model('xgb_5m_optimized.bin')
    print("✅ 优化后的5m模型已保存: xgb_5m_optimized.bin")
    
    return best_params, study.best_value

def main():
    print("🚀 开始Optuna超参数优化...")
    
    # 优化15m模型
    best_params_15m, best_auc_15m = optimize_xgb_15m()
    
    # 优化5m模型
    best_params_5m, best_auc_5m = optimize_xgb_5m()
    
    # 结果总结
    print("\n📊 优化结果总结:")
    print(f"15m最佳AUC: {best_auc_15m:.4f}")
    print(f"5m最佳AUC: {best_auc_5m:.4f}")
    
    print("\n🎉 超参数优化完成!")

if __name__ == "__main__":
    main() 