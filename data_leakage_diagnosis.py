#!/usr/bin/env python3
"""
数据泄漏诊断 - 找出过拟合的根本原因
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def diagnose_leakage():
    """诊断数据泄漏"""
    print("🔍 数据泄漏诊断...")
    
    # 读取数据
    df = pd.read_parquet('/Users/qiutianyu/data/processed/features_15m_fixed.parquet')
    
    # 准备特征
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'trend_1h', 'trend_4h']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"样本数量: {len(X)}")
    
    # 检查标签分布
    print(f"\n📊 标签分布:")
    print(f"做多信号 (1): {y.sum()} ({y.mean():.2%})")
    print(f"做空信号 (0): {(y==0).sum()} ({(y==0).mean():.2%})")
    
    # 检查时间分布
    df_sorted = df.sort_values('timestamp')
    print(f"\n📅 时间分布:")
    print(f"开始时间: {df_sorted['timestamp'].min()}")
    print(f"结束时间: {df_sorted['timestamp'].max()}")
    print(f"时间跨度: {(df_sorted['timestamp'].max() - df_sorted['timestamp'].min()).days} 天")
    
    # 检查特征与标签的相关性
    print(f"\n🔗 特征与标签相关性 (Top-10):")
    correlations = []
    for col in feature_cols:
        corr = abs(X[col].corr(y))
        correlations.append((col, corr))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    for i, (col, corr) in enumerate(correlations[:10]):
        print(f"{i+1:2d}. {col:20s}: {corr:.4f}")
    
    # 检查可疑特征
    suspicious_features = []
    for col, corr in correlations:
        if corr > 0.1:  # 相关性超过0.1的特征
            suspicious_features.append((col, corr))
    
    print(f"\n⚠️ 高相关性特征 (|corr| > 0.1): {len(suspicious_features)} 个")
    for col, corr in suspicious_features[:10]:
        print(f"  {col}: {corr:.4f}")
    
    # 检查收益率特征
    ret_features = [col for col in feature_cols if 'ret_' in col]
    print(f"\n📈 收益率特征分析:")
    for col in ret_features:
        corr = abs(X[col].corr(y))
        print(f"  {col}: {corr:.4f}")
    
    # 检查波动率特征
    vol_features = [col for col in feature_cols if 'volatility' in col]
    print(f"\n📊 波动率特征分析:")
    for col in vol_features:
        corr = abs(X[col].corr(y))
        print(f"  {col}: {corr:.4f}")
    
    return suspicious_features

def test_feature_removal():
    """测试移除可疑特征后的效果"""
    print("\n🧪 测试特征移除效果...")
    
    # 读取数据
    df = pd.read_parquet('/Users/qiutianyu/data/processed/features_15m_fixed.parquet')
    
    # 准备特征
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'trend_1h', 'trend_4h']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    # 时间排序
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    X_sorted = X.loc[df_sorted.index]
    y_sorted = y.loc[df_sorted.index]
    
    # 分割数据
    split_idx = int(len(df_sorted) * 0.8)
    X_train = X_sorted[:split_idx]
    y_train = y_sorted[:split_idx]
    X_test = X_sorted[split_idx:]
    y_test = y_sorted[split_idx:]
    
    # 测试不同特征组合
    test_cases = [
        ("全部特征", feature_cols),
        ("移除收益率特征", [col for col in feature_cols if 'ret_' not in col]),
        ("移除波动率特征", [col for col in feature_cols if 'volatility' not in col]),
        ("只保留技术指标", [col for col in feature_cols if any(x in col for x in ['rsi', 'macd', 'bb', 'adx', 'stoch', 'atr', 'ema'])]),
    ]
    
    results = []
    
    for case_name, selected_features in test_cases:
        print(f"\n🔧 测试: {case_name}")
        print(f"特征数量: {len(selected_features)}")
        
        X_train_subset = X_train[selected_features]
        X_test_subset = X_test[selected_features]
        
        # 训练模型
        params = {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_subset, y_train, verbose=0)
        
        # 评估
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train_subset)[:, 1])
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test_subset)[:, 1])
        overfitting = train_auc - test_auc
        
        results.append({
            'case': case_name,
            'features': len(selected_features),
            'train_auc': train_auc,
            'test_auc': test_auc,
            'overfitting': overfitting
        })
        
        print(f"训练AUC: {train_auc:.4f}")
        print(f"测试AUC: {test_auc:.4f}")
        print(f"过拟合程度: {overfitting:.4f}")
    
    # 总结
    print(f"\n📊 特征移除测试总结:")
    for result in results:
        status = "✅" if result['overfitting'] < 0.05 else "⚠️"
        print(f"{status} {result['case']}: 测试AUC={result['test_auc']:.4f}, 过拟合={result['overfitting']:.4f}")
    
    return results

def main():
    print("🔍 数据泄漏深度诊断")
    
    # 诊断泄漏
    suspicious_features = diagnose_leakage()
    
    # 测试特征移除
    results = test_feature_removal()
    
    print(f"\n🎯 诊断结论:")
    if suspicious_features:
        print(f"发现 {len(suspicious_features)} 个高相关性特征，可能存在数据泄漏")
    else:
        print("未发现明显的数据泄漏特征")
    
    # 找出最佳特征组合
    best_result = min(results, key=lambda x: x['overfitting'])
    print(f"最佳特征组合: {best_result['case']}")
    print(f"测试AUC: {best_result['test_auc']:.4f}")
    print(f"过拟合程度: {best_result['overfitting']:.4f}")

if __name__ == "__main__":
    main() 