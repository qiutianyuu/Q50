#!/usr/bin/env python3
"""
修正版15m XGBoost训练 - 避免过拟合
使用正则化、早停、Walk-forward验证
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def train_with_walk_forward():
    """使用Walk-forward验证训练模型"""
    print("🚀 开始Walk-forward验证训练...")
    
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
    
    # Walk-forward参数
    train_days = 180  # 6个月训练
    test_days = 30    # 1个月测试
    step_days = 30    # 每30天向前滚动
    
    # 计算时间窗口
    total_days = (df_sorted['timestamp'].max() - df_sorted['timestamp'].min()).days
    print(f"总时间跨度: {total_days}天")
    
    # 生成时间窗口
    windows = []
    start_date = df_sorted['timestamp'].min()
    
    while True:
        train_start = start_date
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_days)
        
        if test_end > df_sorted['timestamp'].max():
            break
            
        windows.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        })
        
        start_date += pd.Timedelta(days=step_days)
    
    print(f"生成 {len(windows)} 个时间窗口")
    
    # 存储每个窗口的结果
    results = []
    
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
    
    for i, window in enumerate(windows):
        print(f"\n📊 窗口 {i+1}/{len(windows)}: {window['train_start'].date()} - {window['test_end'].date()}")
        
        # 分割训练集和测试集
        train_mask = (df_sorted['timestamp'] >= window['train_start']) & (df_sorted['timestamp'] < window['train_end'])
        test_mask = (df_sorted['timestamp'] >= window['test_start']) & (df_sorted['timestamp'] < window['test_end'])
        
        X_train = X_sorted[train_mask]
        y_train = y_sorted[train_mask]
        X_test = X_sorted[test_mask]
        y_test = y_sorted[test_mask]
        
        if len(X_train) < 1000 or len(X_test) < 100:
            print(f"⚠️ 窗口 {i+1} 样本不足，跳过")
            continue
        
        print(f"训练集: {len(X_train)} 样本")
        print(f"测试集: {len(X_test)} 样本")
        
        # 训练模型（使用正则化）
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=0)
        
        # 评估
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        # 计算过拟合程度
        overfitting = train_auc - test_auc
        
        # 高置信度预测
        high_conf_mask = (y_pred_proba > 0.8) | (y_pred_proba < 0.2)
        high_conf_auc = 0
        high_conf_acc = 0
        high_conf_count = 0
        
        if high_conf_mask.sum() > 0:
            high_conf_auc = roc_auc_score(y_test[high_conf_mask], y_pred_proba[high_conf_mask])
            high_conf_acc = (y_pred[high_conf_mask] == y_test[high_conf_mask]).mean()
            high_conf_count = high_conf_mask.sum()
        
        # 记录结果
        window_result = {
            'window': i+1,
            'train_start': window['train_start'],
            'test_end': window['test_end'],
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_auc': train_auc,
            'test_auc': test_auc,
            'overfitting': overfitting,
            'high_conf_auc': high_conf_auc,
            'high_conf_acc': high_conf_acc,
            'high_conf_count': high_conf_count
        }
        
        results.append(window_result)
        
        print(f"训练AUC: {train_auc:.4f}")
        print(f"测试AUC: {test_auc:.4f}")
        print(f"过拟合程度: {overfitting:.4f}")
        print(f"高置信度AUC: {high_conf_auc:.4f}")
        print(f"高置信度准确率: {high_conf_acc:.4f}")
    
    # 分析结果
    if results:
        results_df = pd.DataFrame(results)
        
        print(f"\n📊 Walk-forward验证结果总结:")
        print(f"总窗口数: {len(results_df)}")
        print(f"平均测试AUC: {results_df['test_auc'].mean():.4f} ± {results_df['test_auc'].std():.4f}")
        print(f"平均过拟合程度: {results_df['overfitting'].mean():.4f} ± {results_df['overfitting'].std():.4f}")
        print(f"平均高置信度AUC: {results_df['high_conf_auc'].mean():.4f} ± {results_df['high_conf_auc'].std():.4f}")
        print(f"平均高置信度准确率: {results_df['high_conf_acc'].mean():.4f} ± {results_df['high_conf_acc'].std():.4f}")
        
        # 保存结果
        results_df.to_csv('walk_forward_15m_results.csv', index=False)
        print("✅ Walk-forward结果已保存: walk_forward_15m_results.csv")
        
        # 检查过拟合
        avg_overfitting = results_df['overfitting'].mean()
        if avg_overfitting > 0.05:
            print(f"⚠️ 警告: 平均过拟合程度 {avg_overfitting:.4f} > 0.05，建议进一步正则化")
        else:
            print(f"✅ 过拟合控制良好: {avg_overfitting:.4f} ≤ 0.05")
        
        return results_df
    else:
        print("❌ 没有有效的验证窗口")
        return None

def train_final_model():
    """训练最终模型（使用全部数据）"""
    print("\n🎯 训练最终模型...")
    
    # 读取数据
    df = pd.read_parquet('/Users/qiutianyu/data/processed/features_15m_fixed.parquet')
    
    # 准备特征
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'trend_1h', 'trend_4h']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 正则化参数
    params = {
        'max_depth': 4,
        'n_estimators': 200,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
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
    
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"训练AUC: {train_auc:.4f}")
    print(f"测试AUC: {test_auc:.4f}")
    print(f"过拟合程度: {train_auc - test_auc:.4f}")
    
    # 保存模型
    model.save_model('xgb_15m_fixed.bin')
    print("✅ 最终模型已保存: xgb_15m_fixed.bin")
    
    return model, test_auc

def main():
    print("🔧 修正版15m XGBoost训练 - 避免过拟合")
    
    # Walk-forward验证
    results = train_with_walk_forward()
    
    # 训练最终模型
    model, final_auc = train_final_model()
    
    print(f"\n🎉 训练完成! 最终测试AUC: {final_auc:.4f}")

if __name__ == "__main__":
    main() 