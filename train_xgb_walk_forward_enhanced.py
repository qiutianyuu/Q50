#!/usr/bin/env python3
"""
增强版XGBoost Walk-Forward训练 - 使用成本感知标签和增强特征
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
    
    # 合并特征和标签
    merged_df = features_df.merge(labels_df[['timestamp', 'label']], on='timestamp', how='inner')
    
    print(f"📊 合并后数据形状: {merged_df.shape}")
    print(f"📅 时间范围: {merged_df['timestamp'].min()} 到 {merged_df['timestamp'].max()}")
    
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

def walk_forward_validation(df, train_days=180, test_days=30, step_days=30):
    """Walk-Forward验证"""
    print(f"🔄 Walk-Forward验证: {train_days}天训练, {test_days}天测试, {step_days}天步长")
    
    # 按时间排序
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 计算时间窗口
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()
    
    # 生成时间窗口
    windows = []
    current_start = start_date
    while current_start + pd.Timedelta(days=train_days + test_days) <= end_date:
        train_start = current_start
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_days)
        
        windows.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        })
        
        current_start += pd.Timedelta(days=step_days)
    
    print(f"📊 生成 {len(windows)} 个时间窗口")
    
    results = []
    
    for i, window in enumerate(windows):
        print(f"\n🔄 窗口 {i+1}/{len(windows)}")
        print(f"训练: {window['train_start'].date()} - {window['train_end'].date()}")
        print(f"测试: {window['test_start'].date()} - {window['test_end'].date()}")
        
        # 分割数据
        train_mask = (df['timestamp'] >= window['train_start']) & (df['timestamp'] < window['train_end'])
        test_mask = (df['timestamp'] >= window['test_start']) & (df['timestamp'] < window['test_end'])
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if len(train_df) < 1000 or len(test_df) < 100:
            print("⚠️ 样本不足，跳过")
            continue
        
        # 准备特征
        X_train, y_train, feature_cols = prepare_features(train_df)
        X_test, y_test, _ = prepare_features(test_df)
        
        # 只保留交易信号
        train_trade_mask = y_train != -1
        test_trade_mask = y_test != -1
        
        if train_trade_mask.sum() < 500 or test_trade_mask.sum() < 50:
            print("⚠️ 交易信号不足，跳过")
            continue
        
        X_train_trade = X_train[train_trade_mask]
        y_train_trade = y_train[train_trade_mask]
        X_test_trade = X_test[test_trade_mask]
        y_test_trade = y_test[test_trade_mask]
        
        print(f"训练样本: {len(X_train_trade)} (交易信号: {train_trade_mask.sum()})")
        print(f"测试样本: {len(X_test_trade)} (交易信号: {test_trade_mask.sum()})")
        
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
            X_train_trade, y_train_trade,
            eval_set=[(X_test_trade, y_test_trade)],
            verbose=0
        )
        
        # 预测
        train_proba = model.predict_proba(X_train_trade)[:, 1]
        test_proba = model.predict_proba(X_test_trade)[:, 1]
        
        # 评估
        train_auc = roc_auc_score(y_train_trade, train_proba)
        test_auc = roc_auc_score(y_test_trade, test_proba)
        
        # 概率校准
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(train_proba, y_train_trade)
        test_proba_calibrated = calibrator.predict(test_proba)
        test_auc_calibrated = roc_auc_score(y_test_trade, test_proba_calibrated)
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        result = {
            'window': i + 1,
            'train_start': window['train_start'],
            'train_end': window['train_end'],
            'test_start': window['test_start'],
            'test_end': window['test_end'],
            'train_samples': len(X_train_trade),
            'test_samples': len(X_test_trade),
            'train_auc': train_auc,
            'test_auc': test_auc,
            'test_auc_calibrated': test_auc_calibrated,
            'overfitting': train_auc - test_auc,
            'top_features': feature_importance.head(10)['feature'].tolist(),
            'model': model,
            'calibrator': calibrator
        }
        
        results.append(result)
        
        print(f"训练AUC: {train_auc:.4f}")
        print(f"测试AUC: {test_auc:.4f}")
        print(f"校准后AUC: {test_auc_calibrated:.4f}")
        print(f"过拟合程度: {train_auc - test_auc:.4f}")
        print(f"Top特征: {', '.join(feature_importance.head(3)['feature'].tolist())}")
    
    return results

def analyze_results(results):
    """分析Walk-Forward结果"""
    if not results:
        print("❌ 没有有效结果")
        return
    
    print(f"\n📊 Walk-Forward结果分析")
    print(f"有效窗口数: {len(results)}")
    
    # 统计指标
    test_aucs = [r['test_auc'] for r in results]
    test_aucs_calibrated = [r['test_auc_calibrated'] for r in results]
    overfitting = [r['overfitting'] for r in results]
    
    print(f"\n📈 AUC统计:")
    print(f"测试AUC均值: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
    print(f"校准后AUC均值: {np.mean(test_aucs_calibrated):.4f} ± {np.std(test_aucs_calibrated):.4f}")
    print(f"过拟合均值: {np.mean(overfitting):.4f} ± {np.std(overfitting):.4f}")
    
    print(f"\n📊 稳定性分析:")
    print(f"AUC > 0.55的窗口: {sum(1 for auc in test_aucs if auc > 0.55)}/{len(test_aucs)}")
    print(f"AUC > 0.57的窗口: {sum(1 for auc in test_aucs if auc > 0.57)}/{len(test_aucs)}")
    print(f"过拟合 < 0.05的窗口: {sum(1 for of in overfitting if of < 0.05)}/{len(overfitting)}")
    
    # 特征重要性统计
    all_features = []
    for result in results:
        all_features.extend(result['top_features'][:5])
    
    feature_counts = pd.Series(all_features).value_counts()
    print(f"\n🏆 最常出现的重要特征:")
    for feature, count in feature_counts.head(10).items():
        print(f"  {feature}: {count}次")
    
    return results

def save_results(results, output_path):
    """保存结果"""
    if not results:
        return
    
    # 保存详细结果
    results_df = pd.DataFrame([
        {
            'window': r['window'],
            'train_start': r['train_start'],
            'train_end': r['train_end'],
            'test_start': r['test_start'],
            'test_end': r['test_end'],
            'train_samples': r['train_samples'],
            'test_samples': r['test_samples'],
            'train_auc': r['train_auc'],
            'test_auc': r['test_auc'],
            'test_auc_calibrated': r['test_auc_calibrated'],
            'overfitting': r['overfitting']
        }
        for r in results
    ])
    
    results_df.to_csv(output_path, index=False)
    print(f"✅ 结果已保存: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='增强版XGBoost Walk-Forward训练')
    parser.add_argument('--features', type=str, required=True, help='特征文件路径')
    parser.add_argument('--labels', type=str, required=True, help='标签文件路径')
    parser.add_argument('--output', type=str, default='walk_forward_results_enhanced.csv', help='输出文件路径')
    parser.add_argument('--train_days', type=int, default=180, help='训练天数')
    parser.add_argument('--test_days', type=int, default=30, help='测试天数')
    parser.add_argument('--step_days', type=int, default=30, help='步长天数')
    
    args = parser.parse_args()
    
    print("🚀 增强版XGBoost Walk-Forward训练")
    print(f"📁 特征文件: {args.features}")
    print(f"📁 标签文件: {args.labels}")
    print(f"⏱️ 训练天数: {args.train_days}")
    print(f"⏱️ 测试天数: {args.test_days}")
    print(f"⏱️ 步长天数: {args.step_days}")
    
    # 加载数据
    df = load_data(args.features, args.labels)
    
    # Walk-Forward验证
    results = walk_forward_validation(df, args.train_days, args.test_days, args.step_days)
    
    # 分析结果
    analyze_results(results)
    
    # 保存结果
    save_results(results, args.output)

if __name__ == "__main__":
    main() 