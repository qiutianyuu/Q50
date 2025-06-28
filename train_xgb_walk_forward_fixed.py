#!/usr/bin/env python3
"""
修复版XGBoost Walk-Forward训练 - 正确处理三分类标签
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
    labels_df['timestamp'] = pd.to_datetime(labels_df['timestamp'], format='mixed')
    # 强制标签时间戳为UTC
    if labels_df['timestamp'].dt.tz is None:
        labels_df['timestamp'] = labels_df['timestamp'].dt.tz_localize('UTC')
    print(f"特征时间戳格式: {features_df['timestamp'].dtype}")
    print(f"标签时间戳格式: {labels_df['timestamp'].dtype}")
    
    # 合并特征和标签
    merged_df = features_df.merge(labels_df[['timestamp', 'label']], on='timestamp', how='inner')
    
    print(f"📊 合并后数据形状: {merged_df.shape}")
    print(f"📅 时间范围: {merged_df['timestamp'].min()} 到 {merged_df['timestamp'].max()}")
    
    # 显示标签分布
    print(f"📊 标签分布:")
    print(merged_df['label'].value_counts())
    print(merged_df['label'].value_counts(normalize=True) * 100)
    
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
    
    # 重新映射标签：-1->0, 0->1, 1->2
    y = y.map({-1: 0, 0: 1, 1: 2})
    
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
        
        # 检查标签分布
        print(f"训练标签分布: {y_train.value_counts().to_dict()}")
        print(f"测试标签分布: {y_test.value_counts().to_dict()}")
        
        # 确保有足够的样本
        if len(y_train.unique()) < 2 or len(y_test.unique()) < 2:
            print("⚠️ 标签类别不足，跳过")
            continue
        
        # 训练模型 - 使用三分类
        model = xgb.XGBClassifier(
            max_depth=4,
            n_estimators=200,
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=42,
            eval_metric='mlogloss',  # 多分类使用mlogloss
            early_stopping_rounds=50
        )
        
        # 训练
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=0
        )
        
        # 预测
        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)
        
        # 对于三分类，我们需要计算每个类别的AUC
        # 这里我们主要关注多头(1)和空头(-1)的预测
        if 1 in y_train.unique() and -1 in y_train.unique():
            # 多头vs其他
            y_train_long = (y_train == 1).astype(int)
            y_test_long = (y_test == 1).astype(int)
            long_proba = test_proba[:, 1] if test_proba.shape[1] > 1 else test_proba[:, 0]
            long_auc = roc_auc_score(y_test_long, long_proba)
            
            # 空头vs其他
            y_train_short = (y_test == -1).astype(int)
            y_test_short = (y_test == -1).astype(int)
            short_proba = test_proba[:, 2] if test_proba.shape[1] > 2 else test_proba[:, 0]
            short_auc = roc_auc_score(y_test_short, short_proba)
            
            test_auc = (long_auc + short_auc) / 2
        else:
            test_auc = 0.5
        
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
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'test_auc': test_auc,
            'top_features': feature_importance.head(10)['feature'].tolist(),
            'model': model
        }
        
        results.append(result)
        
        print(f"测试AUC: {test_auc:.4f}")
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
    
    print(f"\n📈 AUC统计:")
    print(f"测试AUC均值: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
    
    print(f"\n📊 稳定性分析:")
    print(f"AUC > 0.55的窗口: {sum(1 for auc in test_aucs if auc > 0.55)}/{len(test_aucs)}")
    print(f"AUC > 0.57的窗口: {sum(1 for auc in test_aucs if auc > 0.57)}/{len(test_aucs)}")
    
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
            'test_auc': r['test_auc']
        }
        for r in results
    ])
    
    results_df.to_csv(output_path, index=False)
    print(f"✅ 结果已保存: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='修复版XGBoost Walk-Forward训练')
    parser.add_argument('--features', type=str, required=True, help='特征文件路径')
    parser.add_argument('--labels', type=str, required=True, help='标签文件路径')
    parser.add_argument('--output', type=str, default='walk_forward_results_fixed.csv', help='输出文件路径')
    parser.add_argument('--train_days', type=int, default=180, help='训练天数')
    parser.add_argument('--test_days', type=int, default=30, help='测试天数')
    parser.add_argument('--step_days', type=int, default=30, help='步长天数')
    
    args = parser.parse_args()
    
    print("🚀 修复版XGBoost Walk-Forward训练")
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