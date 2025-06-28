#!/usr/bin/env python3
"""
RexKing – XGBoost Training 15m (Walk-Forward)

读取 15m 特征数据, 使用walk-forward验证训练XGBoost模型, 输出AUC和SHAP重要性.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
import shap
import warnings
warnings.filterwarnings('ignore')

# ---------- 路径配置 ----------
DATA_DIR = Path("/Users/qiutianyu/data/processed")
FEATURES_FILE = DATA_DIR / "features_15m_2023_2025.parquet"
MODEL_FILE = "xgb_15m_model.bin"

# ---------- 数据预处理 ----------
def prepare_data(df: pd.DataFrame) -> tuple:
    """准备训练数据"""
    # 过滤数值特征
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'trend_1h', 'trend_4h']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"样本数量: {len(X)}")
    print(f"标签分布: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols, feature_cols

# ---------- Walk-Forward 分割 ----------
def create_walk_forward_splits(df: pd.DataFrame, train_days: int = 180, test_days: int = 30, offset_days: int = 30):
    """创建walk-forward分割（滚动窗口）"""
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # 计算时间范围
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()
    
    print(f"数据时间范围: {start_date} 到 {end_date}")
    
    splits = []
    current_start = start_date
    
    while current_start < end_date:
        # 训练集结束时间
        train_end = current_start + pd.Timedelta(days=train_days)
        # 测试集结束时间
        test_end = train_end + pd.Timedelta(days=test_days)
        
        if test_end > end_date:
            break
            
        # 获取训练集和测试集索引
        train_mask = (df['timestamp'] >= current_start) & (df['timestamp'] < train_end)
        test_mask = (df['timestamp'] >= train_end) & (df['timestamp'] < test_end)
        
        train_indices = df[train_mask].index.tolist()
        test_indices = df[test_mask].index.tolist()
        
        # 确保有足够的样本
        if len(train_indices) >= 1000 and len(test_indices) >= 200:
            splits.append({
                'train_indices': train_indices,
                'test_indices': test_indices,
                'train_start': current_start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end
            })
        
        # 移动到下一个分割（滚动窗口）
        current_start = current_start + pd.Timedelta(days=offset_days)
    
    print(f"创建了 {len(splits)} 个walk-forward分割")
    
    # 如果没有创建分割，使用简单的时间分割
    if len(splits) == 0:
        print("使用简单的时间分割...")
        total_samples = len(df)
        train_size = int(total_samples * 0.7)
        test_size = int(total_samples * 0.15)
        
        splits = [{
            'train_indices': list(range(0, train_size)),
            'test_indices': list(range(train_size, train_size + test_size)),
            'train_start': df.iloc[0]['timestamp'],
            'train_end': df.iloc[train_size-1]['timestamp'],
            'test_start': df.iloc[train_size]['timestamp'],
            'test_end': df.iloc[train_size + test_size - 1]['timestamp']
        }]
        print(f"创建了 {len(splits)} 个简单分割")
    
    return splits

# ---------- 训练模型 ----------
def train_model(X: pd.DataFrame, y: pd.Series, feature_names: list) -> xgb.XGBClassifier:
    """训练XGBoost模型"""
    # 模型参数
    params = {
        'max_depth': 4,
        'n_estimators': 200,
        'learning_rate': 0.05,
        'reg_lambda': 1.5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=0)
    
    return model

# ---------- Walk-Forward 验证 ----------
def walk_forward_validation(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, feature_names: list):
    """执行walk-forward验证"""
    splits = create_walk_forward_splits(df)
    
    results = []
    
    for i, split in enumerate(splits):
        print(f"\n🔄 分割 {i+1}/{len(splits)}")
        print(f"训练期: {split['train_start'].strftime('%Y-%m')} 到 {split['train_end'].strftime('%Y-%m')}")
        print(f"测试期: {split['test_start'].strftime('%Y-%m')} 到 {split['test_end'].strftime('%Y-%m')}")
        
        # 获取训练集和测试集
        X_train = X.iloc[split['train_indices']]
        y_train = y.iloc[split['train_indices']]
        X_test = X.iloc[split['test_indices']]
        y_test = y.iloc[split['test_indices']]
        
        print(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
        
        # 训练模型
        model = train_model(X_train, y_train, feature_names)
        
        # 预测
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # 计算其他指标
        y_pred = (y_pred_proba > 0.5).astype(int)
        accuracy = (y_pred == y_test).mean()
        
        # 高置信度预测分析
        high_conf_mask = (y_pred_proba > 0.7) | (y_pred_proba < 0.3)
        high_conf_accuracy = 0
        if high_conf_mask.sum() > 0:
            high_conf_accuracy = (y_test[high_conf_mask] == (y_pred_proba[high_conf_mask] > 0.5)).mean()
        
        result = {
            'split': i + 1,
            'train_start': split['train_start'],
            'train_end': split['train_end'],
            'test_start': split['test_start'],
            'test_end': split['test_end'],
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'auc': auc,
            'accuracy': accuracy,
            'high_conf_accuracy': high_conf_accuracy,
            'high_conf_samples': high_conf_mask.sum()
        }
        
        results.append(result)
        
        print(f"AUC: {auc:.4f}, 准确率: {accuracy:.4f}, 高置信度准确率: {high_conf_accuracy:.4f}")
    
    return results

# ---------- 结果分析 ----------
def analyze_walk_forward_results(results: list):
    """分析walk-forward结果"""
    df_results = pd.DataFrame(results)
    
    print("\n📊 Walk-Forward 验证结果汇总:")
    print("=" * 80)
    print(f"总分割数: {len(results)}")
    print(f"平均AUC: {df_results['auc'].mean():.4f} ± {df_results['auc'].std():.4f}")
    print(f"平均准确率: {df_results['accuracy'].mean():.4f} ± {df_results['accuracy'].std():.4f}")
    print(f"平均高置信度准确率: {df_results['high_conf_accuracy'].mean():.4f} ± {df_results['high_conf_accuracy'].std():.4f}")
    print(f"平均高置信度样本数: {df_results['high_conf_samples'].mean():.1f}")
    
    print("\n📈 各分割详细结果:")
    print(df_results[['split', 'test_start', 'test_end', 'test_samples', 'auc', 'accuracy', 'high_conf_accuracy']].to_string(index=False))
    
    # 保存结果
    out_csv = DATA_DIR / "walk_forward_15m_results.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n结果已保存: {out_csv}")
    
    return df_results

# ---------- 主流程 ----------
def main():
    print("📥 读取特征数据...")
    df = pd.read_parquet(FEATURES_FILE)
    print(f"数据形状: {df.shape}")
    
    # 准备数据
    X, y, feature_names, _ = prepare_data(df)
    
    # Walk-Forward验证
    results = walk_forward_validation(df, X, y, feature_names)
    
    # 分析结果
    df_results = analyze_walk_forward_results(results)
    
    # 使用最后一个分割的模型作为最终模型
    if results:
        last_split = results[-1]
        print(f"\n🎯 使用最后一个分割训练最终模型...")
        
        # 获取最后一个分割的训练数据
        splits = create_walk_forward_splits(df)
        last_split_data = splits[-1]
        
        X_final_train = X.iloc[last_split_data['train_indices']]
        y_final_train = y.iloc[last_split_data['train_indices']]
        
        # 训练最终模型
        final_model = train_model(X_final_train, y_final_train, feature_names)
        final_model.save_model(MODEL_FILE)
        print(f"最终模型已保存: {MODEL_FILE}")
    
    print("\n🎉 Walk-Forward验证完成!")

if __name__ == "__main__":
    main() 