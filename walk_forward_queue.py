#!/usr/bin/env python3
"""
滚动Walk-Forward训练脚本（集成队列特征）
使用队列模拟器特征进行滚动训练和验证
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import json
import glob
import os
from utils.labeling import make_labels, get_label_stats

class WalkForwardQueue:
    def __init__(self, train_hours: int = 3, test_hours: int = 0.5, step_hours: int = 0.5):
        """
        初始化Walk-Forward训练器
        
        Args:
            train_hours: 训练窗口小时数
            test_hours: 测试窗口小时数
            step_hours: 滚动步长小时数
        """
        self.train_hours = train_hours
        self.test_hours = test_hours
        self.step_hours = step_hours
        
    def load_latest_queue_features(self):
        """加载最新的队列特征数据"""
        files = glob.glob("data/realtime_features_queue_*.parquet")
        if not files:
            raise FileNotFoundError("No queue features files found")
        latest_file = max(files, key=os.path.getctime)
        print(f"Loading: {latest_file}")
        df = pd.read_parquet(latest_file)
        print(f"Loaded {len(df)} rows")
        return df, latest_file
    
    def create_walk_forward_splits(self, df: pd.DataFrame) -> list:
        """创建Walk-Forward分割"""
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 转换为小时
        train_minutes = int(self.train_hours * 60)
        test_minutes = int(self.test_hours * 60)
        step_minutes = int(self.step_hours * 60)
        
        # 估算每分钟的行数
        total_minutes = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60
        rows_per_minute = len(df) / total_minutes
        
        train_rows = int(train_minutes * rows_per_minute)
        test_rows = int(test_minutes * rows_per_minute)
        step_rows = int(step_minutes * rows_per_minute)
        
        print(f"估算参数: 训练{train_rows}行, 测试{test_rows}行, 步长{step_rows}行")
        
        splits = []
        start_idx = 0
        
        while start_idx + train_rows + test_rows <= len(df):
            train_end = start_idx + train_rows
            test_end = train_end + test_rows
            
            split = {
                'split_id': len(splits),
                'train_start': start_idx,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
                'train_data': df.iloc[start_idx:train_end],
                'test_data': df.iloc[train_end:test_end]
            }
            
            splits.append(split)
            start_idx += step_rows
        
        print(f"创建了 {len(splits)} 个Walk-Forward分割")
        return splits
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备特征列"""
        # 排除非特征列
        exclude_cols = ['timestamp']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return df[feature_cols]
    
    def generate_labels(self, df: pd.DataFrame, horizon: int = 240, alpha: float = 1.0, 
                       mode: str = 'maker', require_fill: bool = True) -> pd.Series:
        """生成标签"""
        labels = make_labels(df['mid_price'], df['rel_spread'], horizon, alpha, mode=mode, require_fill=require_fill)
        return labels
    
    def train_and_evaluate(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                          label_name: str) -> dict:
        """训练和评估模型"""
        print(f"\n=== 训练模型: {label_name} ===")
        
        # 准备特征
        X_train = self.prepare_features(train_df)
        X_test = self.prepare_features(test_df)
        
        # 生成标签
        y_train = self.generate_labels(train_df)
        y_test = self.generate_labels(test_df)
        
        # 移除中性标签
        train_mask = y_train != 0
        test_mask = y_test != 0
        
        X_train_filtered = X_train[train_mask]
        y_train_filtered = y_train[train_mask]
        X_test_filtered = X_test[test_mask]
        y_test_filtered = y_test[test_mask]
        
        if len(X_train_filtered) == 0 or len(X_test_filtered) == 0:
            print("没有有效标签，跳过训练")
            return None
        
        # 转换为二分类
        y_train_binary = (y_train_filtered == 1).astype(int)
        y_test_binary = (y_test_filtered == 1).astype(int)
        
        print(f"训练样本: {len(X_train_filtered)} (long: {y_train_binary.sum()}, short: {len(y_train_binary)-y_train_binary.sum()})")
        print(f"测试样本: {len(X_test_filtered)} (long: {y_test_binary.sum()}, short: {len(y_test_binary)-y_test_binary.sum()})")
        
        # 训练模型
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='auc'
        )
        
        model.fit(X_train_filtered, y_train_binary)
        
        # 评估
        y_pred = model.predict(X_test_filtered)
        y_proba = model.predict_proba(X_test_filtered)[:, 1]
        
        train_accuracy = accuracy_score(y_train_binary, model.predict(X_train_filtered))
        test_accuracy = accuracy_score(y_test_binary, y_pred)
        train_auc = roc_auc_score(y_train_binary, model.predict_proba(X_train_filtered)[:, 1])
        test_auc = roc_auc_score(y_test_binary, y_proba)
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'feature_importance': feature_importance.head(10).to_dict('records'),
            'train_samples': len(X_train_filtered),
            'test_samples': len(X_test_filtered),
            'model': model
        }
        
        print(f"训练准确率: {train_accuracy:.3f}, 测试准确率: {test_accuracy:.3f}")
        print(f"训练AUC: {train_auc:.3f}, 测试AUC: {test_auc:.3f}")
        
        return results
    
    def run_walk_forward(self, label_params: list = None) -> dict:
        """运行完整的Walk-Forward训练"""
        print("🚀 开始Walk-Forward训练（集成队列特征）...")
        
        # 加载数据
        df, input_file = self.load_latest_queue_features()
        
        # 默认标签参数
        if label_params is None:
            label_params = [
                {'horizon': 240, 'alpha': 1.0, 'mode': 'maker', 'require_fill': True, 'name': 'h240_a1.0_maker_fill'},
                {'horizon': 240, 'alpha': 0.6, 'mode': 'maker', 'require_fill': True, 'name': 'h240_a0.6_maker_fill'},
                {'horizon': 60, 'alpha': 0.3, 'mode': 'maker', 'require_fill': True, 'name': 'h60_a0.3_maker_fill'}
            ]
        
        # 创建分割
        splits = self.create_walk_forward_splits(df)
        
        all_results = {}
        
        for label_param in label_params:
            print(f"\n{'='*60}")
            print(f"处理标签: {label_param['name']}")
            print(f"{'='*60}")
            
            split_results = []
            
            for split in splits:
                print(f"\n--- Split {split['split_id']} ---")
                
                # 训练和评估
                result = self.train_and_evaluate(
                    split['train_data'], 
                    split['test_data'], 
                    label_param['name']
                )
                
                if result is not None:
                    result['split_id'] = split['split_id']
                    result['label_params'] = label_param
                    split_results.append(result)
            
            # 汇总结果
            if split_results:
                all_results[label_param['name']] = {
                    'split_results': split_results,
                    'summary': self.summarize_results(split_results)
                }
        
        return all_results
    
    def summarize_results(self, split_results: list) -> dict:
        """汇总分割结果"""
        train_accuracies = [r['train_accuracy'] for r in split_results]
        test_accuracies = [r['test_accuracy'] for r in split_results]
        train_aucs = [r['train_auc'] for r in split_results]
        test_aucs = [r['test_auc'] for r in split_results]
        
        summary = {
            'num_splits': len(split_results),
            'avg_train_accuracy': np.mean(train_accuracies),
            'avg_test_accuracy': np.mean(test_accuracies),
            'avg_train_auc': np.mean(train_aucs),
            'avg_test_auc': np.mean(test_aucs),
            'std_train_accuracy': np.std(train_accuracies),
            'std_test_accuracy': np.std(test_accuracies),
            'std_train_auc': np.std(train_aucs),
            'std_test_auc': np.std(test_aucs),
            'overfitting_score': np.mean(train_aucs) - np.mean(test_aucs)
        }
        
        return summary
    
    def save_results(self, results: dict, output_file: str = None):
        """保存结果"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"walk_forward_results_queue_{timestamp}.json"
        
        # 转换模型对象为可序列化格式
        serializable_results = {}
        for label_name, label_results in results.items():
            serializable_results[label_name] = {
                'summary': label_results['summary'],
                'split_results': []
            }
            
            for split_result in label_results['split_results']:
                serializable_split = {k: v for k, v in split_result.items() if k != 'model'}
                serializable_results[label_name]['split_results'].append(serializable_split)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"结果已保存: {output_file}")
        return output_file

def main():
    # 创建Walk-Forward训练器 - 使用更小的窗口
    wf = WalkForwardQueue(train_hours=0.5, test_hours=0.1, step_hours=0.1)
    
    # 运行训练
    results = wf.run_walk_forward()
    
    # 保存结果
    wf.save_results(results)
    
    # 打印汇总
    print("\n" + "="*80)
    print("WALK-FORWARD 训练汇总")
    print("="*80)
    
    for label_name, label_results in results.items():
        summary = label_results['summary']
        print(f"\n📊 {label_name}:")
        print(f"  分割数: {summary['num_splits']}")
        print(f"  平均训练AUC: {summary['avg_train_auc']:.3f} ± {summary['std_train_auc']:.3f}")
        print(f"  平均测试AUC: {summary['avg_test_auc']:.3f} ± {summary['std_test_auc']:.3f}")
        print(f"  过拟合分数: {summary['overfitting_score']:.3f}")
        print(f"  平均训练准确率: {summary['avg_train_accuracy']:.3f}")
        print(f"  平均测试准确率: {summary['avg_test_accuracy']:.3f}")
    
    return results

if __name__ == "__main__":
    main() 