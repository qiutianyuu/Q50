#!/usr/bin/env python3
"""
测试订单流特征效果的训练脚本
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib
from datetime import datetime
import glob
import os

def load_test_data():
    """加载测试数据"""
    files = glob.glob("data/test_features_5m_orderflow_*.parquet")
    if not files:
        raise FileNotFoundError("No test features files found")
    
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading: {latest_file}")
    df = pd.read_parquet(latest_file)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def prepare_features(df, use_order_flow=True):
    """准备特征"""
    # 排除非特征列
    exclude_cols = ['timestamp', 'label', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if not use_order_flow:
        # 只使用技术指标特征
        order_flow_cols = [
            'price_momentum_1m', 'price_momentum_3m', 'price_momentum_5m',
            'volume_ratio', 'volume_ma_3m', 'volume_ma_5m', 'spread_ratio', 
            'spread_trend', 'spread_ma', 'spread_std', 'imbalance_trend', 
            'pressure_trend', 'volume_imbalance_ma', 'bid_ask_imbalance', 
            'price_pressure_ma', 'price_pressure_std', 'liquidity_trend', 
            'fill_prob_trend', 'liquidity_score', 'liquidity_ma', 'bid_fill_prob', 
            'ask_fill_prob', 'bid_price_impact', 'ask_price_impact', 'volatility_ma', 
            'volatility_ratio', 'price_volatility', 'price_jump', 'volume_spike', 
            'spread_widening', 'vwap_deviation', 'vwap_deviation_ma', 'buy_ratio', 
            'price_trend', 'trend_deviation', 'price_impact_imbalance', 'fill_prob_imbalance'
        ]
        feature_cols = [col for col in feature_cols if col not in order_flow_cols]
        print(f"Using technical features only: {len(feature_cols)} features")
    else:
        print(f"Using all features: {len(feature_cols)} features")
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    return X, y, feature_cols

def train_and_evaluate(X, y, feature_cols, model_name):
    """训练和评估模型"""
    print(f"\n=== Training {model_name} ===")
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # XGBoost参数
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42
    }
    
    # 训练模型
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"Test AUC: {auc:.3f}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Features:")
    print(feature_importance.head(10))
    
    return {
        'model': model,
        'accuracy': accuracy,
        'auc': auc,
        'feature_importance': feature_importance,
        'test_predictions': y_proba
    }

def main():
    print("=== 订单流特征效果测试 ===")
    
    # 加载数据
    df = load_test_data()
    
    # 测试1：只用技术指标
    print("\n" + "="*50)
    X_tech, y_tech, tech_cols = prepare_features(df, use_order_flow=False)
    tech_results = train_and_evaluate(X_tech, y_tech, tech_cols, "Technical Only")
    
    # 测试2：技术指标 + 订单流特征
    print("\n" + "="*50)
    X_all, y_all, all_cols = prepare_features(df, use_order_flow=True)
    all_results = train_and_evaluate(X_all, y_all, all_cols, "Technical + Order Flow")
    
    # 对比结果
    print("\n" + "="*50)
    print("=== 结果对比 ===")
    print(f"Technical Only - Accuracy: {tech_results['accuracy']:.3f}, AUC: {tech_results['auc']:.3f}")
    print(f"Technical + Order Flow - Accuracy: {all_results['accuracy']:.3f}, AUC: {all_results['auc']:.3f}")
    
    accuracy_improvement = all_results['accuracy'] - tech_results['accuracy']
    auc_improvement = all_results['auc'] - tech_results['auc']
    
    print(f"\nImprovement:")
    print(f"Accuracy: {accuracy_improvement:+.3f}")
    print(f"AUC: {auc_improvement:+.3f}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存技术指标模型
    tech_model_file = f"xgb_test_technical_{timestamp}.bin"
    joblib.dump(tech_results['model'], tech_model_file)
    print(f"\nTechnical model saved: {tech_model_file}")
    
    # 保存完整模型
    all_model_file = f"xgb_test_orderflow_{timestamp}.bin"
    joblib.dump(all_results['model'], all_model_file)
    print(f"Order flow model saved: {all_model_file}")
    
    # 保存对比结果
    comparison_results = {
        'technical_accuracy': tech_results['accuracy'],
        'technical_auc': tech_results['auc'],
        'orderflow_accuracy': all_results['accuracy'],
        'orderflow_auc': all_results['auc'],
        'accuracy_improvement': accuracy_improvement,
        'auc_improvement': auc_improvement,
        'technical_features': len(tech_cols),
        'orderflow_features': len(all_cols)
    }
    
    import json
    comparison_file = f"orderflow_comparison_{timestamp}.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Comparison results saved: {comparison_file}")
    
    # 结论
    print(f"\n=== 结论 ===")
    if auc_improvement > 0.05:
        print("✅ 订单流特征显著提升了模型性能")
    elif auc_improvement > 0.02:
        print("⚠️ 订单流特征有一定提升，但效果有限")
    else:
        print("❌ 订单流特征在当前数据上效果不明显")
    
    print(f"建议: {'继续收集更多订单流数据' if auc_improvement > 0 else '考虑其他特征工程方向'}")

if __name__ == "__main__":
    main() 