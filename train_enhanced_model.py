#!/usr/bin/env python3
"""
RexKing – Enhanced Model Training with Order Flow Features

整合订单流特征重新训练模型，提升预测能力
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ---------- 配置 ----------
DATA_DIR = Path("/Users/qiutianyu/data/processed")
FEATURES_FILE = DATA_DIR / "features_15m_enhanced.parquet"
ORDERFLOW_FILE = Path("data/mid_features_15m_orderflow.parquet")
MODEL_FILE = "xgb_15m_enhanced.bin"

def load_and_merge_data():
    """加载并合并特征数据"""
    print("📥 加载基础特征数据...")
    df = pd.read_parquet(FEATURES_FILE)
    print(f"基础特征数据: {len(df)} 行, {len(df.columns)} 列")
    
    if ORDERFLOW_FILE.exists():
        print("📊 加载订单流特征...")
        orderflow = pd.read_parquet(ORDERFLOW_FILE)
        orderflow['timestamp'] = pd.to_datetime(orderflow['timestamp'], utc=True)
        
        # 确保主数据也有UTC时区
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # 选择高信息密度的订单流特征
        orderflow_cols = [
            'liquidity_pressure', 'liquidity_pressure_ma',
            'taker_imbalance', 'taker_imbalance_ma',
            'order_flow_intensity', 'order_flow_intensity_ma',
            'liquidity_impact', 'liquidity_impact_ma',
            'buy_pressure_ratio', 'sell_pressure_ratio',
            'order_flow_strength', 'order_flow_strength_ma',
            'liquidity_stress', 'liquidity_stress_ma',
            'spread_compression', 'volume_imbalance',
            'price_pressure', 'vwap_deviation'
        ]
        
        # 只保留存在的列
        available_cols = [col for col in orderflow_cols if col in orderflow.columns]
        print(f"可用订单流特征: {len(available_cols)}")
        
        # 合并订单流特征
        df = pd.merge_asof(
            df.sort_values('timestamp'),
            orderflow[['timestamp'] + available_cols].sort_values('timestamp'),
            on='timestamp', direction='backward'
        )
        
        # 填充NaN
        df[available_cols] = df[available_cols].fillna(0)
        
        print(f"✅ 订单流特征整合完成，新增 {len(available_cols)} 个特征")
        print(f"合并后数据: {len(df)} 行, {len(df.columns)} 列")
    else:
        print("⚠️ 订单流文件不存在，跳过订单流特征")
    
    return df

def prepare_features(df):
    """准备训练特征"""
    # 排除非特征列
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'trend_1h', 'trend_4h']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"样本数量: {len(X)}")
    print(f"标签分布: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols

def train_enhanced_model(X, y, feature_cols):
    """训练增强版模型"""
    print("🚀 训练增强版模型...")
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 优化参数
    params = {
        'max_depth': 6,
        'n_estimators': 300,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
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
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"测试集AUC: {auc:.4f}")
    
    # 高置信度预测
    high_conf_mask = (y_pred_proba > 0.8) | (y_pred_proba < 0.2)
    if high_conf_mask.sum() > 0:
        high_conf_auc = roc_auc_score(y_test[high_conf_mask], y_pred_proba[high_conf_mask])
        high_conf_acc = (y_pred[high_conf_mask] == y_test[high_conf_mask]).mean()
        print(f"高置信度样本AUC: {high_conf_auc:.4f}")
        print(f"高置信度样本准确率: {high_conf_acc:.4f}")
        print(f"高置信度样本数量: {high_conf_mask.sum()}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 Top 15 特征重要性:")
    for i, row in feature_importance.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 保存模型
    model.save_model(MODEL_FILE)
    print(f"✅ 增强版模型已保存: {MODEL_FILE}")
    
    # 保存特征列表
    feature_info = {
        'feature_cols': feature_cols,
        'auc': auc,
        'high_conf_auc': high_conf_auc if high_conf_mask.sum() > 0 else 0,
        'high_conf_acc': high_conf_acc if high_conf_mask.sum() > 0 else 0
    }
    
    import json
    with open('enhanced_model_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2, default=str)
    
    return model, auc, feature_importance

def main():
    print("=== RexKing Enhanced Model Training ===")
    
    # 加载并合并数据
    df = load_and_merge_data()
    
    # 准备特征
    X, y, feature_cols = prepare_features(df)
    
    # 训练模型
    model, auc, feature_importance = train_enhanced_model(X, y, feature_cols)
    
    print(f"\n🎉 增强版模型训练完成!")
    print(f"最终AUC: {auc:.4f}")
    print(f"特征数量: {len(feature_cols)}")
    
    return model, auc

if __name__ == "__main__":
    main() 