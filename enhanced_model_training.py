#!/usr/bin/env python3
"""
RexKing – Enhanced Model Training with Order Flow Features & Calibration

整合订单流特征重新训练模型，并进行概率校准
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ---------- 配置 ----------
DATA_DIR = Path("/Users/qiutianyu/data/processed")
FEATURES_FILE = DATA_DIR / "features_15m_enhanced.parquet"  # 使用合并后的特征表
ORDERFLOW_FILE = Path("data/mid_features_15m_orderflow.parquet")
MODEL_FILE = "xgb_15m_enhanced_calibrated.bin"

def create_orderflow_features(df):
    """创建订单流特征（基于现有特征）"""
    print("🔧 创建订单流特征...")
    
    # 基于现有特征创建订单流相关指标
    # 使用volume和price相关特征
    df['volume_imbalance'] = (df['volume'] - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).std() + 1e-8)
    df['price_pressure'] = df['price_change'] * df['volume_ratio']
    
    df['liquidity_pressure'] = df['volume_imbalance'] * df['price_pressure']
    df['liquidity_pressure_ma'] = df['liquidity_pressure'].rolling(20).mean()
    
    # 使用buy_ratio如果存在，否则用其他指标
    if 'buy_ratio' in df.columns:
        df['taker_imbalance'] = (df['buy_ratio'] - 0.5) * 2
    else:
        df['taker_imbalance'] = df['price_change'] * df['volume_ratio']
    df['taker_imbalance_ma'] = df['taker_imbalance'].rolling(20).mean()
    
    df['order_flow_intensity'] = df['price_pressure'] * df['volume_imbalance']
    df['order_flow_intensity_ma'] = df['order_flow_intensity'].rolling(20).mean()
    
    df['liquidity_impact'] = df['price_change'] / (df['volume'] + 1e-8)
    df['liquidity_impact_ma'] = df['liquidity_impact'].rolling(20).mean()
    
    # 使用volume相关指标
    df['volume_pressure_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
    df['volume_pressure_ratio_ma'] = df['volume_pressure_ratio'].rolling(20).mean()
    
    df['order_flow_strength'] = df['order_flow_intensity'] * df['volume_ratio']
    df['order_flow_strength_ma'] = df['order_flow_strength'].rolling(20).mean()
    
    # 使用spread相关指标
    if 'spread_bps' in df.columns:
        df['liquidity_stress'] = df['liquidity_pressure'] * (df['spread_bps'] / df['spread_bps'].rolling(20).mean())
    else:
        df['liquidity_stress'] = df['liquidity_pressure'] * df['volatility_24']
    df['liquidity_stress_ma'] = df['liquidity_stress'].rolling(20).mean()
    
    if 'spread_bps' in df.columns:
        df['spread_compression'] = df['spread_bps'] / df['spread_bps'].rolling(20).mean()
    else:
        df['spread_compression'] = df['volatility_24'] / df['volatility_24'].rolling(20).mean()
    
    # 新增：价格动量与订单流结合
    df['price_momentum_flow'] = df['price_momentum'] * df['volume_imbalance']
    df['price_momentum_flow_ma'] = df['price_momentum_flow'].rolling(20).mean()
    
    # 新增：波动率与订单流结合
    df['volatility_flow'] = df['volatility_24'] * df['order_flow_intensity']
    df['volatility_flow_ma'] = df['volatility_flow'].rolling(20).mean()
    
    print(f"✅ 创建了 {len([col for col in df.columns if 'flow' in col or 'pressure' in col or 'stress' in col])} 个订单流特征")
    return df

def prepare_enhanced_features(df):
    """准备增强特征"""
    print("📊 准备增强特征...")
    
    # 排除非特征列
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'trend_1h', 'trend_4h']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"样本数量: {len(X)}")
    print(f"标签分布: {y.value_counts().to_dict()}")
    
    # 显示订单流特征统计
    orderflow_cols = [col for col in feature_cols if any(keyword in col for keyword in ['flow', 'pressure', 'stress', 'imbalance', 'spread', 'liquidity'])]
    print(f"订单流特征数量: {len(orderflow_cols)}")
    
    # 检查订单流特征的非零比例
    if orderflow_cols:
        non_zero_ratio = (X[orderflow_cols] != 0).sum().sum() / (len(X) * len(orderflow_cols))
        print(f"订单流特征非零比例: {non_zero_ratio:.2%}")
    
    return X, y, feature_cols

def train_calibrated_model(X, y, feature_cols):
    """训练校准后的模型"""
    print("🚀 训练校准模型...")
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 基础XGBoost参数
    base_params = {
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
    
    # 训练基础模型
    base_model = xgb.XGBClassifier(**base_params)
    base_model.fit(X_train, y_train, verbose=0)
    
    # 概率校准
    print("🔧 进行概率校准...")
    calibrated_model = CalibratedClassifierCV(
        base_model, 
        cv=5, 
        method='isotonic',  # 使用isotonic回归校准
        n_jobs=-1
    )
    calibrated_model.fit(X_train, y_train)
    
    # 评估
    y_pred_proba_base = base_model.predict_proba(X_test)[:, 1]
    y_pred_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred_base = base_model.predict(X_test)
    y_pred_cal = calibrated_model.predict(X_test)
    
    # 基础模型评估
    auc_base = roc_auc_score(y_test, y_pred_proba_base)
    brier_base = brier_score_loss(y_test, y_pred_proba_base)
    
    # 校准模型评估
    auc_cal = roc_auc_score(y_test, y_pred_proba_cal)
    brier_cal = brier_score_loss(y_test, y_pred_proba_cal)
    
    print(f"基础模型 AUC: {auc_base:.4f}, Brier Score: {brier_base:.4f}")
    print(f"校准模型 AUC: {auc_cal:.4f}, Brier Score: {brier_cal:.4f}")
    
    # 高置信度预测对比
    high_conf_mask = (y_pred_proba_cal > 0.8) | (y_pred_proba_cal < 0.2)
    if high_conf_mask.sum() > 0:
        high_conf_auc = roc_auc_score(y_test[high_conf_mask], y_pred_proba_cal[high_conf_mask])
        high_conf_acc = (y_pred_cal[high_conf_mask] == y_test[high_conf_mask]).mean()
        print(f"高置信度样本AUC: {high_conf_auc:.4f}")
        print(f"高置信度样本准确率: {high_conf_acc:.4f}")
        print(f"高置信度样本数量: {high_conf_mask.sum()}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': base_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 Top 15 特征重要性:")
    for i, row in feature_importance.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 保存模型
    calibrated_model.save_model(MODEL_FILE)
    print(f"✅ 校准模型已保存: {MODEL_FILE}")
    
    # 保存特征信息
    feature_info = {
        'feature_cols': feature_cols,
        'auc_base': auc_base,
        'auc_calibrated': auc_cal,
        'brier_base': brier_base,
        'brier_calibrated': brier_cal,
        'high_conf_auc': high_conf_auc if high_conf_mask.sum() > 0 else 0,
        'high_conf_acc': high_conf_acc if high_conf_mask.sum() > 0 else 0
    }
    
    import json
    with open('enhanced_model_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2, default=str)
    
    return calibrated_model, base_model, auc_cal, feature_importance

def analyze_calibration_quality(calibrated_model, X_test, y_test):
    """分析校准质量"""
    print("\n📈 校准质量分析...")
    
    y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    # 分箱分析
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bin_edges) - 1
    
    calibration_data = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_pred = y_pred_proba[mask].mean()
            mean_actual = y_test[mask].mean()
            count = mask.sum()
            calibration_data.append({
                'bin': i,
                'mean_pred': mean_pred,
                'mean_actual': mean_actual,
                'count': count
            })
    
    cal_df = pd.DataFrame(calibration_data)
    print("校准分箱分析:")
    print(cal_df.to_string(index=False, float_format='%.3f'))
    
    # 计算校准误差
    calibration_error = np.mean((cal_df['mean_pred'] - cal_df['mean_actual'])**2)
    print(f"校准误差: {calibration_error:.4f}")
    
    return cal_df

def main():
    print("=== RexKing Enhanced Model Training with Calibration ===")
    
    # 加载数据
    print("📥 加载增强特征数据...")
    df = pd.read_parquet(FEATURES_FILE)
    print(f"数据行数: {len(df)}")
    
    # 准备增强特征
    X, y, feature_cols = prepare_enhanced_features(df)
    
    # 训练校准模型
    calibrated_model, base_model, auc, feature_importance = train_calibrated_model(X, y, feature_cols)
    
    # 分析校准质量
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cal_df = analyze_calibration_quality(calibrated_model, X_test, y_test)
    
    print(f"\n🎉 增强版模型训练完成!")
    print(f"最终校准AUC: {auc:.4f}")
    print(f"特征数量: {len(feature_cols)}")
    
    return calibrated_model, auc

if __name__ == "__main__":
    main() 