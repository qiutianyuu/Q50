#!/usr/bin/env python3
"""
带填单验证的微观特征XGBoost训练脚本
使用最佳填单验证标签训练模型
"""

import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import json
import glob
import os
import joblib

def load_latest_labeled_data():
    """Load the latest labeled features file"""
    files = glob.glob("data/realtime_features_with_fill_labels_*.parquet")
    if not files:
        raise FileNotFoundError("No labeled features files found")
    
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading: {latest_file}")
    df = pd.read_parquet(latest_file)
    print(f"Loaded {len(df)} rows")
    return df

def prepare_features(df):
    """Prepare features for training"""
    # Exclude non-feature columns
    exclude_cols = ['timestamp', 'label', 'bid_price', 'ask_price', 'mid_price', 'rel_spread']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    # Map labels from [-1, 0, 1] to [0, 1, 2] for XGBoost
    y = y.map({-1: 0, 0: 1, 1: 2})
    
    return X, y, feature_cols

def train_model(X, y, feature_cols):
    """Train XGBoost model with regularization"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # XGBoost parameters with regularization
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 4,  # Reduced depth to prevent overfitting
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'random_state': 42,
        'eval_metric': 'mlogloss'
    }
    
    # Train model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Metrics
    print("\n=== Model Performance ===")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    
    # AUC for each class
    for i, class_name in enumerate(['Short', 'Neutral', 'Long']):
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score((y_test == i).astype(int), y_proba[:, i])
            print(f"{class_name} AUC: {auc:.3f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Short', 'Neutral', 'Long']))
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    print(feature_importance.head(10))
    
    return model, X_test, y_test, feature_importance

def main():
    # Load data
    df = load_latest_labeled_data()
    
    # Prepare features
    X, y, feature_cols = prepare_features(df)
    
    # Train model
    model, X_test, y_test, feature_importance = train_model(X, y, feature_cols)
    
    # Save model and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_file = f"xgb_micro_label_h240_a0.3_maker_fill_{timestamp}.bin"
    joblib.dump(model, model_file)
    print(f"\nModel saved: {model_file}")
    
    # Save feature importance
    importance_file = f"feature_importance_micro_fill_{timestamp}.csv"
    feature_importance.to_csv(importance_file, index=False)
    print(f"Feature importance saved: {importance_file}")
    
    # Save test predictions for analysis
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    test_results = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred,
        'prob_short': y_proba[:, 0],
        'prob_neutral': y_proba[:, 1],
        'prob_long': y_proba[:, 2]
    })
    
    results_file = f"micro_model_results_fill_{timestamp}.json"
    test_results.to_json(results_file, orient='records')
    print(f"Test results saved: {results_file}")

if __name__ == "__main__":
    main() 