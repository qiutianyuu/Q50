import xgboost as xgb
import numpy as np
import pandas as pd
import shap
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
import joblib

logger = logging.getLogger("SteadyBullXGB")

# ========== XGBoost模型管理 ========== #
class XGBoostManager:
    def __init__(self, model_path="xgb_model.bin"):
        self.model_path = model_path
        self.model = None
        self.shap_values = None
        self.shap_importance = None
        self.last_auc = None

    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"XGBoost model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            self.model = None

    def save_model(self):
        if self.model:
            joblib.dump(self.model, self.model_path)
            logger.info(f"XGBoost model saved to {self.model_path}")

    def train(self, X, y, params=None):
        if params is None:
            params = {
                'max_depth': 4,
                'n_estimators': 250,
                'learning_rate': 0.05,
                'reg_lambda': 1.5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
            }
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict_proba(X_val)[:,1]
        self.last_auc = roc_auc_score(y_val, y_pred)
        logger.info(f"XGBoost trained. Validation AUC: {self.last_auc:.4f}")
        self.save_model()

    def predict(self, X):
        if self.model is None:
            self.load_model()
        if self.model is not None:
            return self.model.predict_proba(X)[:,1]
        else:
            logger.error("No XGBoost model loaded.")
            return np.zeros(X.shape[0])

    def shap_analysis(self, X, feature_names):
        if self.model is None:
            self.load_model()
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(X)
        shap_sum = np.abs(self.shap_values).mean(axis=0)
        self.shap_importance = dict(zip(feature_names, shap_sum / shap_sum.sum()))
        logger.info(f"SHAP importance: {self.shap_importance}")
        return self.shap_importance

    def grid_search(self, X, y):
        param_grid = {
            'max_depth': [3, 4, 5],
            'reg_lambda': [1.0, 1.5, 2.0],
            'learning_rate': [0.03, 0.05, 0.07]
        }
        xgb_clf = xgb.XGBClassifier(n_estimators=250, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, objective='binary:logistic', eval_metric='auc')
        grid = GridSearchCV(xgb_clf, param_grid, scoring='roc_auc', cv=3, verbose=2)
        grid.fit(X, y)
        logger.info(f"GridSearch best params: {grid.best_params_}")
        self.model = grid.best_estimator_
        self.save_model()
        return grid.best_params_

    def get_shap_weights(self):
        """返回当前SHAP权重字典，供指标加权用"""
        return self.shap_importance or {}

logger.info("[XGBoost Module] XGBoostManager loaded (first 100 lines)")

# ========== API预测占位 ========== #
def api_predict_xgb(features):
    """占位：如需远程API预测，可在此实现"""
    logger.warning("API XGBoost预测未实现，返回0.5概率")
    return 0.5

# ========== 特征工程工具 ========== #
def build_feature_vector(indicator_scores, shap_weights):
    """将16指标分数和SHAP权重加权，生成最终特征向量"""
    weighted = [indicator_scores[k] * shap_weights.get(k, 0.05) for k in indicator_scores]
    return np.array(weighted).reshape(1, -1)

logger.info("[XGBoost Module] Feature engineering tools loaded (line 120)") 