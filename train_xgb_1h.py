import pandas as pd
from src.xgboost_module import XGBoostManager

# 路径
FEATURE_FILE = "/Users/qiutianyu/data/processed/features_1h_2023_2025.parquet"

# 读取特征
print(f"读取特征文件: {FEATURE_FILE}")
df = pd.read_parquet(FEATURE_FILE)

# 选择特征列（排除时间、标签、非数值型等）
feature_cols = [c for c in df.columns if c not in ["timestamp", "label"] and pd.api.types.is_numeric_dtype(df[c])]
X = df[feature_cols]
y = df["label"]

# 训练模型
xgbm = XGBoostManager(model_path="xgb_1h_model.bin")
xgbm.train(X, y)

# SHAP 分析
shap_imp = xgbm.shap_analysis(X, feature_cols)
print("\nTop 10 SHAP特征:")
print(sorted(shap_imp.items(), key=lambda x: -x[1])[:10])

print(f"\n验证集AUC: {xgbm.last_auc:.4f}") 