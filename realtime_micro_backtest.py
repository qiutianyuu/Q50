#!/usr/bin/env python3
"""
实时微观回测 – RexKing
默认参数:
  long_th      = 0.7
  short_th     = 0.3
  hold_rows    = 20          # 行数 = 20 条数据
  fee_rate     = 0.0005      # 0.05%
"""

import glob
import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb

# ---------- 参数 ----------
LONG_TH   = 0.65  # 多头阈值
SHORT_TH  = 0.35  # 空头阈值
HOLD_ROWS = 10    # 持仓步数
FEE_RATE  = 0.001 # 手续费0.1%

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("micro_backtest")

# ---------- 工具 ----------
def latest_file(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No file match: {pattern}")
    return max(files, key=os.path.getctime)

# ---------- 数据与模型 ----------
feature_path = latest_file("data/analysis/micro_features_*.parquet")
model_path   = latest_file("xgb_realtime_*.bin")
feat_list_js = model_path.replace(".bin", ".json").replace("xgb_", "feature_list_")

logger.info(f"🔹Features : {feature_path}")
logger.info(f"🔹Model    : {model_path}")

df  = pd.read_parquet(feature_path).dropna().reset_index(drop=True)

with open(feat_list_js) as f:
    feats = json.load(f)

# 保证特征列顺序一致，缺失列填 0
for col in feats:
    if col not in df.columns: df[col] = 0
X = df[feats].astype(float).fillna(0)

clf = xgb.XGBClassifier()
clf.load_model(model_path)
proba = clf.predict_proba(X)[:, 1]

# ---------- 生成信号 ----------
df["prob"]   = proba
df["signal"] = 0
df.loc[df["prob"] >= LONG_TH , "signal"] =  1   # 做多
df.loc[df["prob"] <= SHORT_TH, "signal"] = -1   # 做空

logger.info(f"信号统计  多:{(df.signal==1).sum()} 空:{(df.signal==-1).sum()}  无:{(df.signal==0).sum()}")

# ---------- 交易回测 ----------
position     = 0
entry_price  = 0.0
entry_index  = -1
pnl_list     = []

for i, row in df.iterrows():
    price  = row["mid_price"]
    sig    = row["signal"]

    # 平仓条件：持仓达到 HOLD_ROWS
    if position != 0 and (i - entry_index) >= HOLD_ROWS:
        ret = (price - entry_price)/entry_price if position==1 else (entry_price - price)/entry_price
        ret -= FEE_RATE*2               # 进+出 手续费
        pnl_list.append(ret)
        position   = 0
        entry_price= 0
        entry_index= -1

    # 开仓条件：当前无仓且出现信号
    if position == 0 and sig != 0:
        position    = sig
        entry_price = price
        entry_index = i

# 避免末尾持仓未平：按最新价强平
if position != 0:
    price = df.iloc[-1]["mid_price"]
    ret   = (price - entry_price)/entry_price if position==1 else (entry_price - price)/entry_price
    ret  -= FEE_RATE*2
    pnl_list.append(ret)

# ---------- 结果 ----------
pnl_series = pd.Series(pnl_list)
total_ret  = pnl_series.sum()
win_rate   = (pnl_series > 0).mean()
avg_ret    = pnl_series.mean() if len(pnl_series) else 0

logger.info("============= 回测结果 =============")
logger.info(f"交易笔数  : {len(pnl_series)}")
logger.info(f"胜率      : {win_rate:.2%}")
logger.info(f"平均收益  : {avg_ret:.4%}")
logger.info(f"累计收益  : {total_ret:.4%}")
logger.info("====================================")

# 保存结果
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_df = pd.DataFrame({
    'trade_id': range(len(pnl_series)),
    'returns': pnl_series,
    'cumulative_returns': pnl_series.cumsum()
})
results_df.to_csv(f"micro_backtest_results_{timestamp}.csv", index=False)
logger.info(f"结果已保存: micro_backtest_results_{timestamp}.csv")

# 保存信号数据
signals_df = df[['timestamp', 'mid_price', 'prob', 'signal']].copy()
signals_df.to_csv(f"micro_signals_{timestamp}.csv", index=False)
logger.info(f"信号已保存: micro_signals_{timestamp}.csv") 
 