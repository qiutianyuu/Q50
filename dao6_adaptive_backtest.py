"""
dao6_adaptive_backtest.py
---------------------------------
• 数据: 币安 K 线 CSV, 第一列为微秒时间戳（16 位）。
• 频率: 15-minute
• 目标: 验证 2025-04 / 其他月份
"""

import pandas as pd, numpy as np, talib as ta

# === 0. 全局配置 ===
CSV_PATH     = '/Users/qiutianyu/Downloads/ETHUSDT-15m-2025-04.csv'
FEE_RATE     = 0.002          # 单边手续费
INIT_CAP     = 1_000.0
ATR_LEN      = 14
MOM_LEN      = 10             # 动量窗口
EMA_FAST, EMA_SLOW = 20, 50   # 趋势确认
COOLDOWN_MAX = 6              # h
COOLDOWN_MIN = 2              # h

# === 1. 读取 & 指标 ===
col = ['ts','open','high','low','close','vol','end','qvol','trades',
       'tbvol','tbqvol','ignore']
df  = pd.read_csv(CSV_PATH, names=col)
df['ts'] = pd.to_datetime(df['ts'], unit='us')
df.set_index('ts', inplace=True)

# 技术指标
df['atr']  = ta.ATR(df['high'], df['low'], df['close'], timeperiod=ATR_LEN)
df['mom']  = df['close'].pct_change(MOM_LEN)          # ΔP/P
df['ema_f']= ta.EMA(df['close'], timeperiod=EMA_FAST)
df['ema_s']= ta.EMA(df['close'], timeperiod=EMA_SLOW)
df.dropna(inplace=True)

# === 2. 回测变量 ===
capital      = INIT_CAP
equity_curve = [capital]
vitality     = 50.0
position_qty = 0.0
entry_price  = stop = tp = 0.0
cooldown_end = df.index[0]

trades = []

# === 3. 主循环 ===
for i, row in df.iterrows():
    price, atr = row.close, row.atr
    sigma = atr / price                              # 实时噪声刻度 σₜ

    # ---------- 更新活力 ----------
    if trades:
        last_pnl = trades[-1]['pnl'] / (sigma*INIT_CAP)   # 无量纲化盈亏
        vitality = 0.9*vitality + 0.1*last_pnl*100
        vitality = np.clip(vitality, 10, 100)

    # ---------- 仓位预算 ----------
    if sigma == 0: continue
    edge  = 1.0        # 预期 TP/SL 比 = 2:1 → Edge≈1
    kelly = (edge) / (sigma * 100)          # 极简化 Kelly
    pos_frac = np.clip(kelly*0.5, 0.02, 0.08)
    pos_frac *= (vitality/100)              # 活力加权

    # ---------- 冷却期 ----------
    if i < cooldown_end: 
        equity_curve.append(capital)
        continue

    # ---------- 有仓位: 监控止盈/止损 ----------
    if position_qty:
        if price <= stop or price >= tp:
            exit_price = price
            pnl_cap    = (exit_price - entry_price) * position_qty
            fees       = FEE_RATE * abs(position_qty) * (entry_price + exit_price)
            pnl_net    = pnl_cap - fees
            capital   += pnl_net

            trades.append(dict(t=i, pnl=pnl_net))
            position_qty = 0.0

            # 冷却：跌破 SL 属于逆势，时间按 σₜ 放大
            if exit_price == stop:
                cd_h = np.clip(sigma/0.01 * 2, COOLDOWN_MIN, COOLDOWN_MAX)
                cooldown_end = i + pd.Timedelta(hours=cd_h)

    # ---------- 空仓: 寻找入场 ----------
    else:
        # 信号：动量跌破阈 ＆ EMA20>EMA50（多头）
        mom_th  = -1.2 * sigma               # 动态阈
        long_ok = (row.mom < mom_th) and (row.ema_f > row.ema_s)

        if long_ok:
            position_qty = capital * pos_frac / price
            entry_price  = price
            stop = price - 0.8*atr
            tp   = price + 2.0*atr

    equity_curve.append(capital if not position_qty else
                        capital + (price-entry_price)*position_qty)

# === 4. 统计 ===
equity_curve = np.arangeray(equity_curve)
ret_month    = equity_curve[-1]/INIT_CAP - 1
dd = 1 - equity_curve / np.maximum.accumulate(equity_curve)
max_dd       = dd.max()
wins         = [t for t in trades if t['pnl']>0]
win_rate     = len(wins)/len(trades) if trades else 0
avg_pnl      = np.mean([t['pnl'] for t in trades]) if trades else 0

print(f"RESULT -->  MonthRet {ret_month:.2%} | MaxDD {max_dd:.2%} | "
      f"Trades {len(trades)} | WinRate {win_rate:.1%} | AvgPnL ${avg_pnl:,.2f}") 