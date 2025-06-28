import pandas as pd

df = pd.read_csv('/Users/qiutianyu/ETHUSDT-4h/merged_4h_2023_2025.csv')

# 计算4H条件
df['obv_ma14'] = df['obv'].rolling(14).mean()
cond_4h = (df['obv'] > df['obv_ma14']) & (df['ema20'] > df['ema60']) & (df['atr'] > 0.003 * df['close']) & (df['bb'] > 2)
cond_1h = (df['funding'] < 0.00005)
cond_15m = df['breakout_15m'] & df['volume_surge_15m']
cond_w1 = (df['w1_value'] > 3000) & (df['w1_zscore'] > 1.5) & (df['w1_signal_rolling'] > 0)

# 找到15m信号
m15_signals = df[cond_15m]
print(f"15m信号数量: {len(m15_signals)}")

print("前5个15m信号时间点:")
for idx, row in m15_signals.head(5).iterrows():
    print(f"{row['timestamp']}")

# 找到W1信号
w1_signals = df[cond_w1]
print(f"\nW1信号数量: {len(w1_signals)}")

print("W1信号时间点:")
for idx, row in w1_signals.iterrows():
    print(f"{row['timestamp']}")

# 检查联合信号
joint_signals = df[cond_4h & cond_1h & cond_15m & cond_w1]
print(f"\n联合信号数量: {len(joint_signals)}")

if len(joint_signals) > 0:
    print("联合信号时间点:")
    for idx, row in joint_signals.iterrows():
        print(f"{row['timestamp']}") 