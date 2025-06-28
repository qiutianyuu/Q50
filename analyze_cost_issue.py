import pandas as pd
import numpy as np

# 分析成本问题
df = pd.read_csv('cost_aware_trades.csv')

print("=== 成本问题分析 ===")
print(f"总交易数: {len(df)}")
print(f"平均成本: ${df['costs'].mean():.2f}")
print(f"平均PnL: ${df['pnl'].mean():.2f}")
print(f"成本/PnL比率: {100*df['costs'].mean()/abs(df['pnl'].mean()):.1f}%")

print("\n=== 样本交易分析 ===")
sample = df.head(3)
for i, row in sample.iterrows():
    print(f"交易 {i+1}:")
    print(f"  入场价: ${row['entry_price']:.2f}")
    print(f"  出场价: ${row['exit_price']:.2f}")
    print(f"  方向: {'多头' if row['position']==1 else '空头'}")
    print(f"  成本: ${row['costs']:.2f}")
    print(f"  PnL: ${row['pnl']:.2f}")
    print(f"  毛收益: {100*row['gross_return']:.2f}%")
    print(f"  净收益: {100*row['net_return']:.2f}%")
    print()

# 计算理论成本
position_size = 1000  # USD
maker_fee = 0.0001
taker_fee = 0.0005
slippage_basis = 0.0002
funding_rate = 0.0001

print("=== 理论成本计算 ===")
print(f"固定仓位: ${position_size}")
print(f"Maker费率: {100*maker_fee:.2f}%")
print(f"Taker费率: {100*taker_fee:.2f}%")
print(f"基础滑点: {100*slippage_basis:.2f}%")
print(f"资金费率: {100*funding_rate:.2f}%")

# 假设平均价格1200，持仓1小时
avg_price = 1200
holding_hours = 1

theoretical_cost = position_size * (maker_fee + maker_fee + slippage_basis + slippage_basis + funding_rate * holding_hours/8)
print(f"理论成本 (1小时持仓): ${theoretical_cost:.2f}")

# 检查实际成本是否合理
print(f"\n实际平均成本: ${df['costs'].mean():.2f}")
print(f"成本放大倍数: {df['costs'].mean()/theoretical_cost:.1f}x") 