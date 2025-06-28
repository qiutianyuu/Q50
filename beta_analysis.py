import pandas as pd
import numpy as np
import talib as ta
import json
from datetime import datetime, timedelta
import argparse
from collections import defaultdict

def prepare_data(file_path):
    """准备数据"""
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)
    
    # 计算指标
    df['atr'] = ta.ATR(df.high, df.low, df.close, 14)
    df['sigma'] = df.atr / df.close
    df['sigma_50'] = df.sigma.rolling(50).median()  # 预计算sigma_50
    df['mom3'] = df.close.pct_change(3) / df.sigma
    df['rsi'] = ta.RSI(df.close, 14)
    df['adx'] = ta.ADX(df.high, df.low, df.close, 14)
    df.dropna(inplace=True)
    
    return df

def calc_position_size(sigma, sigma_50, sigma_ref):
    """动态仓位计算 - 优化版"""
    base_size = 0.1  # 10.0%基础仓位
    
    # 处理无效值
    if pd.isna(sigma) or pd.isna(sigma_ref) or sigma == 0:
        return base_size
    
    vol_factor = np.clip(sigma_ref / sigma, 0.9, 1.1)  # 限制波动率调整范围在±10%
    pos_size = base_size * vol_factor
    return np.clip(pos_size, 0.1, 0.3)  # 限制在10%-30%之间

def run_strategy(df, params, log_file=None):
    """运行策略 - 修正版"""
    vitality = 0.5
    pos_size = 0.05 + 0.35 * vitality
    max_pos = 0.4

    df['month'] = df.index.to_period('M')
    fee = params['fee']
    
    # 计算参考波动率（仍保留以备后续多 Agent 使用）
    df['sigma_ref'] = df.sigma.rolling(200).median()
    
    # --- 信号生成：双人格模式（均值回复 & 趋势跟随） ---
    # ATR阈值，可配置，默认60分位
    atr_q = df.atr.quantile(params.get('atr_quantile', 0.6))

    # 计算20周期均线供趋势过滤使用
    df['sma20'] = df.close.rolling(20).mean()

    # 初始化信号列
    df['signal'] = False
    df['side'] = 0
    df['persona'] = ''

    # 1) 均值回复人格（ADX 低）
    mask_revert_long = (
        (df.adx < params.get('adx_th_low', 18)) &
        (df.atr > atr_q) &
        (df.rsi < params.get('rsi_long_th', 40)) &
        (df.mom3 < -params.get('mom_th_revert', 0.18)) &
        ((df.close - df.sma20).abs() > 1.5 * df.atr)
    )

    mask_revert_short = (
        (df.adx < params.get('adx_th_low', 18)) &
        (df.atr > atr_q) &
        (df.rsi > params.get('rsi_short_th', 60)) &
        (df.mom3 > params.get('mom_th_revert', 0.18)) &
        ((df.close - df.sma20).abs() > 1.5 * df.atr)
    )

    # 2) 趋势人格（ADX 高）
    mask_trend_long = (
        (df.adx >= params.get('adx_th', 22)) &
        (df.atr > atr_q) &
        (df.mom3 > params.get('mom_th_trend', 0.08)) &
        (df.close > df.sma20)
    )

    mask_trend_short = (
        (df.adx >= params.get('adx_th', 22)) &
        (df.atr > atr_q) &
        (df.mom3 < -params.get('mom_th_trend', 0.08)) &
        (df.close < df.sma20)
    )

    # 写入信号
    df.loc[mask_revert_long,  ['signal', 'side', 'persona']] = [True,  1,  'revert_long']
    df.loc[mask_revert_short, ['signal', 'side', 'persona']] = [True, -1,  'revert_short']
    df.loc[mask_trend_long,   ['signal', 'side', 'persona']] = [True,  1,  'trend_long']
    df.loc[mask_trend_short,  ['signal', 'side', 'persona']] = [True, -1,  'trend_short']

    print(
        "Signals -> revert_long:", mask_revert_long.sum(),
        "revert_short:", mask_revert_short.sum(),
        "trend_long:", mask_trend_long.sum(),
        "trend_short:", mask_trend_short.sum()
    )

    # 占位列（后续统计）
    df['pos'] = pos_size  # 单 Agent，持仓仓位列保持常数，本轮仅用于打印

    # 初始化回测列
    df['pnl_pct'] = 0.0
    df['trade_id'] = 0

    holding = False
    entry_price = 0
    current_tp = 0
    current_sl = 0
    current_pos = 0
    daily_equity = 0
    last_date = None
    trade_id = 0
    cooldown_until = None

    # 动态胜率 - 调整窗口和初始值
    win_hist = []
    cur_win = 0.60  # 提高初始胜率假设

    # 日志记录
    if log_file:
        log_handle = open(log_file, 'w')

    # 先统计上一月的正EV人格
    persona_stats = defaultdict(lambda: defaultdict(list))   # {month: {persona: [pnl,...]}}
    prev_month = None
    i = 0
    enabled_personas = set(['revert_long','revert_short','trend_long','trend_short'])
    while i < len(df) - 3:
        cur_month = df['month'].iloc[i]
        # 月初切换人格！
        if prev_month is not None and cur_month != prev_month:
            if prev_month in persona_stats:
                # 统计上一月各人格胜率！
                stats = {}
                for persona, pnl_list in persona_stats[prev_month].items():
                    if len(pnl_list) >= 5:
                        win_rate = np.mean([p>0 for p in pnl_list])
                        avg_pnl = np.mean(pnl_list)
                        stats[persona] = (win_rate, avg_pnl)
                # 只保留正EV人格
                enabled_personas = set([k for k,v in stats.items() if v[1]>0])
                if not enabled_personas:
                    enabled_personas = set(['revert_long','revert_short','trend_long','trend_short'])
                print(f"\n==== {prev_month} 启用人格: {enabled_personas} ====")
        prev_month = cur_month

        current_date = df.index[i].date()
        # 重置日亏计数
        if last_date != current_date:
            daily_equity = 0
            last_date = current_date
        # 检查日亏熔断
        if daily_equity < -0.015:
            i += 1
            continue
        # 检查冷却期
        if cooldown_until and df.index[i] < cooldown_until:
            i += 1
            continue
        if not holding and df.signal.iloc[i] and df.persona.iloc[i] in enabled_personas:
            # 同步pos列，方便复盘
            df.at[df.index[i], 'pos'] = pos_size
            # ---------------- 开仓 ----------------
            entry_price = df.close.iloc[i]
            atr_now     = df.atr.iloc[i]
            side        = df.side.iloc[i]
            # 统一 RR 设置
            tp_factor, sl_factor = 3.0, 1.0
            current_tp = entry_price + side * tp_factor * atr_now
            current_sl = entry_price - side * sl_factor * atr_now
            current_pos = pos_size
            holding = True
            half_exit_done = False  # 是否已执行 50% 止盈
            trade_id += 1
            print(f"\n开仓 {trade_id}:")
            print(f"时间: {df.index[i]}")
            print(f"方向: {'多' if side==1 else '空'}")
            print(f"价格: {entry_price:.2f}")
            print(f"止盈: {current_tp:.2f}")
            print(f"止损: {current_sl:.2f}")
            print(f"仓位: {current_pos:.4f}")
            print(f"活力: {vitality:.2f}")
            entry_idx = i
        if holding:
            # 持仓期间每根K线都检查追踪止损 - 延长观察期到12根K线
            for j in range(i+1, min(i+13, len(df))):
                px = df.close.iloc[j]
                # 多空方向自适应
                profit_pct = (px - entry_price) / entry_price if side == 1 else (entry_price - px) / entry_price

                # 追踪止损触发点：浮盈1.0 ATR 百分比
                trail_start = 1.0 * atr_now / entry_price
                if profit_pct > trail_start:
                    if side == 1:
                        current_sl = max(current_sl, entry_price + max((px - entry_price) * 0.5, 0.8 * atr_now))
                    else:
                        current_sl = min(current_sl, entry_price - max((entry_price - px) * 0.5, 0.8 * atr_now))
                # 检查止盈止损
                trigger = (px >= current_tp if side == 1 else px <= current_tp) or (px <= current_sl if side == 1 else px >= current_sl)
                ret = (px - entry_price) / entry_price if side == 1 else (entry_price - px) / entry_price
                if trigger:
                    pnl = ret * current_pos - 2 * fee
                    df.iat[j, df.columns.get_loc('pnl_pct')] = pnl
                    df.iat[j, df.columns.get_loc('trade_id')] = trade_id
                    df.iat[j, df.columns.get_loc('side')] = side
                    daily_equity += pnl
                    # persona盈亏累积
                    month_key = df['month'].iloc[j]
                    persona_stats[month_key][df.persona.iloc[entry_idx]].append(pnl)
                    holding = False
                    print(f"\n平仓 {trade_id}:")
                    print(f"时间: {df.index[j]}")
                    print(f"入场价: {entry_price:.2f}")
                    print(f"出场价: {px:.2f}")
                    print(f"收益率: {ret:.4f}")
                    print(f"手续费: {2 * fee:.4f}")
                    print(f"净收益: {pnl:.4f}")
                    print(f"日累计: {daily_equity:.4f}")
                    if log_file:
                        log = {
                            'ts': str(df.index[entry_idx]),
                            'entry': entry_price,
                            'exit': px,
                            'pnl': pnl,
                            'pos': current_pos,
                            'ret': ret,
                            'fee': 2 * fee,
                            'daily_equity': daily_equity
                        }
                        log_handle.write(json.dumps(log) + '\n')
                    # ------- 3. 更新活力 / 仓位 / 胜率 -------
                    vitality = float(np.clip(vitality + pnl * 5, 0.05, 0.95))
                    pos_size = float(np.clip(0.05 + 0.35 * vitality, 0.05, max_pos))

                    # 动态胜率更新 - 缩短窗口到20
                    win_hist.append(int(pnl > 0))
                    if len(win_hist) > 20:  # 缩短窗口
                        win_hist.pop(0)
                    # 只有当样本≥5笔时才更新胜率，避免初期崩塌
                    if len(win_hist) >= 5:
                        cur_win = sum(win_hist) / len(win_hist)
                    # 冷却期落在下一根K线后（保留）
                    cooldown_until = df.index[j+1] + timedelta(hours=4) if (j+1)<len(df) else None
                    print(f"触发冷却期至: {cooldown_until}")
                    i = j  # 跳到平仓K线
                    break
            else:
                i += 1
                continue
        i += 1
    if log_file:
        log_handle.close()
    return df

def print_results(df):
    """打印策略结果+多空分桶"""
    trades = df[df.trade_id > 0]
    signals = df.signal.sum()
    
    # 计算加权收益
    pnl_series = trades.pnl_pct
    total_return = pnl_series.sum()
    avg_return = pnl_series.mean() * 100
    win_rate = (pnl_series > 0).mean() * 100
    
    # 计算回撤
    equity = (1 + pnl_series.fillna(0)).cumprod()
    dd = (equity.cummax() - equity) / equity.cummax()
    max_dd = dd.max() * 100
    
    print(f'信号数: {signals}')
    print(f'实际交易: {len(trades)}')
    print(f'胜率: {win_rate:.2f}%')
    print(f'平均收益: {avg_return:.4f}%')
    print(f'总收益: {total_return:.2%}')
    print(f'最大回撤: {max_dd:.2f}%')
    
    # 多空分桶统计
    if 'side' in trades.columns:
        for label, group in trades.groupby('side', dropna=False):
            if label == 1:
                tag = '多'
            elif label == -1:
                tag = '空'
            else:
                tag = '异常'
            print(f'\n【{tag}】交易数: {len(group)}, 胜率: {(group.pnl_pct>0).mean()*100:.2f}%, 平均收益: {group.pnl_pct.mean()*100:.4f}%')
    
    # 按月统计
    monthly = df.resample('M').agg({
        'trade_id': lambda x: len(x[x > 0]),
        'pnl_pct': 'sum'
    })
    print('\n月度统计:')
    print(monthly)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='数据文件路径')
    parser.add_argument('--log', help='日志文件路径')
    args = parser.parse_args()
    params = {
        'th': 0.5,
        'rsi_th': 37,
        'adx_th': 25,
        'tp_pct': 0.028,
        'sl_pct': 0.012,
        'fee': 0.0005,
        'pos_fix': 0.1,
        'atr_quantile': 0.6,
        'rsi_long_th': 35,
        'rsi_short_th': 65,
        'mom_th_revert': 0.18,
        'mom_th_trend': 0.08,
        'adx_th_low': 20,
    }
    df = prepare_data(args.file)
    df = run_strategy(df, params, args.log)
    print_results(df)

if __name__ == '__main__':
    main() 