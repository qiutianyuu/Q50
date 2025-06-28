import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
from utils.labeling import make_labels, get_label_stats, compute_cost

def load_latest_features():
    """Load the latest realtime features file"""
    files = glob.glob("data/realtime_features_*.parquet")
    if not files:
        raise FileNotFoundError("No realtime features files found")
    
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading: {latest_file}")
    df = pd.read_parquet(latest_file)
    print(f"Loaded {len(df)} rows")
    return df

def scan_params(df, horizons, alphas, fee_rate, mode='taker', require_fill=False):
    results = []
    for horizon in horizons:
        for alpha in alphas:
            labels = make_labels(df['mid_price'], df['rel_spread'], horizon, alpha, fee_rate, mode, require_fill)
            stats = get_label_stats(labels)
            if mode == 'maker':
                avg_cost = (0.5 * df['rel_spread'] + 0.0001).mean()
            else:
                avg_cost = compute_cost(df['rel_spread'], fee_rate).mean()
            avg_abs_ret = ((df['mid_price'].shift(-horizon) - df['mid_price']) / df['mid_price']).abs().mean()
            results.append({
                'mode': mode,
                'require_fill': require_fill,
                'horizon': horizon,
                'alpha': alpha,
                'long_pct': stats['long_pct'],
                'short_pct': stats['short_pct'],
                'neutral_pct': stats['neutral_pct'],
                'n': stats['total'],
                'avg_cost': avg_cost,
                'avg_abs_ret': avg_abs_ret
            })
    return results

def print_results(results, csv_path=None):
    print("mode   | fill  | horizon | alpha | long% | short% | neutral% | n | avg_cost | avg_abs_ret")
    print("-"*100)
    rows = []
    for r in results:
        fill_str = "Yes" if r['require_fill'] else "No"
        print(f"{r['mode']:<7} | {fill_str:<4} | {r['horizon']:>7} | {r['alpha']:<5.2f} | {r['long_pct']:>6.1f} | {r['short_pct']:>7.1f} | {r['neutral_pct']:>8.1f} | {r['n']:>5} | {r['avg_cost']:.5f} | {r['avg_abs_ret']:.5f}")
        rows.append(r)
    if csv_path:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"结果已保存: {csv_path}")

def main():
    df = load_latest_features()
    horizons = [60, 120, 240, 600, 1200]  # 扩展horizon
    alphas = [0.3, 0.6, 1.0]
    fee_rate = 0.0005
    
    all_results = []
    
    # 扫描maker模式 - 无填单验证
    print("=== MAKER模式扫描 (无填单验证) ===")
    maker_results = scan_params(df, horizons, alphas, fee_rate, 'maker', require_fill=False)
    all_results.extend(maker_results)
    
    # 扫描maker模式 - 有填单验证
    print("\n=== MAKER模式扫描 (有填单验证) ===")
    maker_fill_results = scan_params(df, horizons, alphas, fee_rate, 'maker', require_fill=True)
    all_results.extend(maker_fill_results)
    
    # 扫描taker模式 - 无填单验证
    print("\n=== TAKER模式扫描 (无填单验证) ===")
    taker_results = scan_params(df, horizons, alphas, fee_rate, 'taker', require_fill=False)
    all_results.extend(taker_results)
    
    # 扫描taker模式 - 有填单验证
    print("\n=== TAKER模式扫描 (有填单验证) ===")
    taker_fill_results = scan_params(df, horizons, alphas, fee_rate, 'taker', require_fill=True)
    all_results.extend(taker_fill_results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"micro_label_param_scan_{timestamp}.csv"
    print_results(all_results, csv_path)
    
    # 找出最佳组合
    print("\n=== 最佳组合（有效信号≥20%） ===")
    best_results = [r for r in all_results if (r['long_pct'] + r['short_pct']) >= 20]
    if best_results:
        best_results.sort(key=lambda x: abs(x['long_pct'] - x['short_pct']))  # 按平衡性排序
        print_results(best_results[:10])  # 显示前10个最佳组合
    else:
        print("没有找到有效信号≥20%的组合")
    
    # 对比填单验证的影响
    print("\n=== 填单验证影响分析 ===")
    for mode in ['maker', 'taker']:
        for horizon in [60, 120]:
            for alpha in [0.3, 0.6]:
                no_fill = [r for r in all_results if r['mode'] == mode and r['horizon'] == horizon and r['alpha'] == alpha and not r['require_fill']]
                with_fill = [r for r in all_results if r['mode'] == mode and r['horizon'] == horizon and r['alpha'] == alpha and r['require_fill']]
                
                if no_fill and with_fill:
                    no_fill = no_fill[0]
                    with_fill = with_fill[0]
                    signal_loss = (no_fill['long_pct'] + no_fill['short_pct']) - (with_fill['long_pct'] + with_fill['short_pct'])
                    print(f"{mode} h{horizon} α{alpha}: 信号损失 {signal_loss:.1f}% (无填单: {no_fill['long_pct']+no_fill['short_pct']:.1f}% → 有填单: {with_fill['long_pct']+with_fill['short_pct']:.1f}%)")

if __name__ == "__main__":
    main() 