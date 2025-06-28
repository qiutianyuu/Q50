#!/usr/bin/env python3
"""
RexKing ETH Exploratory Statistics
运行探索性统计分析，输出高胜率条件
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ta.trend import ADXIndicator
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

TIMEFRAMES = {
    "5m": "merged_5m_2023_2025.parquet",
    "15m": "merged_15m_2023_2025.parquet",
    "1h": "merged_1h_2023_2025.parquet",
}

def calculate_max_dd(returns):
    """计算最大回撤"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def analyze_grid_optimized(df: pd.DataFrame, tf: str) -> list:
    """优化版本，减少内存使用"""
    results = []
    
    # 只选择关键列，减少内存
    df_work = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # 网格搜索参数（减少组合数）
    adx_thresholds = [20, 25, 30]
    bb_thresholds = [0.03, 0.04, 0.05]
    n_forward_periods = [1, 3, 5]
    
    print(f'  - 计算技术指标...')
    # 计算技术指标
    adx = ADXIndicator(df_work['high'], df_work['low'], df_work['close'], window=14).adx()
    bb = BollingerBands(df_work['close'], window=20, window_dev=2)
    bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / df_work['close']
    
    # 预计算所有forward returns
    forward_returns = {}
    for n_forward in n_forward_periods:
        forward_returns[n_forward] = df_work['close'].pct_change().shift(-n_forward)
    
    print(f'  - 网格搜索 ({len(adx_thresholds)}×{len(bb_thresholds)}×{len(n_forward_periods)} 组合)...')
    combo_count = 0
    total_combos = len(adx_thresholds) * len(bb_thresholds) * len(n_forward_periods) * 2  # *2 for Trend/Range
    
    # 网格搜索
    for adx_th in adx_thresholds:
        for bb_th in bb_thresholds:
            # Regime分类
            regime = np.where((adx > adx_th) & (bb_width > bb_th), 'Trend', 'Range')
            
            for n_forward in n_forward_periods:
                ret_fwd = forward_returns[n_forward]
                
                # 分析每个regime
                for regime_type in ['Trend', 'Range']:
                    combo_count += 1
                    if combo_count % 10 == 0:
                        print(f'    - 进度: {combo_count}/{total_combos}')
                    
                    mask = regime == regime_type
                    if mask.sum() < 100:  # 降低样本数要求
                        continue
                        
                    regime_returns = ret_fwd[mask].dropna()
                    if len(regime_returns) < 50:  # 降低样本数要求
                        continue
                    
                    # 计算统计指标
                    win_rate = (regime_returns > 0).mean()
                    avg_ret = regime_returns.mean()
                    median_ret = regime_returns.median()
                    std_ret = regime_returns.std()
                    
                    # 期望值
                    expected_value = win_rate * avg_ret - (1 - win_rate) * abs(avg_ret)
                    
                    # 只保存有意义的信号 - 降低阈值
                    if expected_value > 0.0001 and win_rate > 0.45:  # 降低阈值
                        # Profit Factor (简化版)
                        positive_returns = regime_returns[regime_returns > 0]
                        negative_returns = regime_returns[regime_returns < 0]
                        profit_factor = abs(positive_returns.sum() / negative_returns.sum()) if len(negative_returns) > 0 else float('inf')
                        
                        # 计算最大回撤（只在需要时计算）
                        max_dd = calculate_max_dd(regime_returns)
                        
                        results.append({
                            'timeframe': tf,
                            'condition': f'{regime_type}段',
                            'win_rate': win_rate,
                            'avg_ret': avg_ret,
                            'median_ret': median_ret,
                            'std_ret': std_ret,
                            'profit_factor': profit_factor,
                            'max_dd': max_dd,
                            'expected_value': expected_value,
                            'type': 'regime_grid',
                            'n': int(mask.sum()),
                            'adx_threshold': adx_th,
                            'bb_threshold': bb_th,
                            'n_forward': n_forward
                        })
    
    # 形态分析（简化版，只做最重要的）
    print(f'  - 形态分析...')
    def is_pin_bar(r):
        total = r['high'] - r['low']
        if total == 0:
            return False
        upper = r['high'] - max(r['close'], r['open'])
        lower = min(r['close'], r['open']) - r['low']
        return (upper/total > .5) or (lower/total > .5)
    
    df_work['pin'] = df_work.apply(is_pin_bar, axis=1)
    
    # 只分析n_forward=3的情况
    ret_pin = df_work.loc[df_work.pin, 'close'].shift(-3) / df_work.loc[df_work.pin, 'close'] - 1
    if len(ret_pin.dropna()) > 50:  # 降低样本数要求
        win_rate = (ret_pin > 0).mean()
        avg_ret = ret_pin.mean()
        expected_value = win_rate * avg_ret - (1 - win_rate) * abs(avg_ret)
        
        if expected_value > 0.0002 and win_rate > 0.45:  # 降低阈值
            results.append({
                'timeframe': tf,
                'condition': 'Pin Bar后3根',
                'win_rate': win_rate,
                'avg_ret': avg_ret,
                'median_ret': ret_pin.median(),
                'std_ret': ret_pin.std(),
                'profit_factor': abs(ret_pin[ret_pin > 0].sum() / ret_pin[ret_pin < 0].sum()) if len(ret_pin[ret_pin < 0]) > 0 else float('inf'),
                'max_dd': calculate_max_dd(ret_pin),
                'expected_value': expected_value,
                'type': 'pattern',
                'n': int(df_work.pin.sum()),
                'adx_threshold': None,
                'bb_threshold': None,
                'n_forward': 3
            })
    
    return results

def analyze_cross_timeframe_optimized(df_1h: pd.DataFrame, df_5m: pd.DataFrame) -> list:
    """跨周期共振分析（优化版）"""
    results = []
    
    print('  - 跨周期分析...')
    
    # 只选择需要的列
    df_1h_work = df_1h[['timestamp', 'open', 'high', 'low', 'close']].copy()
    df_5m_work = df_5m[['timestamp', 'open', 'high', 'low', 'close']].copy()
    
    # 1h Regime
    adx_1h = ADXIndicator(df_1h_work['high'], df_1h_work['low'], df_1h_work['close'], window=14).adx()
    bb_1h = BollingerBands(df_1h_work['close'], window=20, window_dev=2)
    bb_width_1h = (bb_1h.bollinger_hband() - bb_1h.bollinger_lband()) / df_1h_work['close']
    regime_1h = np.where((adx_1h > 25) & (bb_width_1h > 0.04), 'Trend', 'Range')
    
    # 5m Regime
    adx_5m = ADXIndicator(df_5m_work['high'], df_5m_work['low'], df_5m_work['close'], window=14).adx()
    bb_5m = BollingerBands(df_5m_work['close'], window=20, window_dev=2)
    bb_width_5m = (bb_5m.bollinger_hband() - bb_5m.bollinger_lband()) / df_5m_work['close']
    regime_5m = np.where((adx_5m > 25) & (bb_width_5m > 0.04), 'Trend', 'Range')
    
    # 将1h regime映射到5m
    df_5m_work['timestamp_1h'] = df_5m_work['timestamp'].dt.floor('H')
    df_1h_work['timestamp_1h'] = df_1h_work['timestamp'].dt.floor('H')
    regime_1h_df = pd.DataFrame({'timestamp_1h': df_1h_work['timestamp_1h'], 'regime_1h': regime_1h})
    df_5m_work = df_5m_work.merge(regime_1h_df, on='timestamp_1h', how='left')
    
    # 跨周期条件
    cross_trend = (df_5m_work['regime_1h'] == 'Trend') & (regime_5m == 'Trend')
    cross_range = (df_5m_work['regime_1h'] == 'Range') & (regime_5m == 'Range')
    
    for condition_name, mask in [('1h+5m双Trend', cross_trend), ('1h+5m双Range', cross_range)]:
        if mask.sum() > 200:
            ret_fwd = df_5m_work.loc[mask, 'close'].pct_change().shift(-3)
            ret_fwd = ret_fwd.dropna()
            
            if len(ret_fwd) > 100:
                win_rate = (ret_fwd > 0).mean()
                avg_ret = ret_fwd.mean()
                expected_value = win_rate * avg_ret - (1 - win_rate) * abs(avg_ret)
                
                if expected_value > 0.0002 and win_rate > 0.48:
                    results.append({
                        'timeframe': '5m',
                        'condition': condition_name,
                        'win_rate': win_rate,
                        'avg_ret': avg_ret,
                        'median_ret': ret_fwd.median(),
                        'std_ret': ret_fwd.std(),
                        'profit_factor': abs(ret_fwd[ret_fwd > 0].sum() / ret_fwd[ret_fwd < 0].sum()) if len(ret_fwd[ret_fwd < 0]) > 0 else float('inf'),
                        'max_dd': calculate_max_dd(ret_fwd),
                        'expected_value': expected_value,
                        'type': 'cross_timeframe',
                        'n': int(mask.sum()),
                        'adx_threshold': 25,
                        'bb_threshold': 0.04,
                        'n_forward': 3
                    })
    
    return results

def main():
    DATA_PATH = Path('/Users/qiutianyu/data/processed')
    all_stats = []
    
    # 加载数据
    dfs = {}
    for tf, fname in TIMEFRAMES.items():
        print(f'▶ 加载 {tf}...')
        dfs[tf] = pd.read_parquet(DATA_PATH / fname)
        print(f'  - 数据形状: {dfs[tf].shape}')
    
    # 单周期网格搜索
    for tf, df in dfs.items():
        print(f'▶ 网格搜索 {tf}...')
        stats = analyze_grid_optimized(df, tf)
        all_stats.extend(stats)
        print(f'  - 找到 {len(stats)} 个信号')
    
    # 跨周期共振分析
    # if '1h' in dfs and '5m' in dfs:
    #     print('▶ 跨周期共振分析...')
    #     cross_stats = analyze_cross_timeframe_optimized(dfs['1h'], dfs['5m'])
    #     all_stats.extend(cross_stats)
    #     print(f'  - 找到 {len(cross_stats)} 个跨周期信号')
    
    # 生成热力图（只做5m的）
    print('▶ 生成热力图...')
    df_5m = dfs['5m']
    pivot = (df_5m.assign(ret=df_5m['close'].pct_change().shift(-1),
                         win=lambda d: (d.ret > 0).astype(int))
               .pivot_table(index='hour', columns='weekday',
                            values='win', aggfunc='mean'))
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5)
    plt.title('5m 多头胜率热力图 (hour × weekday)')
    plt.tight_layout()
    plt.savefig('/Users/qiutianyu/heatmap_winrate_5m.png', dpi=300)
    plt.close()
    
    # 保存结果
    stats_df = pd.DataFrame(all_stats)
    if len(stats_df) > 0:
        # 过滤有效结果
        stats_df = stats_df[
            (stats_df['n'] >= 200) &  # 提高样本数要求
            (stats_df['expected_value'] > 0.0002) &  # 提高期望值要求
            (stats_df['win_rate'] > 0.48)  # 提高胜率要求
        ]
        
        # 排序
        stats_df = stats_df.sort_values(['expected_value', 'win_rate'], ascending=[False, False])
        
        # 保存
        stats_df.to_csv('/Users/qiutianyu/research_stats.csv', index=False)
        
        print(f'✅ 分析完成！找到 {len(stats_df)} 个有效信号')
        print('\n📊 Top 10 信号:')
        print(stats_df.head(10)[['timeframe', 'condition', 'win_rate', 'avg_ret', 'expected_value', 'n']].to_string(index=False))
    else:
        print('⚠️ 没有找到符合条件的信号')
    
    print('\n🎯 下一步建议:')
    print('1. 检查 Top 信号，选择期望值最高的组合')
    print('2. 整合 Whale 数据，看是否能提升胜率')
    print('3. 开发多因子模型，组合多个信号')

if __name__ == "__main__":
    main() 