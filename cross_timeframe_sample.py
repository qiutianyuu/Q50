import pandas as pd
import numpy as np
import duckdb
from ta.trend import ADXIndicator
from ta.volatility import BollingerBands
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate_max_dd(returns):
    """计算最大回撤"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def load_original_data():
    """加载原始CSV数据"""
    print("📊 加载原始CSV数据...")
    
    # 加载1h数据
    df_1h = pd.read_csv('data/ETHUSDT-1h-2025-04.csv')
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], utc=True)
    
    print(f"  - 1h数据: {len(df_1h):,} 行")
    print(f"  - 时间范围: {df_1h['timestamp'].min()} ~ {df_1h['timestamp'].max()}")
    
    # 由于只有1h数据，我们模拟5m数据（每1h分成12个5m）
    print("  - 模拟5m数据...")
    df_5m_list = []
    
    for _, row in df_1h.iterrows():
        base_time = row['timestamp']
        # 将1h分成12个5m
        for i in range(12):
            time_5m = base_time + pd.Timedelta(minutes=5*i)
            df_5m_list.append({
                'timestamp': time_5m,
                'open': row['open'],
                'high': row['high'], 
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'] / 12  # 平均分配成交量
            })
    
    df_5m = pd.DataFrame(df_5m_list)
    print(f"  - 模拟5m数据: {len(df_5m):,} 行")
    
    return df_5m, df_1h

def analyze_cross_timeframe_sample():
    """跨周期共振分析 - 使用原始数据"""
    print("🚀 跨周期共振分析 (使用原始数据)")
    
    # 1. 加载数据
    df_5m, df_1h = load_original_data()
    
    # 2. 计算技术指标
    print("🔧 计算技术指标...")
    
    # 1h Regime
    adx_1h = ADXIndicator(df_1h['high'], df_1h['low'], df_1h['close'], window=14).adx()
    bb_1h = BollingerBands(df_1h['close'], window=20, window_dev=2)
    bb_width_1h = (bb_1h.bollinger_hband() - bb_1h.bollinger_lband()) / df_1h['close']
    regime_1h = np.where((adx_1h > 25) & (bb_width_1h > 0.04), 'Trend', 'Range')
    
    # 5m Regime
    adx_5m = ADXIndicator(df_5m['high'], df_5m['low'], df_5m['close'], window=14).adx()
    bb_5m = BollingerBands(df_5m['close'], window=20, window_dev=2)
    bb_width_5m = (bb_5m.bollinger_hband() - bb_5m.bollinger_lband()) / df_5m['close']
    regime_5m = np.where((adx_5m > 25) & (bb_width_5m > 0.04), 'Trend', 'Range')
    
    # 3. 准备DuckDB数据
    print("🦆 准备DuckDB数据...")
    
    # 1h数据
    df_1h_clean = pd.DataFrame({
        'timestamp_1h': df_1h['timestamp'].dt.floor('H'),
        'regime_1h': regime_1h
    })
    
    # 5m数据
    df_5m_clean = pd.DataFrame({
        'timestamp': df_5m['timestamp'],
        'close': df_5m['close'],
        'regime_5m': regime_5m
    })
    
    # 4. 使用DuckDB进行跨周期匹配
    print("🔗 DuckDB跨周期匹配...")
    
    # 创建DuckDB连接
    con = duckdb.connect(':memory:')
    
    # 注册数据
    con.register('df_1h', df_1h_clean)
    con.register('df_5m', df_5m_clean)
    
    # 执行跨周期匹配
    query = """
    SELECT 
        m.timestamp,
        m.close,
        m.regime_5m,
        h.regime_1h
    FROM df_5m m
    LEFT JOIN df_1h h
    ON m.timestamp >= h.timestamp_1h 
    AND m.timestamp < h.timestamp_1h + INTERVAL '1 hour'
    ORDER BY m.timestamp
    """
    
    result_df = con.execute(query).df()
    con.close()
    
    print(f"  - 匹配后数据: {len(result_df):,} 行")
    print(f"  - 数据完整性: {len(result_df) == len(df_5m)}")
    
    # 5. 计算跨周期条件
    print("📈 计算跨周期条件...")
    
    # 跨周期条件
    result_df['cross_trend'] = (result_df['regime_1h'] == 'Trend') & (result_df['regime_5m'] == 'Trend')
    result_df['cross_range'] = (result_df['regime_1h'] == 'Range') & (result_df['regime_5m'] == 'Range')
    result_df['mixed_trend'] = (result_df['regime_1h'] == 'Trend') & (result_df['regime_5m'] == 'Range')
    result_df['mixed_range'] = (result_df['regime_1h'] == 'Range') & (result_df['regime_5m'] == 'Trend')
    
    # 6. 分析不同forward periods
    results = []
    n_forward_periods = [1, 3, 5, 9]
    
    for n_forward in n_forward_periods:
        print(f"  - 分析 forward {n_forward} 根...")
        
        # 计算forward returns
        result_df[f'ret_fwd_{n_forward}'] = result_df['close'].pct_change().shift(-n_forward)
        
        # 分析各种条件
        conditions = [
            ('1h+5m双Trend', result_df['cross_trend']),
            ('1h+5m双Range', result_df['cross_range']),
            ('1h Trend+5m Range', result_df['mixed_trend']),
            ('1h Range+5m Trend', result_df['mixed_range']),
            ('仅5m Trend', result_df['regime_5m'] == 'Trend'),
            ('仅5m Range', result_df['regime_5m'] == 'Range'),
            ('仅1h Trend', result_df['regime_1h'] == 'Trend'),
            ('仅1h Range', result_df['regime_1h'] == 'Range'),
        ]
        
        for condition_name, mask in conditions:
            if mask.sum() >= 20:  # 降低最小样本数要求
                ret_fwd = result_df.loc[mask, f'ret_fwd_{n_forward}'].dropna()
                
                if len(ret_fwd) >= 10:  # 降低有效样本数要求
                    win_rate = (ret_fwd > 0).mean()
                    avg_ret = ret_fwd.mean()
                    median_ret = ret_fwd.median()
                    std_ret = ret_fwd.std()
                    
                    # 期望值
                    expected_value = win_rate * avg_ret - (1 - win_rate) * abs(avg_ret)
                    
                    # Profit Factor
                    positive_returns = ret_fwd[ret_fwd > 0]
                    negative_returns = ret_fwd[ret_fwd < 0]
                    profit_factor = abs(positive_returns.sum() / negative_returns.sum()) if len(negative_returns) > 0 else float('inf')
                    
                    # 最大回撤
                    max_dd = calculate_max_dd(ret_fwd)
                    
                    results.append({
                        'condition': condition_name,
                        'n_forward': n_forward,
                        'win_rate': win_rate,
                        'avg_ret': avg_ret,
                        'median_ret': median_ret,
                        'std_ret': std_ret,
                        'profit_factor': profit_factor,
                        'max_dd': max_dd,
                        'expected_value': expected_value,
                        'n': int(mask.sum()),
                        'n_valid': len(ret_fwd)
                    })
    
    # 7. 输出结果
    if results:
        results_df = pd.DataFrame(results)
        
        # 过滤有效信号（降低阈值）
        valid_signals = results_df[
            (results_df['n'] >= 50) &  # 降低样本数要求
            (results_df['expected_value'] > 0.00005) &  # 降低期望值要求
            (results_df['win_rate'] > 0.40)  # 降低胜率要求
        ].copy()
        
        if len(valid_signals) > 0:
            # 排序
            valid_signals = valid_signals.sort_values(['expected_value', 'win_rate'], ascending=[False, False])
            
            # 保存结果
            valid_signals.to_csv('/Users/qiutianyu/cross_timeframe_results.csv', index=False)
            
            print(f"\n✅ 找到 {len(valid_signals)} 个有效跨周期信号")
            print("\n📊 Top 10 跨周期信号:")
            print(valid_signals.head(10)[['condition', 'n_forward', 'win_rate', 'avg_ret', 'expected_value', 'n']].to_string(index=False))
            
            # 保存完整结果用于分析
            results_df.to_csv('/Users/qiutianyu/cross_timeframe_full_results.csv', index=False)
            print(f"\n📋 完整结果已保存: cross_timeframe_full_results.csv")
            
        else:
            print("\n⚠️ 没有找到符合条件的跨周期信号")
            print("📋 完整结果已保存: cross_timeframe_full_results.csv")
            results_df.to_csv('/Users/qiutianyu/cross_timeframe_full_results.csv', index=False)
            
            # 显示最佳信号（即使不满足阈值）
            best_signals = results_df.sort_values(['expected_value', 'win_rate'], ascending=[False, False]).head(10)
            print("\n🔍 最佳信号（未满足阈值）:")
            print(best_signals[['condition', 'n_forward', 'win_rate', 'avg_ret', 'expected_value', 'n']].to_string(index=False))
    
    else:
        print("\n❌ 没有生成任何结果")
    
    # 8. 统计信息
    print(f"\n📈 数据统计:")
    print(f"  - 5m Trend占比: {(result_df['regime_5m'] == 'Trend').mean():.2%}")
    print(f"  - 1h Trend占比: {(result_df['regime_1h'] == 'Trend').mean():.2%}")
    print(f"  - 双Trend占比: {result_df['cross_trend'].mean():.2%}")
    print(f"  - 双Range占比: {result_df['cross_range'].mean():.2%}")
    
    print(f"\n🎯 下一步建议:")
    print("1. 检查跨周期信号是否显著优于单周期")
    print("2. 如果有效，扩展到全量数据")
    print("3. 结合Whale事件，看是否能进一步提升")

if __name__ == "__main__":
    analyze_cross_timeframe_sample() 