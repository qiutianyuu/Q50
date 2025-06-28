import pandas as pd
import numpy as np
import itertools
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入策略函数
from rexking_eth_10_4_strategy import load_data

def calculate_adx(df, period=14):
    """计算ADX指标"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # 计算+DM和-DM
    high_diff = high.diff()
    low_diff = low.diff()
    
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    
    plus_dm[high_diff > low_diff.abs()] = high_diff[high_diff > low_diff.abs()]
    minus_dm[low_diff.abs() > high_diff] = low_diff.abs()[low_diff.abs() > high_diff]
    
    # 计算TR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 计算平滑值
    tr_smooth = tr.rolling(period).mean()
    plus_dm_smooth = plus_dm.rolling(period).mean()
    minus_dm_smooth = minus_dm.rolling(period).mean()
    
    # 计算+DI和-DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # 计算DX和ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    
    return adx

def quick_backtest(df, adx_threshold, funding_threshold, volume_ratio):
    """快速回测函数"""
    INIT_CAP = 700.0
    FEE_RATE = 0.00075
    
    # 计算funding z-score
    df['funding_z'] = (df['funding'] - df['funding'].rolling(72).mean()) / df['funding'].rolling(72).std()
    
    # 生成信号
    cond_4h = (
        (df['obv'] > df['obv'].rolling(14).mean()) &
        (df['ema20'] > df['ema60']) &
        (df['atr'] > 0.0005 * df['close']) &
        (df['bb'] > 0.5) &
        (df['close'] > df['ema20']) &
        (df['adx'] > adx_threshold)
    )
    
    cond_1h = (df['funding_z'] < funding_threshold)
    
    df['volume_4h_ma'] = df['volume'].rolling(20).mean()
    cond_15m = (
        df['breakout_15m'] & 
        (df['volmean_15m'].rolling(2).mean() > volume_ratio * df['volume_4h_ma']) &
        (df['close'] > df['high_15m'] * 1.001)
    )
    
    df['signal'] = cond_4h & cond_1h & cond_15m
    
    # 简单回测
    capital = INIT_CAP
    trades = []
    position = 0.0
    entry_price = 0.0
    entry_time = None
    
    for i in range(30, len(df)):
        row = df.iloc[i]
        price = row['close']
        atr = row['atr']
        
        # 平仓逻辑
        if position > 0:
            tp_lvl1 = entry_price + 1.0 * atr
            tp_lvl2 = entry_price + 2.5 * atr
            stop_loss = entry_price - 0.7 * atr
            
            # 触发保本
            if price >= tp_lvl1:
                stop_loss = max(stop_loss, entry_price)
            
            # ADX转弱
            adx_weak = row['adx'] < 20
            
            hit_tp2 = price >= tp_lvl2
            hit_sl = price <= stop_loss
            exit_flag = hit_tp2 or hit_sl or adx_weak
            
            if exit_flag:
                pnl = (price - entry_price) * position
                fee = FEE_RATE * position * (entry_price + price)
                capital += pnl - fee
                
                trades.append({
                    'pnl': pnl - fee,
                    'reason': 'tp' if hit_tp2 else ('sl' if hit_sl else 'adx_weak')
                })
                
                position = 0.0
                entry_price = 0.0
                entry_time = None
        
        # 开仓逻辑
        elif row['signal'] and position == 0.0:
            position_value = min(70, capital * 0.015)  # 固定1.5%风险
            position = position_value / price
            entry_price = price
            entry_time = row['timestamp']
    
    # 计算统计
    if not trades:
        return {
            'adx_threshold': adx_threshold,
            'funding_threshold': funding_threshold,
            'volume_ratio': volume_ratio,
            'total_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'rr_ratio': 0,
            'expected_value': 0,
            'annual_signals': 0,
            'final_capital': INIT_CAP,
            'return_pct': 0
        }
    
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['pnl'] > 0])
    win_rate = winning_trades / total_trades * 100
    total_pnl = sum(t['pnl'] for t in trades)
    
    avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
    avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if (total_trades - winning_trades) > 0 else 0
    
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    expected_value = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
    
    # 计算年化信号数
    total_days = (df['timestamp'].max() - df['timestamp'].min()).days
    annual_signals = df['signal'].sum() / total_days * 365
    
    return {
        'adx_threshold': adx_threshold,
        'funding_threshold': funding_threshold,
        'volume_ratio': volume_ratio,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'rr_ratio': rr_ratio,
        'expected_value': expected_value,
        'annual_signals': annual_signals,
        'final_capital': capital,
        'return_pct': (capital/INIT_CAP - 1) * 100
    }

def grid_search():
    """网格搜索主函数"""
    print("=== 网格搜索优化 ===")
    
    # 加载数据
    print("加载数据...")
    df = load_data('/Users/qiutianyu/ETHUSDT-4h/merged_4h_2023_2025.csv')
    df['adx'] = calculate_adx(df, 14)
    print(f"数据加载完成，共{len(df)}条记录")
    
    # 定义参数网格
    adx_thresholds = [20, 22, 24, 26, 28]
    funding_thresholds = [-0.8, -1.0, -1.2, -1.4, -1.6]
    volume_ratios = [0.07, 0.08, 0.09, 0.10, 0.12]
    
    print(f"\n参数网格:")
    print(f"ADX阈值: {adx_thresholds}")
    print(f"Funding阈值: {funding_thresholds}")
    print(f"量比阈值: {volume_ratios}")
    print(f"总组合数: {len(adx_thresholds) * len(funding_thresholds) * len(volume_ratios)}")
    
    # 生成所有组合
    combinations = list(itertools.product(adx_thresholds, funding_thresholds, volume_ratios))
    
    results = []
    
    # 开始搜索
    print(f"\n开始网格搜索...")
    for i, (adx_th, funding_th, vol_ratio) in enumerate(combinations):
        if (i + 1) % 10 == 0:
            print(f"进度: {i+1}/{len(combinations)}")
        
        result = quick_backtest(df, adx_th, funding_th, vol_ratio)
        results.append(result)
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    
    # 过滤条件
    valid_results = df_results[
        (df_results['expected_value'] > 0) &  # 期望值为正
        (df_results['annual_signals'] >= 80) &  # 年化信号>=80
        (df_results['win_rate'] >= 40)  # 胜率>=40%
    ].copy()
    
    if len(valid_results) == 0:
        print("没有找到符合条件的参数组合，放宽条件...")
        valid_results = df_results[
            (df_results['expected_value'] > -0.1) &  # 期望值>-0.1
            (df_results['annual_signals'] >= 50)  # 年化信号>=50
        ].copy()
    
    # 排序
    valid_results['score'] = (
        valid_results['expected_value'] * 100 +  # 期望值权重100
        valid_results['win_rate'] * 0.5 +        # 胜率权重0.5
        valid_results['rr_ratio'] * 10           # R:R权重10
    )
    
    valid_results = valid_results.sort_values('score', ascending=False)
    
    # 输出结果
    print(f"\n=== 搜索结果 ===")
    print(f"有效组合数: {len(valid_results)}")
    
    if len(valid_results) > 0:
        print(f"\nTOP-10 参数组合:")
        print(valid_results.head(10)[['adx_threshold', 'funding_threshold', 'volume_ratio', 
                                     'total_trades', 'win_rate', 'rr_ratio', 'expected_value', 
                                     'annual_signals', 'return_pct', 'score']].round(2))
        
        # 保存结果
        valid_results.to_csv('grid_search_results.csv', index=False)
        print(f"\n完整结果已保存到 grid_search_results.csv")
        
        # 推荐最佳参数
        best = valid_results.iloc[0]
        print(f"\n=== 推荐参数 ===")
        print(f"ADX阈值: {best['adx_threshold']}")
        print(f"Funding阈值: {best['funding_threshold']}")
        print(f"量比阈值: {best['volume_ratio']}")
        print(f"预期胜率: {best['win_rate']:.1f}%")
        print(f"预期R:R: {best['rr_ratio']:.2f}")
        print(f"预期期望值: ${best['expected_value']:.3f}")
        print(f"年化信号数: {best['annual_signals']:.0f}")
        print(f"预期收益率: {best['return_pct']:.1f}%")
        
        return best
    else:
        print("未找到有效参数组合")
        return None

def main():
    best_params = grid_search()
    
    if best_params is not None:
        print(f"\n=== 下一步操作 ===")
        print("1. 使用推荐参数更新策略文件")
        print("2. 运行完整回测验证效果")
        print("3. 进行walk-forward验证")

if __name__ == "__main__":
    main() 