import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def load_data(file_path):
    """加载4H数据"""
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 转换数值列
    numeric_cols = ['open','high','low','close','volume','obv','ema20','ema60',
                   'atr','bb','funding','high_15m','volmean_15m','breakout_15m',
                   'volume_surge_15m','w1_value','w1_zscore','w1_signal','w1_signal_rolling']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

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

def analyze_signals(df):
    """分析信号质量"""
    print("=== 信号质量分析 ===\n")
    
    # 计算ADX
    df['adx'] = calculate_adx(df, 14)
    
    # 计算EMA斜率
    df['ema20_slope'] = df['ema20'].diff(5) / df['ema20'].shift(5) * 100
    
    # 计算funding z-score
    df['funding_zscore'] = (df['funding'] - df['funding'].rolling(100).mean()) / df['funding'].rolling(100).std()
    
    # 计算BB宽度
    df['bb_width'] = (df['bb'] - df['bb'].rolling(20).min()) / (df['bb'].rolling(20).max() - df['bb'].rolling(20).min())
    
    # 计算OI变化（模拟）
    df['oi_change'] = df['volume'].pct_change(4) * 100
    
    # 生成信号
    cond_4h = (
        (df['obv'] > df['obv'].rolling(14).mean()) &
        (df['ema20'] > df['ema60']) &
        (df['atr'] > 0.0005 * df['close']) &
        (df['bb'] > 0.5) &
        (df['close'] > df['ema20']) &
        (df['adx'] > 15)
    )
    
    cond_1h = (df['funding'] < 0.00005)
    
    df['volume_4h_ma'] = df['volume'].rolling(20).mean()
    cond_15m = df['breakout_15m'] & (df['volmean_15m'].rolling(2).mean() > 0.05 * df['volume_4h_ma'])
    
    cond_w1 = (df['w1_value'] > 1000) & (df['w1_zscore'] > 0.5) & (df['w1_signal_rolling'] > 0)
    
    df['signal'] = cond_4h & cond_1h & cond_15m
    
    # 统计信号
    print(f"4H信号: {cond_4h.sum()}条")
    print(f"Funding信号: {cond_1h.sum()}条")
    print(f"15m突破信号: {cond_15m.sum()}条")
    print(f"W1信号: {cond_w1.sum()}条")
    print(f"联合信号: {df['signal'].sum()}条")
    
    # 分析信号分布
    print(f"\n=== 信号分布分析 ===")
    
    # ADX分布
    signal_adx = df[df['signal']]['adx']
    print(f"信号ADX均值: {signal_adx.mean():.1f}")
    print(f"信号ADX中位数: {signal_adx.median():.1f}")
    print(f"ADX > 20的信号: {len(signal_adx[signal_adx > 20])}条")
    print(f"ADX > 25的信号: {len(signal_adx[signal_adx > 25])}条")
    
    # Funding分布
    signal_funding = df[df['signal']]['funding']
    print(f"\n信号Funding均值: {signal_funding.mean():.6f}")
    print(f"Funding < -0.0001的信号: {len(signal_funding[signal_funding < -0.0001])}条")
    print(f"Funding < -0.0002的信号: {len(signal_funding[signal_funding < -0.0002])}条")
    
    # 15m突破质量
    signal_volume = df[df['signal']]['volmean_15m']
    signal_volume_ratio = signal_volume / df[df['signal']]['volume_4h_ma']
    print(f"\n15m量比均值: {signal_volume_ratio.mean():.2f}")
    print(f"量比 > 2的信号: {len(signal_volume_ratio[signal_volume_ratio > 2])}条")
    print(f"量比 > 3的信号: {len(signal_volume_ratio[signal_volume_ratio > 3])}条")
    
    return df

def analyze_false_breakouts(df):
    """分析假突破问题"""
    print(f"\n=== 假突破分析 ===")
    
    # 识别假突破（信号后4根K线内价格回落到突破前水平）
    false_breakouts = []
    
    for i in range(len(df) - 4):
        if df.iloc[i]['signal']:
            entry_price = df.iloc[i]['close']
            entry_high = df.iloc[i]['high_15m']
            
            # 检查后续4根K线
            for j in range(1, 5):
                if i + j >= len(df):
                    break
                
                current_price = df.iloc[i + j]['close']
                price_change = (current_price - entry_price) / entry_price
                
                # 如果价格回落到突破前水平，认为是假突破
                if price_change < -0.005:  # 5%回撤
                    false_breakouts.append({
                        'entry_time': df.iloc[i]['timestamp'],
                        'entry_price': entry_price,
                        'breakout_high': entry_high,
                        'false_breakout_time': df.iloc[i + j]['timestamp'],
                        'false_breakout_price': current_price,
                        'price_change': price_change * 100
                    })
                    break
    
    print(f"假突破数量: {len(false_breakouts)}")
    if false_breakouts:
        avg_false_breakout_change = np.mean([fb['price_change'] for fb in false_breakouts])
        print(f"平均假突破回撤: {avg_false_breakout_change:.1f}%")
    
    return false_breakouts

def plot_signal_analysis(df):
    """绘制信号分析图表"""
    plt.figure(figsize=(15, 10))
    
    # ADX分布
    plt.subplot(2, 3, 1)
    plt.hist(df[df['signal']]['adx'], bins=20, alpha=0.7, color='blue')
    plt.axvline(x=20, color='red', linestyle='--', label='ADX=20')
    plt.axvline(x=25, color='orange', linestyle='--', label='ADX=25')
    plt.xlabel('ADX值')
    plt.ylabel('频次')
    plt.title('信号ADX分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Funding分布
    plt.subplot(2, 3, 2)
    plt.hist(df[df['signal']]['funding'], bins=20, alpha=0.7, color='green')
    plt.axvline(x=-0.0001, color='red', linestyle='--', label='Funding=-0.01%')
    plt.xlabel('Funding Rate')
    plt.ylabel('频次')
    plt.title('信号Funding分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 15m量比分布
    plt.subplot(2, 3, 3)
    signal_volume_ratio = df[df['signal']]['volmean_15m'] / df[df['signal']]['volume_4h_ma']
    plt.hist(signal_volume_ratio, bins=20, alpha=0.7, color='purple')
    plt.axvline(x=2, color='red', linestyle='--', label='量比=2')
    plt.axvline(x=3, color='orange', linestyle='--', label='量比=3')
    plt.xlabel('15m量比')
    plt.ylabel('频次')
    plt.title('信号量比分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 信号时间分布
    plt.subplot(2, 3, 4)
    df['hour'] = df['timestamp'].dt.hour
    signal_hours = df[df['signal']]['hour'].value_counts().sort_index()
    plt.bar(signal_hours.index, signal_hours.values, alpha=0.7, color='orange')
    plt.xlabel('小时')
    plt.ylabel('信号数量')
    plt.title('信号时间分布')
    plt.grid(True, alpha=0.3)
    
    # 信号月度分布
    plt.subplot(2, 3, 5)
    df['month'] = df['timestamp'].dt.to_period('M')
    signal_months = df[df['signal']]['month'].value_counts().sort_index()
    plt.bar(range(len(signal_months)), signal_months.values, alpha=0.7, color='brown')
    plt.xlabel('月份')
    plt.ylabel('信号数量')
    plt.title('信号月度分布')
    plt.xticks(range(len(signal_months)), [str(m) for m in signal_months.index], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 信号强度热力图
    plt.subplot(2, 3, 6)
    signal_data = df[df['signal']][['adx', 'funding', 'bb']].corr()
    sns.heatmap(signal_data, annot=True, cmap='coolwarm', center=0)
    plt.title('信号指标相关性')
    
    plt.tight_layout()
    plt.savefig('signal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def optimize_thresholds(df):
    """优化信号阈值"""
    print(f"\n=== 阈值优化建议 ===")
    
    # ADX阈值优化
    adx_thresholds = [15, 18, 20, 22, 25]
    print("ADX阈值优化:")
    for threshold in adx_thresholds:
        count = len(df[df['adx'] > threshold])
        print(f"  ADX > {threshold}: {count}条信号")
    
    # Funding阈值优化
    funding_thresholds = [-0.00005, -0.0001, -0.00015, -0.0002]
    print("\nFunding阈值优化:")
    for threshold in funding_thresholds:
        count = len(df[df['funding'] < threshold])
        print(f"  Funding < {threshold}: {count}条信号")
    
    # 量比阈值优化
    volume_ratio_thresholds = [0.03, 0.05, 0.08, 0.1]
    print("\n量比阈值优化:")
    for threshold in volume_ratio_thresholds:
        condition = df['volmean_15m'].rolling(2).mean() > threshold * df['volume_4h_ma']
        count = len(df[condition & df['breakout_15m']])
        print(f"  量比 > {threshold}: {count}条信号")

def main():
    # 加载数据
    df = load_data('/Users/qiutianyu/ETHUSDT-4h/merged_4h_2023_2025.csv')
    
    # 信号分析
    df = analyze_signals(df)
    
    # 假突破分析
    false_breakouts = analyze_false_breakouts(df)
    
    # 绘制分析图表
    plot_signal_analysis(df)
    
    # 阈值优化
    optimize_thresholds(df)
    
    # 输出优化建议
    print(f"\n=== 信号优化建议 ===")
    print("1. 提高ADX阈值到20-25，过滤弱趋势信号")
    print("2. 加强funding过滤，要求funding < -0.0001")
    print("3. 提高量比要求到0.08-0.1，确保放量突破")
    print("4. 增加假突破确认机制，要求突破后回撤<2%")
    print("5. 考虑增加EMA斜率过滤，确保趋势明确")

if __name__ == "__main__":
    main() 