import os
import glob
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# 数据目录
DIR_4H = '/Users/qiutianyu/ETHUSDT-4h/'
DIR_1H = '/Users/qiutianyu/ETHUSDT-1h/'
DIR_15M = '/Users/qiutianyu/ETHUSDT-15m/'
DIR_FUND = '/Users/qiutianyu/ETHUSDT-fundingRate/'
FILE_W1 = '/Users/qiutianyu/ETHUSDT-w1/etherscan_w1_2023_2025.csv'
OUT_FILE = '/Users/qiutianyu/ETHUSDT-4h/merged_4h_2023_2025.csv'

# 4H数据字段名（无表头）
COLUMNS_4H = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
              'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
              'taker_buy_quote_asset_volume', 'ignore']

def concat_all_csvs(folder, has_header=True, columns=None):
    files = glob.glob(os.path.join(folder, '**', '*.csv'), recursive=True)
    dfs = []
    for f in files:
        try:
            if has_header:
                df = pd.read_csv(f)
            else:
                df = pd.read_csv(f, header=None, names=columns)
            dfs.append(df)
        except Exception as e:
            print(f"读取{f}失败: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def safe_timestamp_convert(ts):
    """安全的时间戳转换"""
    try:
        if isinstance(ts, str):
            ts = int(ts)
        if ts > 1e12:  # 毫秒时间戳
            return pd.to_datetime(ts, unit='ms')
        else:  # 秒时间戳
            return pd.to_datetime(ts, unit='s')
    except:
        return pd.NaT

def floor_to_4h(ts):
    if pd.isna(ts):
        return pd.NaT
    floored = ts - pd.Timedelta(hours=ts.hour % 4, minutes=ts.minute, seconds=ts.second)
    return floored

def calculate_indicators(df):
    """计算技术指标"""
    # OBV
    df['obv'] = (df['volume'] * (df['close'] > df['close'].shift(1)).astype(int)).cumsum()
    
    # EMA
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema60'] = df['close'].ewm(span=60).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100
    
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

def main():
    print('读取4H数据...')
    df_4h = concat_all_csvs(DIR_4H, has_header=False, columns=COLUMNS_4H)
    df_4h['timestamp'] = df_4h['timestamp'].apply(safe_timestamp_convert)
    df_4h = df_4h.dropna(subset=['timestamp'])
    df_4h['4h_time'] = df_4h['timestamp'].apply(floor_to_4h)
    df_4h = calculate_indicators(df_4h)
    df_4h = df_4h.sort_values('4h_time')
    
    print('读取1H数据...')
    df_1h = concat_all_csvs(DIR_1H, has_header=False, columns=COLUMNS_4H)
    df_1h['timestamp'] = df_1h['timestamp'].apply(safe_timestamp_convert)
    df_1h = df_1h.dropna(subset=['timestamp'])
    df_1h['4h_time'] = df_1h['timestamp'].apply(floor_to_4h)
    df_1h = df_1h.sort_values('4h_time')
    
    print('读取15m数据...')
    df_15m = concat_all_csvs(DIR_15M, has_header=False, columns=COLUMNS_4H)
    df_15m['timestamp'] = df_15m['timestamp'].apply(safe_timestamp_convert)
    df_15m = df_15m.dropna(subset=['timestamp'])
    df_15m['4h_time'] = df_15m['timestamp'].apply(floor_to_4h)
    df_15m = df_15m.sort_values('4h_time')
    
    print('读取funding数据...')
    df_fund = concat_all_csvs(DIR_FUND, has_header=True)
    if 'calc_time' in df_fund.columns:
        df_fund['timestamp'] = df_fund['calc_time'].apply(safe_timestamp_convert)
        df_fund['funding'] = df_fund['last_funding_rate']
        df_fund['4h_time'] = df_fund['timestamp'].apply(floor_to_4h)
        df_fund = df_fund.sort_values('4h_time')
        print(f"Funding数据: {len(df_fund)}条记录")
    else:
        print("Warning: funding数据格式异常")
        df_fund = pd.DataFrame()
    
    print('读取W1数据...')
    df_w1 = pd.read_csv(FILE_W1)
    df_w1['timestamp'] = pd.to_datetime(df_w1['timestamp'])
    df_w1['4h_time'] = df_w1['timestamp'].apply(floor_to_4h)
    df_w1 = df_w1.sort_values('4h_time')
    
    # 以4H为主轴，合并所有数据
    print('合并数据...')
    df = df_4h.copy()
    keep_4h = ['4h_time','open','high','low','close','volume','obv','ema20','ema60','atr','bb']
    df = df[keep_4h]
    
    # 1H: 取每4H区间内funding最小值
    if not df_fund.empty and 'funding' in df_fund.columns:
        df_fund_group = df_fund.groupby('4h_time').agg({'funding':'min'})
        df = df.merge(df_fund_group, on='4h_time', how='left')
        print(f"Funding信号: {(df['funding'] < 0.00005).sum()}条")
    else:
        df['funding'] = 0.0
    
    # 15m: 取每4H区间内最高价突破和成交量均值
    df_15m_group = df_15m.groupby('4h_time').agg({'high':'max','volume':'mean'})
    df_15m_group = df_15m_group.rename(columns={'high':'high_15m','volume':'volmean_15m'})
    df = df.merge(df_15m_group, on='4h_time', how='left')
    
    # 修正15m突破逻辑：应该和前一4H区间的高价比较
    df['high_prev_4h'] = df['high'].shift(1)
    df['breakout_15m'] = df['high_15m'] >= df['high_prev_4h'] * 1.0001  # 突破0.01%
    df['volume_4h_ma'] = df['volume'].rolling(20).mean()
    df['volume_surge_15m'] = (df['volmean_15m'] > 0.05 * df['volume_4h_ma']) & (df['volume_4h_ma'].notna())  # 5%的4H均值
    
    # W1: 取每4H区间内最大value、最大zscore、信号，并添加滚动窗口
    df_w1_group = df_w1.groupby('4h_time').agg({'value':'max','w1_zscore':'max','w1_signal':'max'})
    df_w1_group = df_w1_group.rename(columns={'value':'w1_value','w1_zscore':'w1_zscore','w1_signal':'w1_signal'})
    df = df.merge(df_w1_group, on='4h_time', how='left')
    
    # 填充缺失值
    df['w1_value'] = df['w1_value'].fillna(0)
    df['w1_zscore'] = df['w1_zscore'].fillna(0)
    df['w1_signal'] = df['w1_signal'].fillna(0)
    df['funding'] = df['funding'].fillna(0)
    
    # 添加W1滚动窗口信号（过去12小时内出现过信号）
    df['w1_signal_rolling'] = df['w1_signal'].rolling(window=3, min_periods=1).max()
    
    # 计算ADX指标（趋势强度）
    df['adx'] = calculate_adx(df, 14)
    
    # 整理输出
    df = df.sort_values('4h_time')
    df = df.reset_index(drop=True)
    df = df.rename(columns={'4h_time':'timestamp'})
    
    # 选择输出字段
    output_cols = ['timestamp','open','high','low','close','volume','obv','ema20','ema60','atr','bb',
                   'funding','high_15m','volmean_15m','breakout_15m','volume_surge_15m',
                   'w1_value','w1_zscore','w1_signal','w1_signal_rolling','adx']
    df = df[output_cols]
    
    df.to_csv(OUT_FILE, index=False)
    print(f'合并完成，输出：{OUT_FILE}，共{len(df)}行')
    print(f'数据范围：{df["timestamp"].min()} 到 {df["timestamp"].max()}')
    print(f'W1原始信号: {df["w1_signal"].sum()}条')
    print(f'W1滚动信号: {df["w1_signal_rolling"].sum()}条')
    print(f'15m突破信号: {df["breakout_15m"].sum()}条')
    print(f'15m放量信号: {df["volume_surge_15m"].sum()}条')

if __name__ == '__main__':
    main() 