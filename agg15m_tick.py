import pandas as pd
import numpy as np

def agg_15m(df):
    df['ts'] = pd.to_datetime(df['ts'], unit='us', utc=True)
    df.set_index('ts', inplace=True)
    ohlc = df['price'].resample('15min').ohlc()
    ohlc['vol'] = df['qty'].resample('15min').sum()
    return ohlc.dropna()

if __name__ == '__main__':
    # 加载tick数据
    tick_file = '/Users/qiutianyu/Downloads/ETH_tick_202504.csv'
    print("加载tick数据...")
    tick = pd.read_csv(tick_file, names=['trade_id', 'price', 'qty', 'quote_qty', 'ts', 
                                        'is_buyer_maker', 'is_best_match'])
    
    # 加载新K线
    new_kline_file = '/Users/qiutianyu/Downloads/ETHUSDT-15m-2025-04-new.csv'
    print("加载新K线...")
    new_kline = pd.read_csv(new_kline_file)
    new_kline['ts'] = pd.to_datetime(new_kline['ts'], utc=True)
    new_kline.set_index('ts', inplace=True)
    
    # 加载老K线（改过的）
    old_kline_file = '/Users/qiutianyu/Downloads/ETHUSDT-15m-2025-04.csv'
    print("加载老K线...")
    old_kline = pd.read_csv(old_kline_file, names=['ts', 'open', 'high', 'low', 'close', 'vol'])
    old_kline['ts'] = pd.to_datetime(old_kline['ts'], unit='us', errors='coerce', utc=True)
    old_kline.set_index('ts', inplace=True)
    
    # 聚合tick到15m
    print("聚合tick数据...")
    hist = agg_15m(tick)
    
    # 比较新K线
    print("\n=== 新K线（tick vs. ETHUSDT-15m-2025-04-new.csv） ===")
    consistency_new = hist.equals(new_kline[['open', 'high', 'low', 'close', 'vol']])
    print("一致性:", consistency_new)
    for col in ['open', 'high', 'low', 'close']:
        diff = np.abs((hist[col] - new_kline[col]) / hist[col]) * 100
        max_diff = diff.max()
        print(f"{col} 最大误差: {max_diff:.4f}%")
    
    # 比较老K线
    print("\n=== 老K线（tick vs. ETHUSDT-15m-2025-04.csv） ===")
    consistency_old = hist.equals(old_kline[['open', 'high', 'low', 'close', 'vol']])
    print("一致性:", consistency_old)
    for col in ['open', 'high', 'low', 'close']:
        diff = np.abs((hist[col] - old_kline[col]) / hist[col]) * 100
        max_diff = diff.max()
        print(f"{col} 最大误差: {max_diff:.4f}%")
    
    # 4月15号插针
    april_15 = hist[hist.index.date == pd.to_datetime('2025-04-15').date()]
    if not april_15.empty:
        print("\n4月15号K线数:", len(april_15))
        print("4月15号插针幅度:", 
              (april_15['high'].max() - april_15['low'].min()) / april_15['low'].min() * 100, 
              "%")
        print("插针时间:", april_15['high'].idxmax())
    else:
        print("4月15号数据缺失，检查tick数据") 