import pandas as pd
import numpy as np

def agg_15m(df):
    df['ts'] = pd.to_datetime(df['ts'], unit='us', utc=True)  # 使用微秒
    df.set_index('ts', inplace=True)
    ohlc = df['price'].resample('15min').ohlc()  # 使用min替代T
    ohlc['vol'] = df['qty'].resample('15min').sum()
    return ohlc.dropna()

if __name__ == '__main__':
    # 加载tick数据
    tick_file = '/Users/qiutianyu/Downloads/ETH_tick_202504.csv'
    print("加载tick数据...")
    tick = pd.read_csv(tick_file, names=['trade_id', 'price', 'qty', 'quote_qty', 'ts', 
                                        'is_buyer_maker', 'is_best_match'])
    
    # 聚合到15m
    print("聚合到15分钟K线...")
    kline = agg_15m(tick)
    
    # 保存新K线
    output_file = '/Users/qiutianyu/Downloads/ETHUSDT-15m-2025-04-new.csv'
    print(f"保存到 {output_file}...")
    kline.to_csv(output_file)
    
    # 检查4月15号插针
    april_15 = kline[kline.index.date == pd.to_datetime('2025-04-15').date()]
    if not april_15.empty:
        print("\n4月15号K线数:", len(april_15))
        print("4月15号插针幅度:", 
              (april_15['high'].max() - april_15['low'].min()) / april_15['low'].min() * 100, 
              "%")
        
        # 输出4月15号的具体时间
        spike_time = april_15['high'].idxmax()
        print(f"插针时间: {spike_time}")
    else:
        print("4月15号数据缺失，检查tick数据") 