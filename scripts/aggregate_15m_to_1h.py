import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

INPUT_FILE = "/Users/qiutianyu/Downloads/ETHUSDT-15m-2025-04.csv"
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ETHUSDT-1h-2025-04.csv")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    logging.info(f"Created output directory: {OUTPUT_DIR}")

logging.info(f"Reading 15m data from {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE, header=None, names=['startTime','open','high','low','close','volume','closeTime','quoteAssetVolume','trades','takerBaseVol','takerQuoteVol','ignore'])
df['startTime'] = pd.to_datetime(df['startTime'], unit='us')
# Select relevant columns for aggregation
df = df[['startTime','open','high','low','close','volume']]

logging.info("Aggregating to 1h K-line...")
df_1h = df.resample('1H', on='startTime').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna().reset_index().rename(columns={'startTime':'timestamp'})

logging.info(f"Saving 1h data to {OUTPUT_FILE}")
df_1h.to_csv(OUTPUT_FILE, index=False)
logging.info("Aggregation complete.") 