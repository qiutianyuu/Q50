import numpy as np
import pandas as pd
import logging
from statsmodels.stats.outliers_influence import variance_inflation_factor
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from src.api_module import get_dxy_change, get_sp500_change

logger = logging.getLogger("SteadyBullIndicators")

# ========== 技术面指标 ========== #
def score_rsi(rsi):
    """T1. RSI评分：1h，>70=20，<30=80，线性插值"""
    if rsi >= 70:
        return 20
    elif rsi <= 30:
        return 80
    else:
        # 30-70线性插值
        return 50 + (rsi - 50) / 20 * 30

def score_macd(macd, signal):
    """T2. MACD评分：MACD>signal=80，<signal=20，接近交叉=50"""
    if macd > signal:
        return 80
    elif macd < signal:
        return 20
    else:
        return 50

def score_volume_trend(vol_7d, vol_1d):
    """T3. 交易量趋势评分：7天均量 vs 1天均量，上升=80，下降=20，平稳=50"""
    if vol_1d > vol_7d * 1.05:
        return 80
    elif vol_1d < vol_7d * 0.95:
        return 20
    else:
        return 50

def score_support_strength(price, support):
    """T5. 支撑位强度评分：距2400USD，>5%=80，<5%或跌破=20"""
    pct = (price - support) / support * 100
    if pct > 5:
        return 80
    elif pct < 0:
        return 20
    else:
        return 50

def score_resistance_pressure(price, resistance):
    """T6. 阻力位压力评分：距2700USD，<5%或突破=80，>5%=20"""
    pct = (resistance - price) / resistance * 100
    if pct < 5:
        return 80
    elif pct > 5:
        return 20
    else:
        return 50

def score_kdj(k, d, j):
    """T7. KDJ评分：金叉=80，死叉=20，无交叉=50"""
    if k > d and d > j:
        return 80
    elif k < d and d < j:
        return 20
    else:
        return 50

def score_vwap(price, vwap):
    """T8. VWAP评分：价格>VWAP=80，<VWAP=20，接近=50"""
    if price > vwap * 1.01:
        return 80
    elif price < vwap * 0.99:
        return 20
    else:
        return 50

def score_kline_pattern(pattern):
    """T9. K线形态评分：锤头/看涨吞没=80，吊颈/看跌吞没=20，无形态=50"""
    bullish = ["hammer", "bullish_engulfing"]
    bearish = ["hanging_man", "bearish_engulfing"]
    if pattern in bullish:
        return 80
    elif pattern in bearish:
        return 20
    else:
        return 50

logger.info("[Indicators Module] Technical indicators loaded (first 50 lines)")

# ========== Etherscan链上指标 ========== #
ETHERSCAN_KEY = "CM56ZD9J9KTV8K93U8EXP8P4E1CBIEJ1P5"
ETHERSCAN_URL = "https://api.etherscan.io/api"
EXCHANGE_ADDRS = [
    # 主要交易所ETH热钱包（示例，建议补全）
    "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",  # Bitfinex
    "0x53d284357ec70cE289D6D64134DfAc8E511c8a3D",  # Bitstamp
    # ...
]

def get_eth_netflow_etherscan(day="2024-06-01"):
    """统计所有交易所地址的ETH净流入/流出（单位ETH，正为流入，负为流出）"""
    netflow = 0
    for addr in EXCHANGE_ADDRS:
        params = {
            "module": "account",
            "action": "txlist",
            "address": addr,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "desc",
            "apikey": ETHERSCAN_KEY
        }
        try:
            resp = requests.get(ETHERSCAN_URL, params=params, timeout=10)
            txs = resp.json().get("result", [])
            for tx in txs:
                # 只统计当天
                if not tx["timeStamp"].startswith(day.replace("-", "")):
                    continue
                value = int(tx["value"]) / 1e18
                if tx["to"].lower() == addr.lower():
                    netflow += value
                elif tx["from"].lower() == addr.lower():
                    netflow -= value
        except Exception as e:
            logger.warning(f"Etherscan netflow error: {e}")
    return netflow

def score_exchange_netflow_etherscan(netflow):
    """W1. 交易所净流入评分：净流出=80，净流入>5000=20，接近0=50"""
    if netflow < -1000:
        return 80
    elif netflow > 5000:
        return 20
    else:
        return 50

def get_whale_tx_etherscan(day="2024-06-01", min_eth=1000):
    """统计所有交易所地址当天鲸鱼转账（>1000 ETH）笔数"""
    whale_count = 0
    for addr in EXCHANGE_ADDRS:
        params = {
            "module": "account",
            "action": "txlist",
            "address": addr,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "desc",
            "apikey": ETHERSCAN_KEY
        }
        try:
            resp = requests.get(ETHERSCAN_URL, params=params, timeout=10)
            txs = resp.json().get("result", [])
            for tx in txs:
                if not tx["timeStamp"].startswith(day.replace("-", "")):
                    continue
                value = int(tx["value"]) / 1e18
                if value >= min_eth:
                    whale_count += 1
        except Exception as e:
            logger.warning(f"Etherscan whale tx error: {e}")
    return whale_count

def score_whale_tx_freq_etherscan(whale_txs):
    """W2. 鲸鱼交易频率评分：>50=80，<20=20，20-50=50"""
    if whale_txs > 50:
        return 80
    elif whale_txs < 20:
        return 20
    else:
        return 50

logger.info("[Indicators Module] Etherscan W1/W2 loaded (line 150)")

# ========== CoinGecko 基本面/情绪/黑天鹅 ========== #
COINGECKO_KEY = "CG-YohMPqAh6YHhifSq3NmATyHR"
COINGECKO_URL = "https://pro-api.coingecko.com/api/v3"

def get_fund_flow_coingecko():
    """F2. ETF资金流：近30天ETH资金流（USD）"""
    try:
        url = f"{COINGECKO_URL}/coins/ethereum/market_chart"
        params = {"vs_currency": "usd", "days": 30, "x_cg_pro_api_key": COINGECKO_KEY}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        prices = [p[1] for p in data.get("prices", [])]
        if len(prices) < 2:
            return 0
        flow = prices[-1] - prices[0]
        return flow * 1e6  # 粗略估算，实际应用ETF流API
    except Exception as e:
        logger.warning(f"CoinGecko fund flow error: {e}")
        return 0

def score_etf_fund_flow_coingecko(fund_flow):
    """F2. ETF资金流评分：净流入>10亿=80，净流出=20，接近0=50"""
    if fund_flow > 1e9:
        return 80
    elif fund_flow < 0:
        return 20
    else:
        return 50

def get_sentiment_coingecko():
    """S1. CoinGecko情绪：bullish/bearish ratio"""
    try:
        url = f"{COINGECKO_URL}/coins/ethereum"
        params = {"x_cg_pro_api_key": COINGECKO_KEY, "localization": "false"}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        bullish = data.get("sentiment_votes_up_percentage", 50)
        bearish = data.get("sentiment_votes_down_percentage", 50)
        ratio = bullish / max(bearish, 1)
        return ratio
    except Exception as e:
        logger.warning(f"CoinGecko sentiment error: {e}")
        return 1

def score_x_sentiment_coingecko(ratio):
    """S1. X情绪评分：多空>2:1=80，<1:2=20，1:1=50"""
    if ratio > 2:
        return 80
    elif ratio < 0.5:
        return 20
    else:
        return 50

def get_black_swan_coingecko():
    """M3. 黑天鹅情绪：负面新闻占比（近30天ETH新闻）"""
    try:
        url = f"{COINGECKO_URL}/coins/ethereum/status_updates"
        params = {"x_cg_pro_api_key": COINGECKO_KEY, "per_page": 100}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        news = data.get("status_updates", [])
        nltk.download('vader_lexicon', quiet=True)
        sid = SentimentIntensityAnalyzer()
        neg_count = 0
        for n in news:
            text = n.get("description", "")
            if text and sid.polarity_scores(text)["compound"] < -0.3:
                neg_count += 1
        pct = 100 * neg_count / max(len(news), 1)
        return pct
    except Exception as e:
        logger.warning(f"CoinGecko black swan error: {e}")
        return 50

def score_black_swan_sentiment_coingecko(neg_pct):
    """M3. 黑天鹅情绪评分：负面<10%=80，负面>30%=20，其他=50"""
    if neg_pct < 10:
        return 80
    elif neg_pct > 30:
        return 20
    else:
        return 50

def get_sudden_event_coingecko(keywords=["hack", "regulation", "SEC", "exploit"]):
    """M5. 突发事件：近30天ETH新闻含关键词占比"""
    try:
        url = f"{COINGECKO_URL}/coins/ethereum/status_updates"
        params = {"x_cg_pro_api_key": COINGECKO_KEY, "per_page": 100}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        news = data.get("status_updates", [])
        count = 0
        for n in news:
            text = n.get("description", "").lower()
            if any(kw in text for kw in keywords):
                count += 1
        pct = 100 * count / max(len(news), 1)
        return pct
    except Exception as e:
        logger.warning(f"CoinGecko sudden event error: {e}")
        return 50

def score_sudden_event_coingecko(pct):
    """M5. 突发事件评分：关键词<5%=80，>20%=20，5-20%=50"""
    if pct < 5:
        return 80
    elif pct > 20:
        return 20
    else:
        return 50

logger.info("[Indicators Module] CoinGecko F2/S1/M3/M5 loaded (line 200)")

# ========== 基本面指标 ========== #
def score_etf_fund_flow(fund_flow):
    """F2. ETF资金流评分：净流入>10亿=80，净流出=20，接近0=50"""
    if fund_flow > 1e9:
        return 80
    elif fund_flow < 0:
        return 20
    else:
        return 50

def get_macro_scores():
    """获取DXY和SP500当日涨跌幅（百分比）"""
    dxy_chg = get_dxy_change()
    sp500_chg = get_sp500_change()
    logger.info(f"[Macro] DXY change: {dxy_chg:.2f}%, SP500 change: {sp500_chg:.2f}%")
    return dxy_chg, sp500_chg

def score_macro_economy_alpha_vantage(dxy_chg, sp500_chg):
    """M2. 宏观经济评分：DXY跌+SP500涨=80，DXY涨+SP500跌=20，混合=50"""
    if dxy_chg < 0 and sp500_chg > 0:
        return 80
    elif dxy_chg > 0 and sp500_chg < 0:
        return 20
    else:
        return 50

logger.info("[Indicators Module] Fundamental indicators loaded (line 120)")

# ========== 情绪面指标 ========== #
def score_x_sentiment(bull_bear_ratio):
    """S1. X情绪评分：多空>2:1=80，<1:2=20，1:1=50"""
    if bull_bear_ratio > 2:
        return 80
    elif bull_bear_ratio < 0.5:
        return 20
    else:
        return 50

def score_black_swan_sentiment(neg_pct, anomaly):
    """M3. 黑天鹅情绪评分：负面<10%+无异常=80，负面>30%+异常=20，其他=50"""
    if neg_pct < 10 and not anomaly:
        return 80
    elif neg_pct > 30 and anomaly:
        return 20
    else:
        return 50

def score_volatility_index(vix, eth_vol):
    """M4. 波动指数评分：VIX<20+ETH 1h波动<2%=80，VIX>30+波动>5%=20，其他=50"""
    if vix < 20 and eth_vol < 2:
        return 80
    elif vix > 30 and eth_vol > 5:
        return 20
    else:
        return 50

def score_sudden_event(reg_keywords_pct):
    """M5. 突发事件评分：关键词<5%=80，>20%=20，5-20%=50"""
    if reg_keywords_pct < 5:
        return 80
    elif reg_keywords_pct > 20:
        return 20
    else:
        return 50

logger.info("[Indicators Module] Fundamental & sentiment indicators loaded (line 150)")

# ========== VIF 共线性检查 ========== #
def vif_filter(df, thresh=10):
    """剔除高共线性指标，返回保留列名列表"""
    cols = list(df.columns)
    dropped = True
    while dropped and len(cols) > 1:
        dropped = False
        vif = [variance_inflation_factor(df[cols].values, i) for i in range(len(cols))]
        max_vif = max(vif)
        if max_vif > thresh:
            drop_idx = vif.index(max_vif)
            logger.info(f"[VIF] Drop {cols[drop_idx]} (VIF={max_vif:.2f})")
            cols.pop(drop_idx)
            dropped = True
    return cols

# ========== SHAP 权重管理（占位） ========== #
class SHAPManager:
    def __init__(self, shap_dict=None):
        self.shap_dict = shap_dict or {}
    def get_weight(self, indicator):
        return self.shap_dict.get(indicator, 0.05)  # 默认5%
    def update(self, new_shap):
        self.shap_dict = new_shap

logger.info("[Indicators Module] VIF & SHAP manager loaded (line 200)")

# ========== Alpha Vantage 宏观指标 ========== #
def get_macro_scores():
    """获取DXY和SP500当日涨跌幅（百分比）"""
    dxy_chg = get_dxy_change()
    sp500_chg = get_sp500_change()
    logger.info(f"[Macro] DXY change: {dxy_chg:.2f}%, SP500 change: {sp500_chg:.2f}%")
    return dxy_chg, sp500_chg

def score_macro_economy_alpha_vantage(dxy_chg, sp500_chg):
    """M2. 宏观经济评分：DXY跌+SP500涨=80，DXY涨+SP500跌=20，混合=50"""
    if dxy_chg < 0 and sp500_chg > 0:
        return 80
    elif dxy_chg > 0 and sp500_chg < 0:
        return 20
    else:
        return 50

logger.info("[Indicators Module] Alpha Vantage M2 loaded (line 250)") 