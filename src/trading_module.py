import logging
import numpy as np
from datetime import datetime, timedelta
from src.api_module import BybitWebSocket, TAAPIIoClient, EtherscanClient
from src.indicators_module import *
from src.xgboost_module import XGBoostManager, build_feature_vector

logger = logging.getLogger("SteadyBullTrading")

# ========== 交易信号主逻辑 ========== #
class SteadyBullTrader:
    def __init__(self, api_clients, xgb_manager, shap_manager, config):
        self.bybit_ws = api_clients['bybit']
        self.taapi = api_clients['taapi']
        self.etherscan = api_clients['etherscan']
        self.xgb = xgb_manager
        self.shap = shap_manager
        self.config = config
        self.last_trade_time = None
        self.pause_until = None
        self.position = None
        self.trade_log = []

    def fetch_indicators(self, symbol="ETH/USDT"):
        # 技术面
        rsi = self.taapi.get_rsi(symbol)["value"]
        macd_data = self.taapi.get_macd(symbol)
        macd, signal = macd_data["valueMACD"], macd_data["valueMACDSignal"]
        kdj_data = self.taapi.get_kdj(symbol)
        k, d, j = kdj_data["valueK"], kdj_data["valueD"], kdj_data["valueJ"]
        vwap = self.taapi.get_vwap(symbol)["value"]
        atr = self.taapi.get_atr(symbol)["value"]
        pattern = self.taapi.get_pattern(symbol)["result"]
        # 交易量、支撑阻力等可用Bybit数据/本地计算
        # ...
        # 链上、基本面、情绪面等略（见API模块）
        # ...
        # 返回所有指标分数dict
        indicators = {
            "T1_RSI": score_rsi(rsi),
            "T2_MACD": score_macd(macd, signal),
            "T7_KDJ": score_kdj(k, d, j),
            "T8_VWAP": score_vwap(1, vwap),  # price需补全
            "T9_Pattern": score_kline_pattern(pattern),
            # ... 其余指标
        }
        return indicators, atr

    def should_trade(self, indicators, shap_weights, xgb_prob):
        # 总分加权
        total_score = sum([indicators[k] * shap_weights.get(k, 0.05) for k in indicators])
        bullish_count = sum([1 for v in indicators.values() if v > 50])
        if total_score >= 70 and bullish_count >= 13 and xgb_prob > 0.8:
            return True
        return False

    def check_pause(self, m3, m4, m5):
        # 黑天鹅暂停逻辑
        if m3 < 25 or m4 < 25 or m5 < 25:
            self.pause_until = datetime.utcnow() + timedelta(hours=1)
            logger.warning("Black swan detected, trading paused for 1 hour.")
            return True
        return False

    def trade(self, symbol="ETH/USDT"):
        if self.pause_until and datetime.utcnow() < self.pause_until:
            logger.info("In pause window, skip trading.")
            return None
        indicators, atr = self.fetch_indicators(symbol)
        shap_weights = self.shap.get_shap_weights()
        feature_vec = build_feature_vector(indicators, shap_weights)
        xgb_prob = self.xgb.predict(feature_vec)[0]
        # 多周期确认（15m+1h）
        # ...（略，需补全）
        # 黑天鹅信号
        m3, m4, m5 = indicators.get("M3_BlackSwan", 50), indicators.get("M4_Volatility", 50), indicators.get("M5_Sudden", 50)
        if self.check_pause(m3, m4, m5):
            return None
        if self.should_trade(indicators, shap_weights, xgb_prob):
            # 仓位、止盈止损
            entry_price = 1  # 需补全实时价格
            stop_loss = entry_price - atr
            take_profit = entry_price + entry_price * 0.018  # 1.8%目标
            self.position = {
                "entry": entry_price,
                "stop": stop_loss,
                "tp": take_profit,
                "size": 0.01,
                "time": datetime.utcnow()
            }
            logger.info(f"Trade opened: {self.position}")
            self.trade_log.append(self.position)
            return self.position
        else:
            logger.info("No trade signal this round.")
            return None

logger.info("[Trading Module] SteadyBullTrader loaded (first 100 lines)") 