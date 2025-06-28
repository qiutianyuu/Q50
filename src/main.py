import logging
import time
from src.api_module import BybitWebSocket, TAAPIIoClient, EtherscanClient
from src.indicators_module import SHAPManager
from src.xgboost_module import XGBoostManager
from src.trading_module import SteadyBullTrader
from src.backtest_module import run_backtest

logger = logging.getLogger("SteadyBullMain")

# ========== 配置 ========== #
CONFIG = {
    "symbol": "ETH/USDT",
    "mode": "backtest",
    "datafile": "data/ETHUSDT-15m-2025-04.csv",  # 指向你的15m数据
    "interval": 900,  # 15分钟=900秒
}

# ========== 模块初始化 ========== #
def init_modules():
    api_clients = {
        "bybit": BybitWebSocket([CONFIG["symbol"]]),
        "taapi": TAAPIIoClient(),
        "etherscan": EtherscanClient(),
    }
    xgb_manager = XGBoostManager()
    shap_manager = SHAPManager()
    return api_clients, xgb_manager, shap_manager

# ========== 主循环 ========== #
def main():
    api_clients, xgb_manager, shap_manager = init_modules()
    if CONFIG["mode"] == "backtest":
        logger.info("Running backtest mode...")
        run_backtest(CONFIG["datafile"], xgb_manager, shap_manager, CONFIG)
        return
    trader = SteadyBullTrader(api_clients, xgb_manager, shap_manager, CONFIG)
    while True:
        trader.trade(CONFIG["symbol"])
        time.sleep(CONFIG["interval"])

if __name__ == "__main__":
    main() 