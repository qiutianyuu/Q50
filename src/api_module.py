import websocket
import requests
import threading
import time
import json
import logging
from queue import Queue

# ========== CONFIG ========== #
BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/spot"
TAAPI_IO_URL = "https://api.taapi.io"
ETHERSCAN_URL = "https://api.etherscan.io/api"
INTOBLOCK_WS_URL = "wss://api.intotheblock.com/ws"

BYBIT_API_KEY = "YOUR_BYBIT_API_KEY"
TAAPI_IO_KEY = "YOUR_TAAPI_IO_KEY"
ETHERSCAN_KEY = "YOUR_ETHERSCAN_KEY"
INTOBLOCK_KEY = "YOUR_INTOTHEBLOCK_KEY"

# ========== Alpha Vantage 宏观API ========== #
ALPHA_VANTAGE_KEY = "2GVTWPCHJYMBYWZP"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

# ========== LOGGING ========== #
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("SteadyBullAPI")

# ========== BYBIT WEBSOCKET ========== #
class BybitWebSocket:
    def __init__(self, symbols=["ETHUSDT"], kline_interval="1h"):
        self.url = BYBIT_WS_URL
        self.ws = None
        self.symbols = symbols
        self.kline_interval = kline_interval
        self.data_queue = Queue()
        self.connected = False
        self._stop = threading.Event()
        self._thread = None
        self._ping_interval = 20
        self._last_ping = time.time()
        self._subs = []

    def _on_open(self, ws):
        logger.info("Bybit WebSocket opened.")
        self.connected = True
        # Subscribe to K-line, trades, ticker
        for symbol in self.symbols:
            subs = [
                {"op": "subscribe", "args": [f"kline.{self.kline_interval}.{symbol}"]},
                {"op": "subscribe", "args": [f"publicTrade.{symbol}"]},
                {"op": "subscribe", "args": [f"tickers.{symbol}"]}
            ]
            for sub in subs:
                ws.send(json.dumps(sub))
                self._subs.append(sub)
        logger.info(f"Subscribed to {len(self._subs)} Bybit channels.")

    def _on_message(self, ws, message):
        try:
            msg = json.loads(message)
            if 'topic' in msg:
                self.data_queue.put(msg)
        except Exception as e:
            logger.error(f"Bybit message error: {e}")

    def _on_error(self, ws, error):
        logger.error(f"Bybit WebSocket error: {error}")
        self.connected = False

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"Bybit WebSocket closed: {close_status_code} {close_msg}")
        self.connected = False

    def _run(self):
        while not self._stop.is_set():
            try:
                self.ws = websocket.WebSocketApp(
                    self.url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                self.ws.run_forever(ping_interval=self._ping_interval)
            except Exception as e:
                logger.error(f"Bybit WS run error: {e}")
            time.sleep(5)  # Reconnect delay

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Bybit WebSocket thread started.")

    def stop(self):
        self._stop.set()
        if self.ws:
            self.ws.close()
        logger.info("Bybit WebSocket stopped.")

    def get_data(self, timeout=1):
        try:
            return self.data_queue.get(timeout=timeout)
        except:
            return None

# ========== TAAPI.IO REST API ========== #
class TAAPIIoClient:
    def __init__(self, api_key=TAAPI_IO_KEY):
        self.api_key = api_key
        self.base_url = TAAPI_IO_URL
        self.rate_limit = 500  # free tier per day
        self.calls_today = 0
        self.last_reset = time.strftime('%Y-%m-%d')

    def _check_reset(self):
        today = time.strftime('%Y-%m-%d')
        if today != self.last_reset:
            self.calls_today = 0
            self.last_reset = today

    def _request(self, endpoint, params):
        self._check_reset()
        if self.calls_today >= self.rate_limit:
            logger.warning("TAAPI.IO daily rate limit reached!")
            return None
        params['secret'] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        try:
            resp = requests.get(url, params=params, timeout=5)
            self.calls_today += 1
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.error(f"TAAPI.IO error {resp.status_code}: {resp.text}")
                return None
        except Exception as e:
            logger.error(f"TAAPI.IO request error: {e}")
            return None

    def get_rsi(self, symbol="ETH/USDT", interval="1h"):
        params = {"exchange": "binance", "symbol": symbol, "interval": interval, "indicator": "rsi"}
        return self._request("rsi", params)

    def get_macd(self, symbol="ETH/USDT", interval="1h"):
        params = {"exchange": "binance", "symbol": symbol, "interval": interval, "indicator": "macd"}
        return self._request("macd", params)

    def get_kdj(self, symbol="ETH/USDT", interval="1h"):
        params = {"exchange": "binance", "symbol": symbol, "interval": interval, "indicator": "kdj"}
        return self._request("kdj", params)

    def get_vwap(self, symbol="ETH/USDT", interval="1h"):
        params = {"exchange": "binance", "symbol": symbol, "interval": interval, "indicator": "vwap"}
        return self._request("vwap", params)

    def get_atr(self, symbol="ETH/USDT", interval="1h"):
        params = {"exchange": "binance", "symbol": symbol, "interval": interval, "indicator": "atr"}
        return self._request("atr", params)

    def get_pattern(self, symbol="ETH/USDT", interval="1h"):
        params = {"exchange": "binance", "symbol": symbol, "interval": interval, "indicator": "patterns"}
        return self._request("patterns", params)

# Verbose progress log
logger.info("[API Module] BybitWebSocket and TAAPIIoClient classes loaded (first 100 lines)")

# ========== ETHERSCAN REST API ========== #
class EtherscanClient:
    def __init__(self, api_key=ETHERSCAN_KEY):
        self.api_key = api_key
        self.base_url = ETHERSCAN_URL

    def get_internal_tx(self, address, startblock=0, endblock=99999999):
        params = {
            "module": "account",
            "action": "txlistinternal",
            "address": address,
            "startblock": startblock,
            "endblock": endblock,
            "sort": "desc",
            "apikey": self.api_key
        }
        try:
            resp = requests.get(self.base_url, params=params, timeout=5)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.error(f"Etherscan error {resp.status_code}: {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Etherscan request error: {e}")
            return None

    def get_gas_tracker(self):
        params = {"module": "gastracker", "action": "gasoracle", "apikey": self.api_key}
        try:
            resp = requests.get(self.base_url, params=params, timeout=5)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.error(f"Etherscan error {resp.status_code}: {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Etherscan request error: {e}")
            return None

# Verbose progress log
logger.info("[API Module] EtherscanClient loaded (line 150)")

def get_dxy_change():
    """获取DXY美元指数当日涨跌幅（百分比）"""
    params = {
        "function": "FX_DAILY",
        "from_symbol": "USD",
        "to_symbol": "DX-Y.NYB",
        "apikey": ALPHA_VANTAGE_KEY
    }
    try:
        resp = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=10)
        data = resp.json().get("Time Series FX (Daily)", {})
        dates = sorted(data.keys(), reverse=True)
        if len(dates) < 2:
            return 0
        today, prev = dates[0], dates[1]
        today_close = float(data[today]["4. close"])
        prev_close = float(data[prev]["4. close"])
        pct = 100 * (today_close - prev_close) / prev_close
        return pct
    except Exception as e:
        logger.warning(f"Alpha Vantage DXY error: {e}")
        return 0

def get_sp500_change():
    """获取SP500当日涨跌幅（百分比）"""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": "^GSPC",
        "apikey": ALPHA_VANTAGE_KEY
    }
    try:
        resp = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=10)
        data = resp.json().get("Time Series (Daily)", {})
        dates = sorted(data.keys(), reverse=True)
        if len(dates) < 2:
            return 0
        today, prev = dates[0], dates[1]
        today_close = float(data[today]["4. close"])
        prev_close = float(data[prev]["4. close"])
        pct = 100 * (today_close - prev_close) / prev_close
        return pct
    except Exception as e:
        logger.warning(f"Alpha Vantage SP500 error: {e}")
        return 0

logger.info("[API Module] Alpha Vantage DXY/SP500 loaded (line 250)") 