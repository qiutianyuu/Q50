import requests
import time
import json
import logging
import threading
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os

# ========== CONFIG ========== #
BINANCE_REST_URL = "https://api.binance.com/api/v3"
BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1"

# ========== LOGGING ========== #
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("BinanceRestClient")

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'binance_config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}")
        return {}

class BinanceRestClient:
    """
    Binance REST API客户端，用于获取实时数据
    API限额：1200次/分钟，5分钟拉一次（12次/小时）安全
    """
    
    def __init__(self, config=None):
        self.config = config or load_config()
        self.api_key = self.config.get('api_key', '')
        self.secret_key = self.config.get('secret_key', '')
        self.base_url = BINANCE_REST_URL
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RexKing-Trading-Bot/1.0',
            'Accept': 'application/json'
        })
        
        # 请求限制 - 保守设置
        self.request_count = 0
        self.last_reset = time.time()
        self.rate_limit = 1000  # 保守设置为1000次/分钟，留200次余量
        
        # 数据缓存 - 5分钟缓存
        self.cache = {}
        self.cache_timeout = 300  # 5分钟缓存

    def _check_rate_limit(self):
        """检查请求频率限制"""
        current_time = time.time()
        if current_time - self.last_reset >= 60:
            self.request_count = 0
            self.last_reset = current_time
        
        if self.request_count >= self.rate_limit:
            sleep_time = 60 - (current_time - self.last_reset)
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
            self.request_count = 0
            self.last_reset = time.time()

    def _request(self, endpoint: str, params: Dict = None, method: str = 'GET') -> Optional[Dict]:
        """发送API请求"""
        self._check_rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        try:
            if method == 'GET':
                response = self.session.get(url, params=params, timeout=10)
            else:
                response = self.session.post(url, json=params, timeout=10)
            
            self.request_count += 1
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None

    def get_ticker_24hr(self, symbol: str = 'ETHUSDT') -> Optional[Dict]:
        """获取24小时价格统计"""
        cache_key = f"ticker_{symbol}"
        if cache_key in self.cache and time.time() - self.cache[cache_key]['time'] < self.cache_timeout:
            return self.cache[cache_key]['data']
        
        params = {'symbol': symbol}
        data = self._request('ticker/24hr', params)
        
        if data:
            self.cache[cache_key] = {
                'data': data,
                'time': time.time()
            }
        
        return data

    def get_order_book(self, symbol: str = 'ETHUSDT', limit: int = 100) -> Optional[Dict]:
        """获取order book"""
        cache_key = f"orderbook_{symbol}_{limit}"
        if cache_key in self.cache and time.time() - self.cache[cache_key]['time'] < self.cache_timeout:
            return self.cache[cache_key]['data']
        
        params = {
            'symbol': symbol,
            'limit': limit
        }
        data = self._request('depth', params)
        
        if data:
            # 转换数据类型
            data['bids'] = [[float(price), float(qty)] for price, qty in data['bids']]
            data['asks'] = [[float(price), float(qty)] for price, qty in data['asks']]
            
            self.cache[cache_key] = {
                'data': data,
                'time': time.time()
            }
        
        return data

    def get_recent_trades(self, symbol: str = 'ETHUSDT', limit: int = 100) -> Optional[List[Dict]]:
        """获取最近交易"""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        return self._request('trades', params)

    def get_klines(self, symbol: str = 'ETHUSDT', interval: str = '5m', limit: int = 100) -> Optional[List[List]]:
        """获取K线数据 - 默认5分钟"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        return self._request('klines', params)

    def get_funding_rate(self, symbol: str = 'ETHUSDT') -> Optional[Dict]:
        """获取资金费率（期货）"""
        url = f"{BINANCE_FUTURES_URL}/premiumIndex"
        params = {'symbol': symbol}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Funding rate request error: {e}")
        
        return None

    def get_open_interest(self, symbol: str = 'ETHUSDT') -> Optional[Dict]:
        """获取持仓量（期货）"""
        url = f"{BINANCE_FUTURES_URL}/openInterest"
        params = {'symbol': symbol}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Open interest request error: {e}")
        
        return None

    def get_api_usage_stats(self) -> Dict:
        """获取API使用统计"""
        return {
            'requests_this_minute': self.request_count,
            'rate_limit': self.rate_limit,
            'time_since_reset': time.time() - self.last_reset
        }


class BinanceDataCollector:
    """
    数据收集器，5分钟拉一次数据（12次/小时）
    """
    
    def __init__(self, client: BinanceRestClient, symbol: str = 'ETHUSDT'):
        self.client = client
        self.symbol = symbol
        self.running = False
        self.thread = None
        self.data_buffer = {
            'trades': [],
            'klines': {},
            'order_book': None,
            'ticker': None,
            'last_update': None
        }
        self.callbacks = []
        self.collection_interval = 300  # 5分钟 = 300秒

    def start(self, interval: float = 300.0):
        """启动数据收集 - 默认5分钟间隔"""
        if self.running:
            return
        
        self.running = True
        self.collection_interval = interval
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.thread.start()
        logger.info(f"Data collector started for {self.symbol} (interval: {interval}s)")

    def stop(self):
        """停止数据收集"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Data collector stopped")

    def _collect_loop(self):
        """数据收集循环 - 5分钟一次"""
        while self.running:
            try:
                logger.info(f"Collecting data for {self.symbol}...")
                
                # 获取ticker数据
                ticker = self.client.get_ticker_24hr(self.symbol)
                if ticker:
                    self.data_buffer['ticker'] = ticker

                # 获取order book
                order_book = self.client.get_order_book(self.symbol, limit=100)
                if order_book:
                    self.data_buffer['order_book'] = order_book

                # 获取最近交易
                trades = self.client.get_recent_trades(self.symbol, limit=50)
                if trades:
                    self.data_buffer['trades'] = trades

                # 获取K线数据 - 5分钟、15分钟、1小时
                for interval in ['5m', '15m', '1h']:
                    klines = self.client.get_klines(self.symbol, interval, limit=100)
                    if klines:
                        self.data_buffer['klines'][interval] = klines

                # 更新最后更新时间
                self.data_buffer['last_update'] = datetime.now().isoformat()

                # 触发回调
                for callback in self.callbacks:
                    try:
                        callback(self.data_buffer.copy())
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

                # 打印API使用统计
                stats = self.client.get_api_usage_stats()
                logger.info(f"API usage: {stats['requests_this_minute']}/{stats['rate_limit']} requests this minute")

            except Exception as e:
                logger.error(f"Data collection error: {e}")

            # 等待5分钟
            time.sleep(self.collection_interval)

    def add_callback(self, callback):
        """添加数据回调"""
        self.callbacks.append(callback)

    def get_latest_data(self) -> Dict:
        """获取最新数据"""
        return self.data_buffer.copy()


class MarketDataProcessor:
    """
    市场数据处理器，计算实时特征
    """
    
    def __init__(self, data_collector: BinanceDataCollector):
        self.collector = data_collector
        self.price_history = []
        self.volume_history = []
        
        # 添加数据回调
        self.collector.add_callback(self._on_data_update)

    def _on_data_update(self, data: Dict):
        """数据更新回调"""
        if data.get('ticker'):
            ticker = data['ticker']
            current_price = float(ticker['lastPrice'])
            volume = float(ticker['volume'])
            
            self.price_history.append({
                'time': time.time(),
                'price': current_price,
                'volume': volume
            })
            
            # 保持最近100个价格点（约8小时的数据）
            if len(self.price_history) > 100:
                self.price_history.pop(0)

    def get_price_momentum(self, window: int = 20) -> Dict:
        """计算价格动量 - 基于最近20个数据点"""
        if len(self.price_history) < window:
            return {'momentum': 0, 'volatility': 0, 'current_price': 0}
        
        recent_prices = self.price_history[-window:]
        prices = [p['price'] for p in recent_prices]
        
        if len(prices) < 2:
            return {'momentum': 0, 'volatility': 0, 'current_price': prices[0] if prices else 0}
        
        # 计算动量
        momentum = (prices[-1] - prices[0]) / prices[0]
        
        # 计算波动率
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        volatility = (sum(r**2 for r in returns) / len(returns))**0.5 if returns else 0
        
        return {
            'momentum': momentum,
            'volatility': volatility,
            'current_price': prices[-1]
        }

    def get_volume_imbalance(self) -> Dict:
        """计算成交量不平衡"""
        data = self.collector.get_latest_data()
        order_book = data.get('order_book')
        
        if not order_book:
            return {'imbalance': 0, 'bid_volume': 0, 'ask_volume': 0}
        
        # 计算前10档的成交量
        bid_volume = sum(qty for _, qty in order_book['bids'][:10])
        ask_volume = sum(qty for _, qty in order_book['asks'][:10])
        
        total_volume = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        
        return {
            'imbalance': imbalance,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'total_volume': total_volume
        }

    def get_order_book_features(self) -> Dict:
        """获取order book特征"""
        data = self.collector.get_latest_data()
        order_book = data.get('order_book')
        
        if not order_book:
            return {'spread': 0, 'spread_bps': 0, 'depth': 0, 'pressure_ratio': 1, 'best_bid': 0, 'best_ask': 0}
        
        # 计算spread
        best_bid = order_book['bids'][0][0] if order_book['bids'] else 0
        best_ask = order_book['asks'][0][0] if order_book['asks'] else 0
        spread = best_ask - best_bid
        spread_bps = (spread / best_bid) * 10000 if best_bid > 0 else 0
        
        # 计算深度
        total_depth = sum(qty for _, qty in order_book['bids']) + sum(qty for _, qty in order_book['asks'])
        
        # 计算价格压力
        bid_pressure = sum(price * qty for price, qty in order_book['bids'])
        ask_pressure = sum(price * qty for price, qty in order_book['asks'])
        pressure_ratio = bid_pressure / ask_pressure if ask_pressure > 0 else 1
        
        return {
            'spread': spread,
            'spread_bps': spread_bps,
            'depth': total_depth,
            'pressure_ratio': pressure_ratio,
            'best_bid': best_bid,
            'best_ask': best_ask
        }

    def get_market_sentiment(self) -> Dict:
        """获取市场情绪指标"""
        data = self.collector.get_latest_data()
        ticker = data.get('ticker')
        
        if not ticker:
            return {'sentiment': 0, 'price_change': 0, 'volume_change': 0}
        
        # 计算价格变化
        price_change = float(ticker['priceChangePercent'])
        
        # 计算成交量变化（如果有历史数据）
        volume_change = 0
        if len(self.volume_history) > 1:
            current_volume = float(ticker['volume'])
            prev_volume = self.volume_history[-1]
            volume_change = (current_volume - prev_volume) / prev_volume if prev_volume > 0 else 0
        
        # 简单的情绪指标
        sentiment = 0
        if price_change > 1:  # 价格上涨超过1%
            sentiment += 1
        elif price_change < -1:  # 价格下跌超过1%
            sentiment -= 1
        
        if volume_change > 0.5:  # 成交量增加超过50%
            sentiment += 0.5
        elif volume_change < -0.5:  # 成交量减少超过50%
            sentiment -= 0.5
        
        return {
            'sentiment': sentiment,
            'price_change': price_change,
            'volume_change': volume_change
        }

    def get_kline_features(self, interval: str = '5m') -> Dict:
        """获取K线特征"""
        data = self.collector.get_latest_data()
        klines = data.get('klines', {}).get(interval, [])
        
        if not klines or len(klines) < 20:
            return {'rsi': 50, 'macd': 0, 'volume_ma': 0}
        
        # 计算RSI
        closes = [float(k[4]) for k in klines]  # 收盘价
        gains = [max(0, closes[i] - closes[i-1]) for i in range(1, len(closes))]
        losses = [max(0, closes[i-1] - closes[i]) for i in range(1, len(closes))]
        
        avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0
        avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0
        
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        # 计算MACD
        if len(closes) >= 26:
            ema12 = sum(closes[-12:]) / 12
            ema26 = sum(closes[-26:]) / 26
            macd = ema12 - ema26
        else:
            macd = 0
        
        # 计算成交量均值
        volumes = [float(k[5]) for k in klines]  # 成交量
        volume_ma = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else 0
        
        return {
            'rsi': rsi,
            'macd': macd,
            'volume_ma': volume_ma,
            'current_close': closes[-1] if closes else 0
        }


# ========== 使用示例 ========== #
if __name__ == "__main__":
    # 创建客户端
    client = BinanceRestClient()
    collector = BinanceDataCollector(client, 'ETHUSDT')
    processor = MarketDataProcessor(collector)
    
    # 启动数据收集 - 5分钟间隔
    collector.start(interval=300.0)  # 5分钟 = 300秒
    
    try:
        # 主循环
        while True:
            # 获取实时特征
            momentum = processor.get_price_momentum()
            imbalance = processor.get_volume_imbalance()
            ob_features = processor.get_order_book_features()
            sentiment = processor.get_market_sentiment()
            kline_features = processor.get_kline_features('5m')
            
            print(f"\n=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            print(f"Price: {momentum['current_price']:.2f}")
            print(f"Momentum: {momentum['momentum']:.4f}")
            print(f"Volatility: {momentum['volatility']:.4f}")
            print(f"Volume Imbalance: {imbalance['imbalance']:.4f}")
            print(f"Spread: {ob_features['spread_bps']:.2f} bps")
            print(f"Sentiment: {sentiment['sentiment']:.2f}")
            print(f"RSI: {kline_features['rsi']:.1f}")
            print(f"MACD: {kline_features['macd']:.4f}")
            print(f"Volume MA: {kline_features['volume_ma']:.2f}")
            print("-" * 50)
            
            # 等待5分钟
            time.sleep(300)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        collector.stop() 