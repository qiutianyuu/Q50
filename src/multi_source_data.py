import requests
import time
import json
import logging
import threading
from typing import Dict, List, Optional
from datetime import datetime
import os

# ========== LOGGING ========== #
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("MultiSourceData")

class CoinGeckoClient:
    """CoinGecko API客户端 - 免费，限制少"""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RexKing-Trading-Bot/1.0'
        })
    
    def get_eth_price(self) -> Optional[Dict]:
        """获取ETH价格数据"""
        try:
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': 'ethereum',
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true',
                'include_market_cap': 'true'
            }
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    'price': data['ethereum']['usd'],
                    'price_change_24h': data['ethereum']['usd_24h_change'],
                    'volume_24h': data['ethereum']['usd_24h_vol'],
                    'market_cap': data['ethereum']['usd_market_cap']
                }
        except Exception as e:
            logger.error(f"CoinGecko API error: {e}")
        return None
    
    def get_eth_market_data(self) -> Optional[Dict]:
        """获取ETH市场数据"""
        try:
            url = f"{self.base_url}/coins/ethereum/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': '1',
                'interval': 'hourly'
            }
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    'prices': data['prices'],
                    'volumes': data['total_volumes'],
                    'market_caps': data['market_caps']
                }
        except Exception as e:
            logger.error(f"CoinGecko market data error: {e}")
        return None

class OKXClient:
    """OKX API客户端 - 通常限制较少"""
    
    def __init__(self):
        self.base_url = "https://www.okx.com/api/v5"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RexKing-Trading-Bot/1.0'
        })
    
    def get_ticker(self, symbol="ETH-USDT") -> Optional[Dict]:
        """获取ticker数据"""
        try:
            url = f"{self.base_url}/market/ticker"
            params = {'instId': symbol}
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '0' and data['data']:
                    ticker = data['data'][0]
                    return {
                        'price': float(ticker['last']),
                        'bid': float(ticker['bidPx']) if ticker['bidPx'] else 0,
                        'ask': float(ticker['askPx']) if ticker['askPx'] else 0,
                        'volume_24h': float(ticker['vol24h']) if ticker['vol24h'] else 0,
                        'price_change_24h': float(ticker.get('change24h', 0)) if ticker.get('change24h') else 0
                    }
        except Exception as e:
            logger.error(f"OKX API error: {e}")
        return None
    
    def get_order_book(self, symbol="ETH-USDT", depth=20) -> Optional[Dict]:
        """获取order book"""
        try:
            url = f"{self.base_url}/market/books"
            params = {'instId': symbol, 'sz': depth}
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '0' and data['data']:
                    book_data = data['data'][0]
                    bids = []
                    asks = []
                    
                    # 处理bids
                    for bid in book_data['bids']:
                        if len(bid) >= 2:
                            bids.append([float(bid[0]), float(bid[1])])
                    
                    # 处理asks
                    for ask in book_data['asks']:
                        if len(ask) >= 2:
                            asks.append([float(ask[0]), float(ask[1])])
                    
                    return {
                        'bids': bids,
                        'asks': asks
                    }
        except Exception as e:
            logger.error(f"OKX order book error: {e}")
        return None

class AlphaVantageClient:
    """Alpha Vantage客户端 - 你已经有了密钥"""
    
    def __init__(self, api_key="2GVTWPCHJYMBYWZP"):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session = requests.Session()
    
    def get_crypto_price(self, symbol="ETH", market="USD") -> Optional[Dict]:
        """获取加密货币价格"""
        try:
            params = {
                'function': 'CURRENCY_EXCHANGE_RATE',
                'from_currency': symbol,
                'to_currency': market,
                'apikey': self.api_key
            }
            response = self.session.get(self.base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'Realtime Currency Exchange Rate' in data:
                    rate = data['Realtime Currency Exchange Rate']
                    return {
                        'price': float(rate['5. Exchange Rate']),
                        'last_updated': rate['6. Last Refreshed']
                    }
        except Exception as e:
            logger.error(f"Alpha Vantage API error: {e}")
        return None

class MultiSourceDataCollector:
    """多数据源数据收集器"""
    
    def __init__(self):
        self.coingecko = CoinGeckoClient()
        self.okx = OKXClient()
        self.alphavantage = AlphaVantageClient()
        
        self.data_buffer = {
            'price': None,
            'order_book': None,
            'market_data': None,
            'last_update': None,
            'source': None
        }
        
        self.running = False
        self.thread = None
        self.callbacks = []
    
    def start(self, interval=300):
        """启动数据收集 - 5分钟间隔"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collect_loop, args=(interval,), daemon=True)
        self.thread.start()
        logger.info(f"Multi-source data collector started (interval: {interval}s)")
    
    def stop(self):
        """停止数据收集"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Multi-source data collector stopped")
    
    def _collect_loop(self, interval):
        """数据收集循环"""
        while self.running:
            try:
                logger.info("Collecting data from multiple sources...")
                
                # 尝试从不同数据源获取数据
                data = self._get_best_available_data()
                
                if data:
                    self.data_buffer.update(data)
                    self.data_buffer['last_update'] = datetime.now().isoformat()
                    
                    # 触发回调
                    for callback in self.callbacks:
                        try:
                            callback(self.data_buffer.copy())
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                    
                    logger.info(f"Data collected from {data['source']}")
                else:
                    logger.warning("No data source available")
                
            except Exception as e:
                logger.error(f"Data collection error: {e}")
            
            time.sleep(interval)
    
    def _get_best_available_data(self) -> Optional[Dict]:
        """获取最佳可用数据"""
        # 优先级：OKX > CoinGecko > Alpha Vantage
        
        # 尝试OKX
        try:
            ticker = self.okx.get_ticker()
            order_book = self.okx.get_order_book()
            if ticker:
                return {
                    'price': ticker['price'],
                    'order_book': order_book,
                    'source': 'OKX',
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'volume_24h': ticker['volume_24h'],
                    'price_change_24h': ticker['price_change_24h']
                }
        except Exception as e:
            logger.debug(f"OKX failed: {e}")
        
        # 尝试CoinGecko
        try:
            price_data = self.coingecko.get_eth_price()
            market_data = self.coingecko.get_eth_market_data()
            if price_data:
                return {
                    'price': price_data['price'],
                    'market_data': market_data,
                    'source': 'CoinGecko',
                    'price_change_24h': price_data['price_change_24h'],
                    'volume_24h': price_data['volume_24h'],
                    'market_cap': price_data['market_cap']
                }
        except Exception as e:
            logger.debug(f"CoinGecko failed: {e}")
        
        # 尝试Alpha Vantage
        try:
            price_data = self.alphavantage.get_crypto_price()
            if price_data:
                return {
                    'price': price_data['price'],
                    'source': 'AlphaVantage',
                    'last_updated': price_data['last_updated']
                }
        except Exception as e:
            logger.debug(f"Alpha Vantage failed: {e}")
        
        return None
    
    def add_callback(self, callback):
        """添加回调函数"""
        self.callbacks.append(callback)
    
    def get_latest_data(self) -> Dict:
        """获取最新数据"""
        return self.data_buffer.copy()

class MarketDataProcessor:
    """市场数据处理器"""
    
    def __init__(self, data_collector: MultiSourceDataCollector):
        self.collector = data_collector
        self.price_history = []
        
        # 添加回调
        self.collector.add_callback(self._on_data_update)
    
    def _on_data_update(self, data: Dict):
        """数据更新回调"""
        if data.get('price'):
            self.price_history.append({
                'time': time.time(),
                'price': data['price']
            })
            
            # 保持最近100个价格点
            if len(self.price_history) > 100:
                self.price_history.pop(0)
    
    def get_price_momentum(self, window=20) -> Dict:
        """计算价格动量"""
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
    
    def get_market_features(self) -> Dict:
        """获取市场特征"""
        data = self.collector.get_latest_data()
        
        # 确保price有默认值
        price = data.get('price')
        if price is None:
            price = 0.0
        
        features = {
            'price': price,
            'source': data.get('source', 'Unknown'),
            'last_update': data.get('last_update', 'Unknown')
        }
        
        # 添加可用特征
        if 'bid' in data and 'ask' in data and data['bid'] and data['ask']:
            features['spread'] = data['ask'] - data['bid']
            features['spread_bps'] = (features['spread'] / data['bid']) * 10000 if data['bid'] > 0 else 0.0
        
        if 'price_change_24h' in data and data['price_change_24h'] is not None:
            features['price_change_24h'] = data['price_change_24h']
        
        if 'volume_24h' in data and data['volume_24h'] is not None:
            features['volume_24h'] = data['volume_24h']
        
        return features

# ========== 使用示例 ========== #
if __name__ == "__main__":
    # 创建多数据源收集器
    collector = MultiSourceDataCollector()
    processor = MarketDataProcessor(collector)
    
    # 启动数据收集 - 5分钟间隔
    collector.start(interval=300)
    
    try:
        # 主循环
        while True:
            # 获取市场特征
            momentum = processor.get_price_momentum()
            features = processor.get_market_features()
            
            print(f"\n=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            print(f"Source: {features.get('source', 'Unknown')}")
            print(f"Price: ${features.get('price', 0):.2f}")
            print(f"Momentum: {momentum.get('momentum', 0):.4f}")
            print(f"Volatility: {momentum.get('volatility', 0):.4f}")
            
            if 'spread_bps' in features and features['spread_bps'] is not None:
                print(f"Spread: {features['spread_bps']:.2f} bps")
            
            if 'price_change_24h' in features and features['price_change_24h'] is not None:
                print(f"24h Change: {features['price_change_24h']:.2f}%")
            
            if 'volume_24h' in features and features['volume_24h'] is not None:
                print(f"24h Volume: ${features['volume_24h']:,.0f}")
            
            print("-" * 50)
            
            # 等待5分钟
            time.sleep(300)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        collector.stop() 