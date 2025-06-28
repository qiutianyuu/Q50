import websocket
import json
import threading
import time
import logging
import requests
from queue import Queue
from collections import defaultdict
from typing import Dict, List, Optional, Callable
import pandas as pd
import os

# ========== CONFIG ========== #
BINANCE_WS_URL = "wss://stream.binance.com:9443"
BINANCE_REST_URL = "https://api.binance.com/api/v3"

# ========== LOGGING ========== #
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("BinanceWebSocket")

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'binance_config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}")
        return {}

class BinanceWebSocket:
    """
    Binance WebSocket客户端，支持实时order book和trade数据
    """
    
    def __init__(self, symbol="ethusdt", depth_limit=1000, config=None):
        self.symbol = symbol.lower()
        self.depth_limit = depth_limit
        self.config = config or load_config()
        self.ws = None
        self.connected = False
        self._stop = threading.Event()
        self._thread = None
        self._ping_interval = self.config.get('websocket', {}).get('ping_interval', 20)
        self._last_ping = time.time()
        
        # 数据队列
        self.trade_queue = Queue()
        self.depth_queue = Queue()
        self.kline_queue = Queue()
        
        # Order book管理
        self.order_book = {
            'bids': {},  # price -> quantity
            'asks': {},  # price -> quantity
            'last_update_id': 0
        }
        self.order_book_lock = threading.Lock()
        
        # 回调函数
        self.callbacks = {
            'trade': [],
            'depth': [],
            'kline': [],
            'order_book_updated': []
        }
        
        # 统计信息
        self.stats = {
            'trades_received': 0,
            'depth_updates_received': 0,
            'klines_received': 0,
            'last_trade_time': None,
            'last_depth_time': None,
            'last_kline_time': None
        }

    def _on_open(self, ws):
        """WebSocket连接打开时的回调"""
        logger.info(f"Binance WebSocket opened for {self.symbol}")
        self.connected = True
        
        # 订阅数据流
        self._subscribe_to_streams(ws)
        
        # 获取初始order book snapshot
        self._get_order_book_snapshot()

    def _subscribe_to_streams(self, ws):
        """订阅数据流"""
        # 根据配置文件决定订阅哪些流
        streams_config = self.config.get('streams', {})
        streams = []
        
        if streams_config.get('depth', True):
            streams.append(f"{self.symbol}@depth")
        if streams_config.get('trade', True):
            streams.append(f"{self.symbol}@trade")
        if streams_config.get('kline_5m', False):
            streams.append(f"{self.symbol}@kline_5m")
        if streams_config.get('kline_15m', False):
            streams.append(f"{self.symbol}@kline_15m")
        
        # 限制订阅数量，避免"Too many requests"错误
        if len(streams) > 3:
            streams = streams[:3]  # 只保留前3个最重要的流
        
        for stream in streams:
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [stream],
                "id": int(time.time() * 1000)
            }
            ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to {stream}")

    def _get_order_book_snapshot(self):
        """获取order book快照"""
        try:
            url = f"{BINANCE_REST_URL}/depth"
            params = {
                'symbol': self.symbol.upper(),
                'limit': self.depth_limit
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            with self.order_book_lock:
                # 清空现有order book
                self.order_book['bids'].clear()
                self.order_book['asks'].clear()
                
                # 填充bids
                for price, quantity in data['bids']:
                    self.order_book['bids'][float(price)] = float(quantity)
                
                # 填充asks
                for price, quantity in data['asks']:
                    self.order_book['asks'][float(price)] = float(quantity)
                
                self.order_book['last_update_id'] = data['lastUpdateId']
                
            logger.info(f"Order book snapshot loaded: {len(self.order_book['bids'])} bids, {len(self.order_book['asks'])} asks")
            
        except Exception as e:
            logger.error(f"Failed to get order book snapshot: {e}")

    def _on_message(self, ws, message):
        """处理WebSocket消息"""
        try:
            data = json.loads(message)
            
            # 处理depth update
            if 'e' in data and data['e'] == 'depthUpdate':
                self._handle_depth_update(data)
            
            # 处理trade
            elif 'e' in data and data['e'] == 'trade':
                self._handle_trade(data)
            
            # 处理kline
            elif 'e' in data and data['e'] == 'kline':
                self._handle_kline(data)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _handle_depth_update(self, data):
        """处理depth update"""
        try:
            with self.order_book_lock:
                # 检查update ID连续性
                if data['u'] <= self.order_book['last_update_id']:
                    return  # 忽略过期更新
                
                if data['U'] > self.order_book['last_update_id'] + 1:
                    logger.warning("Order book out of sync, reconnecting...")
                    self._get_order_book_snapshot()
                    return
                
                # 更新bids
                for price, quantity in data['b']:
                    price = float(price)
                    quantity = float(quantity)
                    if quantity == 0:
                        self.order_book['bids'].pop(price, None)
                    else:
                        self.order_book['bids'][price] = quantity
                
                # 更新asks
                for price, quantity in data['a']:
                    price = float(price)
                    quantity = float(quantity)
                    if quantity == 0:
                        self.order_book['asks'].pop(price, None)
                    else:
                        self.order_book['asks'][price] = quantity
                
                self.order_book['last_update_id'] = data['u']
            
            # 更新统计
            self.stats['depth_updates_received'] += 1
            self.stats['last_depth_time'] = time.time()
            
            # 放入队列
            self.depth_queue.put(data)
            
            # 触发回调
            for callback in self.callbacks['depth']:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Depth callback error: {e}")
            
            # 触发order book更新回调
            for callback in self.callbacks['order_book_updated']:
                try:
                    callback(self.get_order_book())
                except Exception as e:
                    logger.error(f"Order book callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling depth update: {e}")

    def _handle_trade(self, data):
        """处理trade数据"""
        try:
            # 更新统计
            self.stats['trades_received'] += 1
            self.stats['last_trade_time'] = time.time()
            
            # 放入队列
            self.trade_queue.put(data)
            
            # 触发回调
            for callback in self.callbacks['trade']:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Trade callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling trade: {e}")

    def _handle_kline(self, data):
        """处理K线数据"""
        try:
            # 更新统计
            self.stats['klines_received'] += 1
            self.stats['last_kline_time'] = time.time()
            
            # 放入队列
            self.kline_queue.put(data)
            
            # 触发回调
            for callback in self.callbacks['kline']:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Kline callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling kline: {e}")

    def _on_error(self, ws, error):
        """WebSocket错误处理"""
        logger.error(f"Binance WebSocket error: {error}")
        self.connected = False

    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket连接关闭处理"""
        logger.warning(f"Binance WebSocket closed: {close_status_code} {close_msg}")
        self.connected = False

    def _run(self):
        """WebSocket运行循环"""
        while not self._stop.is_set():
            try:
                self.ws = websocket.WebSocketApp(
                    f"{BINANCE_WS_URL}/ws/{self.symbol}@depth",
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                self.ws.run_forever(ping_interval=self._ping_interval)
            except Exception as e:
                logger.error(f"WebSocket run error: {e}")
            
            if not self._stop.is_set():
                reconnect_delay = self.config.get('websocket', {}).get('reconnect_delay', 5)
                time.sleep(reconnect_delay)  # 重连延迟

    def start(self):
        """启动WebSocket连接"""
        if self._thread and self._thread.is_alive():
            return
        
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Binance WebSocket thread started")

    def stop(self):
        """停止WebSocket连接"""
        self._stop.set()
        if self.ws:
            self.ws.close()
        logger.info("Binance WebSocket stopped")

    def get_order_book(self) -> Dict:
        """获取当前order book"""
        with self.order_book_lock:
            return {
                'bids': dict(sorted(self.order_book['bids'].items(), reverse=True)),
                'asks': dict(sorted(self.order_book['asks'].items())),
                'last_update_id': self.order_book['last_update_id']
            }

    def get_best_bid_ask(self) -> Dict:
        """获取最佳买卖价"""
        order_book = self.get_order_book()
        best_bid = max(order_book['bids'].keys()) if order_book['bids'] else None
        best_ask = min(order_book['asks'].keys()) if order_book['asks'] else None
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': best_ask - best_bid if best_bid and best_ask else None
        }

    def get_trade_data(self, timeout=1) -> Optional[Dict]:
        """获取trade数据"""
        try:
            return self.trade_queue.get(timeout=timeout)
        except:
            return None

    def get_depth_data(self, timeout=1) -> Optional[Dict]:
        """获取depth数据"""
        try:
            return self.depth_queue.get(timeout=timeout)
        except:
            return None

    def get_kline_data(self, timeout=1) -> Optional[Dict]:
        """获取K线数据"""
        try:
            return self.kline_queue.get(timeout=timeout)
        except:
            return None

    def add_callback(self, event_type: str, callback: Callable):
        """添加回调函数"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            logger.warning(f"Unknown event type: {event_type}")

    def remove_callback(self, event_type: str, callback: Callable):
        """移除回调函数"""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()

    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.connected


class BinanceDataProcessor:
    """
    Binance数据处理器，用于实时特征计算
    """
    
    def __init__(self, websocket_client: BinanceWebSocket):
        self.ws_client = websocket_client
        self.trade_history = []
        self.volume_profile = defaultdict(float)
        self.price_levels = defaultdict(float)
        
        # 添加回调
        self.ws_client.add_callback('trade', self._on_trade)
        self.ws_client.add_callback('order_book_updated', self._on_order_book_update)

    def _on_trade(self, trade_data):
        """处理trade数据"""
        trade = {
            'time': trade_data['T'],
            'price': float(trade_data['p']),
            'quantity': float(trade_data['q']),
            'is_buyer_maker': trade_data['m'],
            'trade_id': trade_data['t']
        }
        
        self.trade_history.append(trade)
        
        # 保持最近1000笔交易
        if len(self.trade_history) > 1000:
            self.trade_history.pop(0)
        
        # 更新成交量分布
        self.volume_profile[trade['price']] += trade['quantity']

    def _on_order_book_update(self, order_book):
        """处理order book更新"""
        # 更新价格水平
        self.price_levels.clear()
        for price, quantity in order_book['bids'].items():
            self.price_levels[price] = quantity
        for price, quantity in order_book['asks'].items():
            self.price_levels[price] = quantity

    def get_recent_trades(self, count=100) -> List[Dict]:
        """获取最近的交易"""
        return self.trade_history[-count:]

    def get_volume_imbalance(self, levels=10) -> Dict:
        """计算成交量不平衡"""
        order_book = self.ws_client.get_order_book()
        
        bid_volume = sum(list(order_book['bids'].values())[:levels])
        ask_volume = sum(list(order_book['asks'].values())[:levels])
        
        total_volume = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        
        return {
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'total_volume': total_volume,
            'imbalance': imbalance
        }

    def get_price_momentum(self, window=100) -> Dict:
        """计算价格动量"""
        if len(self.trade_history) < window:
            return {'momentum': 0, 'volatility': 0, 'current_price': 0}
        
        recent_trades = self.trade_history[-window:]
        prices = [trade['price'] for trade in recent_trades]
        
        if len(prices) < 2:
            return {'momentum': 0, 'volatility': 0, 'current_price': prices[0] if prices else 0}
        
        # 计算动量（价格变化率）
        momentum = (prices[-1] - prices[0]) / prices[0]
        
        # 计算波动率
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        volatility = (sum(r**2 for r in returns) / len(returns))**0.5 if returns else 0
        
        return {
            'momentum': momentum,
            'volatility': volatility,
            'current_price': prices[-1]
        }

    def get_order_book_features(self) -> Dict:
        """获取order book特征"""
        order_book = self.ws_client.get_order_book()
        best_bid_ask = self.ws_client.get_best_bid_ask()
        
        # 计算spread
        spread = best_bid_ask['spread'] or 0
        spread_bps = (spread / best_bid_ask['best_bid']) * 10000 if best_bid_ask['best_bid'] else 0
        
        # 计算order book深度
        bid_depth = sum(order_book['bids'].values())
        ask_depth = sum(order_book['asks'].values())
        
        # 计算价格压力
        bid_pressure = sum(price * qty for price, qty in order_book['bids'].items())
        ask_pressure = sum(price * qty for price, qty in order_book['asks'].items())
        
        return {
            'spread': spread,
            'spread_bps': spread_bps,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'total_depth': bid_depth + ask_depth,
            'bid_pressure': bid_pressure,
            'ask_pressure': ask_pressure,
            'pressure_ratio': bid_pressure / ask_pressure if ask_pressure > 0 else 1
        }


# ========== 使用示例 ========== #
if __name__ == "__main__":
    # 创建WebSocket客户端
    ws_client = BinanceWebSocket("ethusdt")
    processor = BinanceDataProcessor(ws_client)
    
    # 启动连接
    ws_client.start()
    
    try:
        # 等待连接建立
        time.sleep(5)
        
        # 主循环
        while True:
            # 获取实时特征
            momentum = processor.get_price_momentum()
            imbalance = processor.get_volume_imbalance()
            ob_features = processor.get_order_book_features()
            
            print(f"Price: {momentum['current_price']:.2f}")
            print(f"Momentum: {momentum['momentum']:.4f}")
            print(f"Volatility: {momentum['volatility']:.4f}")
            print(f"Volume Imbalance: {imbalance['imbalance']:.4f}")
            print(f"Spread: {ob_features['spread_bps']:.2f} bps")
            print("-" * 50)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        ws_client.stop() 