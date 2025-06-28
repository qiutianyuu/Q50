#!/usr/bin/env python3
"""
OKX WebSocket 实时数据采集 - MVP版本
订阅 books5 (OrderBook top 5) 和 trades 频道
"""

import asyncio
import json
import websockets
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import logging
import argparse
from typing import Dict, List, Optional
import time
import signal
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('okx_websocket.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OKXWebSocket:
    """OKX WebSocket 客户端"""
    
    def __init__(self, symbol: str = "ETH-USDT", save_raw: bool = True, buffer_size: int = 1000):
        self.symbol = symbol
        self.save_raw = save_raw
        self.buffer_size = buffer_size
        
        # WebSocket URL
        self.ws_url = "wss://ws.okx.com:8443/ws/v5/public"
        
        # 数据缓冲区
        self.orderbook_buffer = []
        self.trades_buffer = []
        
        # 文件路径
        self.data_dir = Path("data/websocket")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 运行状态
        self.running = False
        self.connection = None
        
        # 统计信息
        self.stats = {
            "orderbook_count": 0,
            "trades_count": 0,
            "start_time": None,
            "last_orderbook_time": None,
            "last_trade_time": None
        }
    
    async def connect(self):
        """建立WebSocket连接"""
        try:
            logger.info(f"🔌 连接到 OKX WebSocket: {self.ws_url}")
            self.connection = await websockets.connect(self.ws_url)
            logger.info("✅ WebSocket连接成功")
            return True
        except Exception as e:
            logger.error(f"❌ WebSocket连接失败: {e}")
            return False
    
    async def subscribe(self):
        """订阅数据频道"""
        if not self.connection:
            logger.error("❌ WebSocket未连接")
            return False
        
        # 订阅消息
        subscribe_message = {
            "op": "subscribe",
            "args": [
                {
                    "channel": "books5",
                    "instId": self.symbol
                },
                {
                    "channel": "trades",
                    "instId": self.symbol
                }
            ]
        }
        
        try:
            await self.connection.send(json.dumps(subscribe_message))
            logger.info(f"📡 订阅频道: books5, trades ({self.symbol})")
            return True
        except Exception as e:
            logger.error(f"❌ 订阅失败: {e}")
            return False
    
    def parse_orderbook(self, data: Dict) -> Dict:
        """解析OrderBook数据"""
        try:
            # 从data['data']中提取订单簿数据
            orderbook_data = data.get('data', [])
            if not orderbook_data:
                logger.warning("OrderBook数据为空")
                return None
            
            # 取第一个数据点
            ob = orderbook_data[0]
            timestamp = int(ob.get('ts', 0))
            dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            
            # 解析bids和asks
            bids = ob.get('bids', [])
            asks = ob.get('asks', [])
            
            # 提取top 5
            parsed_data = {
                'timestamp': dt,
                'ts': timestamp,
                'symbol': self.symbol,
                'channel': 'books5'
            }
            
            # 添加bids (价格从高到低)
            for i, bid in enumerate(bids[:5]):
                price, size = bid[0], bid[1]
                parsed_data[f'bid{i+1}_price'] = float(price)
                parsed_data[f'bid{i+1}_size'] = float(size)
            
            # 添加asks (价格从低到高)
            for i, ask in enumerate(asks[:5]):
                price, size = ask[0], ask[1]
                parsed_data[f'ask{i+1}_price'] = float(price)
                parsed_data[f'ask{i+1}_size'] = float(size)
            
            # 计算spread
            if len(bids) > 0 and len(asks) > 0:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                parsed_data['spread'] = best_ask - best_bid
                parsed_data['spread_bps'] = (best_ask - best_bid) / best_bid * 10000
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"❌ 解析OrderBook失败: {e}")
            return None
    
    def parse_trade(self, data: Dict) -> List[Dict]:
        """解析Trades数据"""
        try:
            trades = data.get('data', [])
            parsed_trades = []
            
            for trade in trades:
                timestamp = int(trade.get('ts', 0))
                dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
                
                parsed_trade = {
                    'timestamp': dt,
                    'ts': timestamp,
                    'symbol': self.symbol,
                    'channel': 'trades',
                    'trade_id': trade.get('tradeId', ''),
                    'price': float(trade.get('px', 0)),
                    'size': float(trade.get('sz', 0)),
                    'side': trade.get('side', ''),  # 'buy' or 'sell'
                    'exec_type': trade.get('execType', '')
                }
                
                parsed_trades.append(parsed_trade)
            
            return parsed_trades
            
        except Exception as e:
            logger.error(f"❌ 解析Trades失败: {e}")
            return []
    
    async def handle_message(self, message: str):
        """处理接收到的消息"""
        try:
            data = json.loads(message)
            
            # 处理心跳
            if 'event' in data and data['event'] == 'ping':
                await self.connection.send(json.dumps({'event': 'pong'}))
                return
            
            # 处理订阅确认
            if 'event' in data and data['event'] == 'subscribe':
                logger.info(f"✅ 订阅确认: {data}")
                return
            
            # 处理数据
            if 'arg' in data and 'data' in data:
                channel = data['arg'].get('channel', '')
                
                if channel == 'books5':
                    # 处理OrderBook数据
                    orderbook_data = self.parse_orderbook(data)
                    if orderbook_data:
                        self.orderbook_buffer.append(orderbook_data)
                        self.stats['orderbook_count'] += 1
                        self.stats['last_orderbook_time'] = datetime.now()
                        
                        # 缓冲区满时保存
                        if len(self.orderbook_buffer) >= self.buffer_size:
                            await self.save_orderbook_data()
                
                elif channel == 'trades':
                    # 处理Trades数据
                    trades_data = self.parse_trade(data)
                    for trade in trades_data:
                        self.trades_buffer.append(trade)
                        self.stats['trades_count'] += 1
                        self.stats['last_trade_time'] = datetime.now()
                    
                    # 缓冲区满时保存
                    if len(self.trades_buffer) >= self.buffer_size:
                        await self.save_trades_data()
        
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON解析失败: {e}")
        except Exception as e:
            logger.error(f"❌ 消息处理失败: {e}")
    
    async def save_orderbook_data(self):
        """保存OrderBook数据"""
        if not self.orderbook_buffer:
            return
        
        try:
            df = pd.DataFrame(self.orderbook_buffer)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"orderbook_{self.symbol}_{timestamp}.parquet"
            filepath = self.data_dir / filename
            
            df.to_parquet(filepath, index=False)
            logger.info(f"💾 保存OrderBook数据: {len(df)} 条 -> {filepath}")
            
            # 清空缓冲区
            self.orderbook_buffer = []
            
        except Exception as e:
            logger.error(f"❌ 保存OrderBook数据失败: {e}")
    
    async def save_trades_data(self):
        """保存Trades数据"""
        if not self.trades_buffer:
            return
        
        try:
            df = pd.DataFrame(self.trades_buffer)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trades_{self.symbol}_{timestamp}.parquet"
            filepath = self.data_dir / filename
            
            df.to_parquet(filepath, index=False)
            logger.info(f"💾 保存Trades数据: {len(df)} 条 -> {filepath}")
            
            # 清空缓冲区
            self.trades_buffer = []
            
        except Exception as e:
            logger.error(f"❌ 保存Trades数据失败: {e}")
    
    async def print_stats(self):
        """打印统计信息"""
        while self.running:
            try:
                elapsed = datetime.now() - self.stats['start_time'] if self.stats['start_time'] else None
                elapsed_str = str(elapsed).split('.')[0] if elapsed else "N/A"
                
                logger.info(f"📊 统计信息:")
                logger.info(f"  运行时间: {elapsed_str}")
                logger.info(f"  OrderBook消息: {self.stats['orderbook_count']}")
                logger.info(f"  Trades消息: {self.stats['trades_count']}")
                logger.info(f"  OrderBook缓冲区: {len(self.orderbook_buffer)}")
                logger.info(f"  Trades缓冲区: {len(self.trades_buffer)}")
                
                if self.stats['last_orderbook_time']:
                    last_ob = datetime.now() - self.stats['last_orderbook_time']
                    logger.info(f"  最后OrderBook: {last_ob.total_seconds():.1f}秒前")
                
                if self.stats['last_trade_time']:
                    last_trade = datetime.now() - self.stats['last_trade_time']
                    logger.info(f"  最后Trade: {last_trade.total_seconds():.1f}秒前")
                
                await asyncio.sleep(30)  # 每30秒打印一次统计
                
            except Exception as e:
                logger.error(f"❌ 统计信息打印失败: {e}")
    
    async def run(self):
        """运行WebSocket客户端"""
        if not await self.connect():
            return
        
        if not await self.subscribe():
            return
        
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        logger.info("🚀 开始接收实时数据...")
        
        # 启动统计任务
        stats_task = asyncio.create_task(self.print_stats())
        
        try:
            async for message in self.connection:
                await self.handle_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ WebSocket连接断开")
        except Exception as e:
            logger.error(f"❌ WebSocket运行错误: {e}")
        finally:
            self.running = False
            stats_task.cancel()
            
            # 保存剩余数据
            await self.save_orderbook_data()
            await self.save_trades_data()
            
            logger.info("🛑 WebSocket客户端已停止")

def signal_handler(signum, frame):
    """信号处理器"""
    logger.info("🛑 收到停止信号，正在关闭...")
    sys.exit(0)

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='OKX WebSocket 实时数据采集')
    parser.add_argument('--symbol', type=str, default='ETH-USDT', help='交易对')
    parser.add_argument('--save-raw', action='store_true', default=True, help='保存原始数据')
    parser.add_argument('--buffer-size', type=int, default=1000, help='缓冲区大小')
    
    args = parser.parse_args()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🚀 启动 OKX WebSocket 客户端")
    logger.info(f"📊 交易对: {args.symbol}")
    logger.info(f"💾 保存原始数据: {args.save_raw}")
    logger.info(f"📦 缓冲区大小: {args.buffer_size}")
    
    client = OKXWebSocket(
        symbol=args.symbol,
        save_raw=args.save_raw,
        buffer_size=args.buffer_size
    )
    
    await client.run()

if __name__ == "__main__":
    asyncio.run(main()) 