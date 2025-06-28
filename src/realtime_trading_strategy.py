import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.multi_source_data import MultiSourceDataCollector, MarketDataProcessor
from src.xgboost_module import XGBoostManager

# ========== LOGGING ========== #
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("RealtimeTrading")

class RealtimeTradingStrategy:
    """
    实时交易策略 - 整合OKX API和机器学习模型
    """
    
    def __init__(self, config=None):
        self.config = config or self._load_config()
        
        # 数据收集器
        self.data_collector = MultiSourceDataCollector()
        self.data_processor = MarketDataProcessor(self.data_collector)
        
        # 机器学习模型
        self.xgb_manager = XGBoostManager()
        self.model_loaded = False
        
        # 交易状态
        self.position = None
        self.trade_history = []
        self.last_signal_time = None
        self.signal_cooldown = 300  # 5分钟冷却期
        
        # 策略参数
        self.threshold = self.config.get('threshold', 0.55)
        self.confidence = self.config.get('confidence', 0.9)
        self.position_size = self.config.get('position_size', 0.01)
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.02)
        self.take_profit_pct = self.config.get('take_profit_pct', 0.03)
        
        # 统计信息
        self.stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'last_update': None
        }
    
    def _load_config(self) -> Dict:
        """加载配置"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'binance_config.json')
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}")
            return {}
    
    def load_model(self, model_path: str):
        """加载机器学习模型"""
        try:
            self.xgb_manager.load_model(model_path)
            self.model_loaded = True
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
    
    def start(self):
        """启动策略"""
        # 启动数据收集
        self.data_collector.start(interval=300)  # 5分钟间隔
        
        # 启动策略循环
        self.strategy_thread = threading.Thread(target=self._strategy_loop, daemon=True)
        self.strategy_thread.start()
        
        logger.info("Realtime trading strategy started")
    
    def stop(self):
        """停止策略"""
        self.data_collector.stop()
        logger.info("Realtime trading strategy stopped")
    
    def _strategy_loop(self):
        """策略主循环"""
        while True:
            try:
                # 获取市场数据
                market_features = self.data_processor.get_market_features()
                momentum = self.data_processor.get_price_momentum()
                
                # 检查是否有新数据
                if market_features['price'] > 0:
                    # 生成交易信号
                    signal = self._generate_signal(market_features, momentum)
                    
                    if signal:
                        self._process_signal(signal, market_features)
                    
                    # 更新持仓状态
                    self._update_position(market_features)
                    
                    # 更新统计
                    self.stats['last_update'] = datetime.now().isoformat()
                
            except Exception as e:
                logger.error(f"Strategy loop error: {e}")
            
            # 等待1分钟
            time.sleep(60)
    
    def _generate_signal(self, market_features: Dict, momentum: Dict) -> Optional[Dict]:
        """生成交易信号"""
        if not self.model_loaded:
            return None
        
        # 检查冷却期
        if self.last_signal_time and (datetime.now() - self.last_signal_time).seconds < self.signal_cooldown:
            return None
        
        try:
            # 构建特征向量
            features = self._build_feature_vector(market_features, momentum)
            
            # 模型预测
            prediction = self.xgb_manager.predict(features)
            probability = prediction[0] if len(prediction) > 0 else 0.5
            
            # 生成信号
            signal = None
            if probability > self.threshold:
                signal = {
                    'type': 'LONG',
                    'probability': probability,
                    'confidence': 'HIGH' if probability > self.confidence else 'MEDIUM',
                    'price': market_features['price'],
                    'timestamp': datetime.now().isoformat(),
                    'features': features
                }
            elif probability < (1 - self.threshold):
                signal = {
                    'type': 'SHORT',
                    'probability': 1 - probability,
                    'confidence': 'HIGH' if (1 - probability) > self.confidence else 'MEDIUM',
                    'price': market_features['price'],
                    'timestamp': datetime.now().isoformat(),
                    'features': features
                }
            
            if signal:
                self.last_signal_time = datetime.now()
                self.stats['signals_generated'] += 1
                logger.info(f"Signal generated: {signal['type']} {signal['confidence']} (prob: {signal['probability']:.3f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return None
    
    def _build_feature_vector(self, market_features: Dict, momentum: Dict) -> List[float]:
        """构建特征向量"""
        features = []
        
        # 价格特征
        features.append(market_features.get('price', 0))
        
        # 动量特征
        features.append(momentum.get('momentum', 0))
        features.append(momentum.get('volatility', 0))
        features.append(momentum.get('current_price', 0))
        
        # 市场特征
        features.append(market_features.get('spread_bps', 0))
        features.append(market_features.get('price_change_24h', 0))
        features.append(market_features.get('volume_24h', 0))
        
        # 技术指标（简化版）
        features.extend([
            momentum.get('momentum', 0) * 100,  # 动量百分比
            momentum.get('volatility', 0) * 100,  # 波动率百分比
            market_features.get('spread_bps', 0) / 100,  # 标准化spread
        ])
        
        # 填充到固定长度（根据模型要求调整）
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]  # 确保长度一致
    
    def _process_signal(self, signal: Dict, market_features: Dict):
        """处理交易信号"""
        if self.position:
            logger.info(f"Position already open, ignoring signal: {signal['type']}")
            return
        
        # 检查信号质量
        if signal['confidence'] == 'HIGH':
            # 执行交易
            self._execute_trade(signal, market_features)
        else:
            logger.info(f"Signal confidence too low: {signal['confidence']}")
    
    def _execute_trade(self, signal: Dict, market_features: Dict):
        """执行交易"""
        try:
            entry_price = market_features['price']
            
            # 计算止损止盈
            if signal['type'] == 'LONG':
                stop_loss = entry_price * (1 - self.stop_loss_pct)
                take_profit = entry_price * (1 + self.take_profit_pct)
            else:  # SHORT
                stop_loss = entry_price * (1 + self.stop_loss_pct)
                take_profit = entry_price * (1 - self.take_profit_pct)
            
            # 创建持仓
            self.position = {
                'type': signal['type'],
                'entry_price': entry_price,
                'entry_time': datetime.now(),
                'size': self.position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'signal_probability': signal['probability']
            }
            
            self.stats['trades_executed'] += 1
            
            logger.info(f"Trade executed: {signal['type']} at ${entry_price:.2f}")
            logger.info(f"Stop Loss: ${stop_loss:.2f}, Take Profit: ${take_profit:.2f}")
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    
    def _update_position(self, market_features: Dict):
        """更新持仓状态"""
        if not self.position:
            return
        
        current_price = market_features['price']
        position = self.position
        
        # 检查止损止盈
        if position['type'] == 'LONG':
            if current_price <= position['stop_loss']:
                self._close_position(current_price, 'Stop Loss')
            elif current_price >= position['take_profit']:
                self._close_position(current_price, 'Take Profit')
        else:  # SHORT
            if current_price >= position['stop_loss']:
                self._close_position(current_price, 'Stop Loss')
            elif current_price <= position['take_profit']:
                self._close_position(current_price, 'Take Profit')
    
    def _close_position(self, exit_price: float, reason: str):
        """平仓"""
        if not self.position:
            return
        
        entry_price = self.position['entry_price']
        position_type = self.position['type']
        
        # 计算盈亏
        if position_type == 'LONG':
            pnl = (exit_price - entry_price) / entry_price
        else:  # SHORT
            pnl = (entry_price - exit_price) / entry_price
        
        pnl_amount = pnl * self.position_size
        
        # 记录交易
        trade = {
            'entry_time': self.position['entry_time'],
            'exit_time': datetime.now(),
            'type': position_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_amount': pnl_amount,
            'reason': reason,
            'signal_probability': self.position['signal_probability']
        }
        
        self.trade_history.append(trade)
        
        # 更新统计
        self.stats['total_pnl'] += pnl_amount
        
        if len(self.trade_history) > 0:
            winning_trades = sum(1 for t in self.trade_history if t['pnl'] > 0)
            self.stats['win_rate'] = winning_trades / len(self.trade_history)
        
        logger.info(f"Position closed: {reason}")
        logger.info(f"PnL: {pnl:.2%} (${pnl_amount:.2f})")
        
        # 清空持仓
        self.position = None
    
    def get_status(self) -> Dict:
        """获取策略状态"""
        return {
            'position': self.position,
            'stats': self.stats,
            'last_trades': self.trade_history[-5:] if self.trade_history else [],
            'market_data': self.data_processor.get_market_features(),
            'momentum': self.data_processor.get_price_momentum()
        }
    
    def print_status(self):
        """打印策略状态"""
        status = self.get_status()
        
        print(f"\n=== RexKing Trading Strategy Status ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model Loaded: {self.model_loaded}")
        
        # 市场数据
        market_data = status['market_data']
        print(f"Price: ${market_data.get('price', 0):.2f}")
        print(f"Source: {market_data.get('source', 'Unknown')}")
        
        if 'spread_bps' in market_data:
            print(f"Spread: {market_data['spread_bps']:.2f} bps")
        
        if 'price_change_24h' in market_data:
            print(f"24h Change: {market_data['price_change_24h']:.2f}%")
        
        # 动量数据
        momentum = status['momentum']
        print(f"Momentum: {momentum.get('momentum', 0):.4f}")
        print(f"Volatility: {momentum.get('volatility', 0):.4f}")
        
        # 持仓状态
        position = status['position']
        if position:
            print(f"\nPosition: {position['type']}")
            print(f"Entry Price: ${position['entry_price']:.2f}")
            print(f"Stop Loss: ${position['stop_loss']:.2f}")
            print(f"Take Profit: ${position['take_profit']:.2f}")
        else:
            print(f"\nPosition: None")
        
        # 统计信息
        stats = status['stats']
        print(f"\nStatistics:")
        print(f"Signals Generated: {stats['signals_generated']}")
        print(f"Trades Executed: {stats['trades_executed']}")
        print(f"Total PnL: ${stats['total_pnl']:.2f}")
        print(f"Win Rate: {stats['win_rate']:.2%}")
        
        print("-" * 50)


# ========== 使用示例 ========== #
if __name__ == "__main__":
    # 创建策略
    strategy = RealtimeTradingStrategy()
    
    # 加载模型（如果有的话）
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'xgb_15m_optuna_optimized.bin')
    if os.path.exists(model_path):
        strategy.load_model(model_path)
    else:
        logger.warning(f"Model not found: {model_path}")
    
    # 启动策略
    strategy.start()
    
    try:
        # 主循环 - 每5分钟打印一次状态
        while True:
            strategy.print_status()
            time.sleep(300)  # 5分钟
            
    except KeyboardInterrupt:
        print("\nStopping strategy...")
        strategy.stop() 