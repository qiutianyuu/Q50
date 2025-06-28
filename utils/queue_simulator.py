import pandas as pd
import numpy as np
from typing import Tuple, Dict

class QueueSimulator:
    """队列深度模拟器，计算挂单的队列位置和填单概率"""
    
    def __init__(self, depth_levels: int = 5):
        self.depth_levels = depth_levels
    
    def calculate_queue_position(self, orderbook_row: pd.Series, side: str = 'bid') -> Dict:
        """
        计算在指定价格档位的队列位置
        
        Args:
            orderbook_row: 单行orderbook数据
            side: 'bid' 或 'ask'
        
        Returns:
            包含队列信息的字典
        """
        if side == 'bid':
            price_cols = [f'bid{i}_price' for i in range(1, self.depth_levels + 1)]
            size_cols = [f'bid{i}_size' for i in range(1, self.depth_levels + 1)]
        else:
            price_cols = [f'ask{i}_price' for i in range(1, self.depth_levels + 1)]
            size_cols = [f'ask{i}_size' for i in range(1, self.depth_levels + 1)]
        
        # 获取价格和数量
        prices = [orderbook_row[col] for col in price_cols]
        sizes = [orderbook_row[col] for col in size_cols]
        
        # 计算累积数量
        cumulative_sizes = np.cumsum(sizes)
        
        return {
            'prices': prices,
            'sizes': sizes,
            'cumulative_sizes': cumulative_sizes,
            'side': side
        }
    
    def estimate_fill_probability(self, orderbook_row: pd.Series, target_price: float, 
                                side: str = 'bid', trade_volume: float = None) -> float:
        """
        估算在指定价格填单的概率
        
        Args:
            orderbook_row: 单行orderbook数据
            target_price: 目标价格
            side: 'bid' 或 'ask'
            trade_volume: 预期交易量，如果为None则使用默认值
        
        Returns:
            填单概率 (0-1)
        """
        queue_info = self.calculate_queue_position(orderbook_row, side)
        
        # 找到目标价格对应的档位
        price_level = None
        for i, price in enumerate(queue_info['prices']):
            if side == 'bid' and price >= target_price:
                price_level = i
                break
            elif side == 'ask' and price <= target_price:
                price_level = i
                break
        
        if price_level is None:
            return 0.0  # 价格超出深度范围
        
        # 计算在该价格档位之前的累积数量
        if price_level == 0:
            queue_position = 0
        else:
            queue_position = queue_info['cumulative_sizes'][price_level - 1]
        
        # 估算预期交易量
        if trade_volume is None:
            # 使用历史平均交易量或默认值
            trade_volume = 0.1  # 默认0.1 ETH
        
        # 计算填单概率
        # 如果预期交易量大于队列位置，有较高概率填单
        if trade_volume >= queue_position:
            fill_prob = min(0.95, 0.5 + (trade_volume - queue_position) / max(trade_volume, 1))
        else:
            fill_prob = max(0.05, trade_volume / max(queue_position, 1))
        
        return fill_prob
    
    def calculate_market_impact(self, orderbook_row: pd.Series, order_size: float, 
                              side: str = 'bid') -> Dict:
        """
        计算订单对市场的影响
        
        Args:
            orderbook_row: 单行orderbook数据
            order_size: 订单大小
            side: 'bid' 或 'ask'
        
        Returns:
            市场影响信息
        """
        queue_info = self.calculate_queue_position(orderbook_row, side)
        
        # 计算订单会吃掉多少档位
        remaining_size = order_size
        levels_consumed = 0
        weighted_price = 0
        total_consumed = 0
        
        for i, size in enumerate(queue_info['sizes']):
            if remaining_size <= 0:
                break
            
            consumed = min(remaining_size, size)
            weighted_price += consumed * queue_info['prices'][i]
            total_consumed += consumed
            remaining_size -= consumed
            levels_consumed += 1
        
        if total_consumed > 0:
            avg_price = weighted_price / total_consumed
        else:
            avg_price = queue_info['prices'][0]
        
        # 计算价格冲击
        if side == 'bid':
            price_impact = (avg_price - queue_info['prices'][0]) / queue_info['prices'][0]
        else:
            price_impact = (queue_info['prices'][0] - avg_price) / queue_info['prices'][0]
        
        return {
            'levels_consumed': levels_consumed,
            'avg_price': avg_price,
            'price_impact': price_impact,
            'total_consumed': total_consumed,
            'remaining_size': remaining_size
        }
    
    def estimate_optimal_order_size(self, orderbook_row: pd.Series, side: str = 'bid', 
                                  max_impact: float = 0.001) -> float:
        """
        估算最优订单大小（在最大冲击范围内）
        
        Args:
            orderbook_row: 单行orderbook数据
            side: 'bid' 或 'ask'
            max_impact: 最大允许的价格冲击
        
        Returns:
            最优订单大小
        """
        # 二分查找最优订单大小
        min_size = 0.01
        max_size = 10.0  # 最大10 ETH
        
        for _ in range(10):  # 最多10次迭代
            test_size = (min_size + max_size) / 2
            impact = self.calculate_market_impact(orderbook_row, test_size, side)
            
            if impact['price_impact'] <= max_impact:
                min_size = test_size
            else:
                max_size = test_size
        
        return min_size

def calculate_queue_features(orderbook_df: pd.DataFrame) -> pd.DataFrame:
    """
    为orderbook数据计算队列相关特征
    
    Args:
        orderbook_df: orderbook数据框
    
    Returns:
        包含队列特征的数据框
    """
    simulator = QueueSimulator()
    
    features = []
    
    for idx, row in orderbook_df.iterrows():
        # 计算bid和ask的队列信息
        bid_queue = simulator.calculate_queue_position(row, 'bid')
        ask_queue = simulator.calculate_queue_position(row, 'ask')
        
        # 计算填单概率（假设在最优价格挂单）
        bid_fill_prob = simulator.estimate_fill_probability(row, bid_queue['prices'][0], 'bid')
        ask_fill_prob = simulator.estimate_fill_probability(row, ask_queue['prices'][0], 'ask')
        
        # 计算市场冲击
        bid_impact = simulator.calculate_market_impact(row, 0.1, 'bid')
        ask_impact = simulator.calculate_market_impact(row, 0.1, 'ask')
        
        # 计算最优订单大小
        optimal_bid_size = simulator.estimate_optimal_order_size(row, 'bid')
        optimal_ask_size = simulator.estimate_optimal_order_size(row, 'ask')
        
        feature_row = {
            'timestamp': row['timestamp'],
            'bid_fill_prob': bid_fill_prob,
            'ask_fill_prob': ask_fill_prob,
            'bid_price_impact': bid_impact['price_impact'],
            'ask_price_impact': ask_impact['price_impact'],
            'optimal_bid_size': optimal_bid_size,
            'optimal_ask_size': optimal_ask_size,
            'bid_queue_depth': bid_queue['cumulative_sizes'][-1],
            'ask_queue_depth': ask_queue['cumulative_sizes'][-1],
            'bid_levels_available': len([s for s in bid_queue['sizes'] if s > 0]),
            'ask_levels_available': len([s for s in ask_queue['sizes'] if s > 0])
        }
        
        features.append(feature_row)
    
    return pd.DataFrame(features)

if __name__ == "__main__":
    # 测试代码
    print("Queue Simulator 测试完成") 