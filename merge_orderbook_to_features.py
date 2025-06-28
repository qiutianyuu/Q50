#!/usr/bin/env python3
"""
离线批量把所有 orderbook & trades 数据转换成微观特征
按文件流式读取→计算→直接写 Parquet，避免一次性占满内存
"""

import glob
import os
import pandas as pd
import numpy as np
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from realtime_feature_engineering import RealtimeFeatureEngineering
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_micro_price_features_batch(orderbook_df: pd.DataFrame) -> pd.DataFrame:
    """批量计算微价格特征"""
    if orderbook_df.empty:
        return pd.DataFrame()
    
    features = []
    
    for _, row in orderbook_df.iterrows():
        # 直接使用解析好的OrderBook数据
        bid_prices = []
        bid_sizes = []
        ask_prices = []
        ask_sizes = []
        
        # 提取5档买卖盘数据
        for i in range(1, 6):
            bid_price_col = f'bid{i}_price'
            bid_size_col = f'bid{i}_size'
            ask_price_col = f'ask{i}_price'
            ask_size_col = f'ask{i}_size'
            
            if bid_price_col in row and bid_size_col in row:
                bid_prices.append(row[bid_price_col])
                bid_sizes.append(row[bid_size_col])
            
            if ask_price_col in row and ask_size_col in row:
                ask_prices.append(row[ask_price_col])
                ask_sizes.append(row[ask_size_col])
        
        if not bid_prices or not ask_prices:
            continue
        
        # 加权平均价格 (VWAP)
        bid_vwap = np.average(bid_prices, weights=bid_sizes) if bid_sizes else 0
        ask_vwap = np.average(ask_prices, weights=ask_sizes) if ask_sizes else 0
        
        # 中间价
        mid_price = (bid_prices[0] + ask_prices[0]) / 2 if bid_prices and ask_prices else 0
        
        # 买卖价差
        spread = ask_prices[0] - bid_prices[0] if bid_prices and ask_prices else 0
        spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0
        rel_spread = (spread / mid_price) if mid_price > 0 else 0
        
        # 订单簿不平衡
        bid_volume = sum(bid_sizes)
        ask_volume = sum(ask_sizes)
        volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
        
        # 价格压力
        price_pressure = (ask_vwap - mid_price) / mid_price if mid_price > 0 else 0
        
        # 填充概率和价格影响（简化计算）
        bid_fill_prob = 0.5  # 默认值
        ask_fill_prob = 0.5
        bid_price_impact = 0.0
        ask_price_impact = 0.0
        
        # 队列深度
        bid_queue_depth = bid_volume
        ask_queue_depth = ask_volume
        
        # 可用档位
        bid_levels_available = len(bid_prices)
        ask_levels_available = len(ask_prices)
        
        # 最优挂单大小（简化）
        optimal_bid_size = bid_sizes[0] if bid_sizes else 0
        optimal_ask_size = ask_sizes[0] if ask_sizes else 0
        
        features.append({
            'timestamp': row.get('timestamp', pd.Timestamp.now()),
            'bid_price': bid_prices[0] if bid_prices else 0,
            'ask_price': ask_prices[0] if ask_prices else 0,
            'mid_price': mid_price,
            'rel_spread': rel_spread,
            'bid_vwap': bid_vwap,
            'ask_vwap': ask_vwap,
            'spread': spread,
            'spread_bps': spread_bps,
            'volume_imbalance': volume_imbalance,
            'price_pressure': price_pressure,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'total_volume_x': bid_volume + ask_volume,
            'price_mean': mid_price,
            'price_std': 0.0,
            'price_range': 0.0,
            'total_volume_y': bid_volume + ask_volume,
            'avg_trade_size': 0.0,
            'large_trades': 0,
            'buy_ratio': 0.5,
            'price_momentum': 0.0,
            'vwap': mid_price,
            'trade_frequency': 0.0,
            'trade_count': 0,
            'bid_fill_prob': bid_fill_prob,
            'ask_fill_prob': ask_fill_prob,
            'bid_price_impact': bid_price_impact,
            'ask_price_impact': ask_price_impact,
            'optimal_bid_size': optimal_bid_size,
            'optimal_ask_size': optimal_ask_size,
            'bid_queue_depth': bid_queue_depth,
            'ask_queue_depth': ask_queue_depth,
            'bid_levels_available': bid_levels_available,
            'ask_levels_available': ask_levels_available,
            'price_change': 0.0,
            'price_change_abs': 0.0,
            'buy_pressure_change': 0.0,
            'spread_change': 0.0,
            'price_volatility': 0.0,
            'price_trend': 0.0,
            'trend_deviation': 0.0,
            'bid_fill_prob_change': 0.0,
            'ask_fill_prob_change': 0.0,
            'fill_prob_imbalance': 0.0,
            'price_impact_imbalance': 0.0
        })
    
    return pd.DataFrame(features)

def flush_batches(batches, symbol, batch_num):
    """将批次数据写入文件"""
    if not batches:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"data/micro_features_{symbol}_{timestamp}_batch_{batch_num}.parquet"
    
    # 合并所有批次
    combined_df = pd.concat(batches, ignore_index=True)
    combined_df.to_parquet(outfile, compression="zstd", index=False)
    
    total_rows = sum(len(df) for df in batches)
    logger.info(f"▶ 批次 {batch_num} 已写入: {outfile} ({total_rows} 行)")
    
    return outfile

def main(symbol="ETH-USDT"):
    """主函数：批量处理所有orderbook文件"""
    logger.info(f"开始批量处理 {symbol} 的orderbook数据...")
    
    # 获取所有orderbook文件
    order_files = sorted(glob.glob(f"data/websocket/orderbook_{symbol}_*.parquet"))
    logger.info(f"找到 {len(order_files)} 个orderbook文件")
    
    if not order_files:
        logger.error("没有找到orderbook文件")
        return
    
    # 创建输出目录
    os.makedirs("data", exist_ok=True)
    
    all_batches = []  # 当前批次的数据框列表
    batch_num = 1
    total_processed = 0
    output_files = []
    
    for idx, file_path in enumerate(order_files, 1):
        try:
            # 读取orderbook文件
            logger.info(f"处理文件 {idx}/{len(order_files)}: {os.path.basename(file_path)}")
            orderbook_df = pd.read_parquet(file_path)
            
            # 确保时间戳列存在
            if 'ts' in orderbook_df.columns:
                orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['ts'], unit='ms')
            elif 'timestamp' not in orderbook_df.columns:
                orderbook_df['timestamp'] = pd.to_datetime('now')
            
            # 计算微价格特征
            features_df = calculate_micro_price_features_batch(orderbook_df)
            
            if not features_df.empty:
                all_batches.append(features_df)
                total_processed += len(features_df)
            
            # 每累积一定行数就写入文件，避免内存占用过大
            current_batch_size = sum(len(df) for df in all_batches)
            if current_batch_size > 50000:  # 每5万行写入一次
                outfile = flush_batches(all_batches, symbol, batch_num)
                if outfile:
                    output_files.append(outfile)
                all_batches = []
                batch_num += 1
            
            # 进度报告
            if idx % 50 == 0:
                logger.info(f"已处理 {idx}/{len(order_files)} 文件，累计 {total_processed} 行特征")
                
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {e}")
            continue
    
    # 处理最后一批
    if all_batches:
        outfile = flush_batches(all_batches, symbol, batch_num)
        if outfile:
            output_files.append(outfile)
    
    # 合并所有批次文件
    if len(output_files) > 1:
        logger.info("合并所有批次文件...")
        combined_dfs = []
        for file_path in output_files:
            df = pd.read_parquet(file_path)
            combined_dfs.append(df)
        
        final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = f"data/micro_features_{symbol}_{final_timestamp}.parquet"
        final_df = pd.concat(combined_dfs, ignore_index=True)
        final_df.to_parquet(final_file, compression="zstd", index=False)
        
        logger.info(f"✅ 最终合并文件: {final_file} ({len(final_df)} 行)")
        
        # 清理临时批次文件
        for file_path in output_files:
            try:
                os.remove(file_path)
                logger.info(f"清理临时文件: {file_path}")
            except:
                pass
    elif len(output_files) == 1:
        # 只有一个文件，重命名
        final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = f"data/micro_features_{symbol}_{final_timestamp}.parquet"
        os.rename(output_files[0], final_file)
        logger.info(f"✅ 最终文件: {final_file}")
    
    logger.info(f"🎉 批量处理完成！总共处理 {total_processed} 行特征数据")

if __name__ == "__main__":
    main() 