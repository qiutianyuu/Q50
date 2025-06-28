#!/usr/bin/env python3
"""
收集Funding费率和滑点数据
"""
import pandas as pd
import numpy as np
import ccxt
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def collect_funding_rates():
    """收集ETH-USDT的Funding费率"""
    print("📥 收集Funding费率数据...")
    
    # 初始化Binance
    exchange = ccxt.binance({
        'apiKey': 'BS3utDEquhRHvnbv0Kuvjcz6EiUYAovcSxJtNTTRRkFcr8MhTOqTfFWMV0CEDKLQ',
        'secret': 'sjqTZkT56nCEKardas6pIypF2dgPapYogiQ0e3pdR70NaJlfkhXJucrI5oMzunag',
        'sandbox': False,
        'enableRateLimit': True
    })
    
    # 获取历史Funding费率
    funding_data = []
    start_time = int(datetime(2023, 1, 1).timestamp() * 1000)
    end_time = int(datetime(2025, 1, 1).timestamp() * 1000)
    
    current_time = start_time
    while current_time < end_time:
        try:
            # 获取Funding费率
            funding = exchange.fetch_funding_rate_history(
                symbol='ETH/USDT',
                since=current_time,
                limit=1000
            )
            
            for item in funding:
                funding_data.append({
                    'timestamp': pd.to_datetime(item['timestamp'], unit='ms'),
                    'funding_rate': item['fundingRate'],
                    'funding_time': pd.to_datetime(item['fundingTime'], unit='ms')
                })
            
            # 更新时间
            if funding:
                current_time = funding[-1]['timestamp'] + 1
            else:
                current_time += 8 * 60 * 60 * 1000  # 8小时
            
            print(f"已收集到 {len(funding_data)} 条Funding记录...")
            time.sleep(0.1)  # 避免频率限制
            
        except Exception as e:
            print(f"收集Funding费率出错: {e}")
            current_time += 8 * 60 * 60 * 1000
            time.sleep(1)
    
    # 转换为DataFrame并处理
    funding_df = pd.DataFrame(funding_data)
    funding_df = funding_df.drop_duplicates()
    funding_df = funding_df.sort_values('timestamp')
    
    # 重采样到15分钟
    funding_df.set_index('timestamp', inplace=True)
    funding_15m = funding_df['funding_rate'].resample('15T').ffill()
    
    # 计算年化费率到15分钟
    funding_15m_annual = funding_15m * 3 * 365  # 8小时一次，年化到15分钟
    
    funding_15m_df = pd.DataFrame({
        'timestamp': funding_15m.index,
        'funding_rate_15m': funding_15m_annual
    })
    
    # 保存
    funding_15m_df.to_parquet('data/funding_rates_15m.parquet', compression='zstd')
    print(f"💾 Funding费率已保存: {len(funding_15m_df)} 条记录")
    
    return funding_15m_df

def estimate_slippage_model():
    """估算滑点模型"""
    print("📊 估算滑点模型...")
    
    # 基于历史数据的简单滑点模型
    # 实际滑点 = max(1bp, 0.1% / 市场深度)
    # 这里用简化模型：基于交易量和波动率
    
    # 读取价格数据
    df = pd.read_parquet("/Users/qiutianyu/features_offline_15m.parquet")
    
    # 计算基础滑点
    df['volume_usd'] = df['volume'] * df['close']
    df['volatility_15m'] = df['close'].pct_change().abs()
    
    # 滑点模型：基础1bp + 波动率调整 + 成交量调整
    base_slippage = 0.0001  # 1bp基础滑点
    vol_adjustment = df['volatility_15m'] * 10  # 波动率放大
    volume_adjustment = np.where(df['volume_usd'] > 1000000, 0.5, 1.0)  # 大成交量时滑点减半
    
    df['slippage_bp'] = (base_slippage + vol_adjustment) * volume_adjustment
    df['slippage_bp'] = df['slippage_bp'].clip(0.0001, 0.01)  # 限制在1-100bp
    
    # 保存滑点数据
    slippage_df = df[['timestamp', 'close', 'volume', 'slippage_bp']].copy()
    slippage_df.to_parquet('data/slippage_model_15m.parquet', compression='zstd')
    
    print(f"💾 滑点模型已保存: {len(slippage_df)} 条记录")
    print(f"滑点统计: 平均={slippage_df['slippage_bp'].mean():.4f}, 最大={slippage_df['slippage_bp'].max():.4f}")
    
    return slippage_df

def create_cost_table():
    """创建综合成本表"""
    print("🔧 创建综合成本表...")
    
    # 读取基础数据
    df = pd.read_parquet("/Users/qiutianyu/features_offline_15m.parquet")
    
    # 读取Funding费率
    try:
        funding_df = pd.read_parquet('data/funding_rates_15m.parquet')
        df = df.merge(funding_df, on='timestamp', how='left')
        df['funding_rate_15m'] = df['funding_rate_15m'].fillna(0)
    except:
        print("⚠️ 未找到Funding费率数据，使用0")
        df['funding_rate_15m'] = 0
    
    # 计算滑点
    df['volume_usd'] = df['volume'] * df['close']
    df['volatility_15m'] = df['close'].pct_change().abs()
    base_slippage = 0.0001
    vol_adjustment = df['volatility_15m'] * 10
    volume_adjustment = np.where(df['volume_usd'] > 1000000, 0.5, 1.0)
    df['slippage_bp'] = (base_slippage + vol_adjustment) * volume_adjustment
    df['slippage_bp'] = df['slippage_bp'].clip(0.0001, 0.01)
    
    # 计算总成本
    fee_rate = 0.0004  # 0.04%手续费
    df['total_cost_bp'] = fee_rate * 2 + df['slippage_bp'] + df['funding_rate_15m']  # 开平各一次手续费
    
    # 保存成本表
    cost_df = df[['timestamp', 'close', 'volume', 'funding_rate_15m', 'slippage_bp', 'total_cost_bp']].copy()
    cost_df.to_parquet('data/cost_table_15m.parquet', compression='zstd')
    
    print(f"💾 成本表已保存: {len(cost_df)} 条记录")
    print(f"成本统计:")
    print(f"  手续费: {fee_rate*2*10000:.1f}bp")
    print(f"  平均滑点: {cost_df['slippage_bp'].mean()*10000:.1f}bp")
    print(f"  平均Funding: {cost_df['funding_rate_15m'].mean()*10000:.1f}bp")
    print(f"  总成本: {cost_df['total_cost_bp'].mean()*10000:.1f}bp")
    
    return cost_df

def main():
    """主函数"""
    print("🚀 开始收集成本数据...")
    
    # 创建数据目录
    import os
    os.makedirs('data', exist_ok=True)
    
    # 收集Funding费率
    try:
        funding_df = collect_funding_rates()
        print("✅ Funding费率收集成功")
    except Exception as e:
        print(f"⚠️ Funding费率收集失败: {e}")
        print("使用0作为Funding费率")
    
    # 估算滑点模型
    slippage_df = estimate_slippage_model()
    
    # 创建综合成本表
    cost_df = create_cost_table()
    
    print("✅ 成本数据收集完成！")

if __name__ == "__main__":
    main() 