import numpy as np
import pandas as pd

def compute_cost(rel_spread, fee_rate=0.0005, include_double_fee=True):
    """计算交易成本（相对数值）"""
    fee = fee_rate * (2 if include_double_fee else 1)
    return rel_spread + fee

def make_labels(mid, rel_spread, horizon, alpha, fee_rate=0.0005, mode='taker', require_fill=False):
    """
    向量化生成标签
    
    Args:
        mid: mid_price Series
        rel_spread: 相对spread Series (spread / mid_price)
        horizon: 预测步数
        alpha: 成本倍数
        fee_rate: 手续费率
        mode: 'taker' 或 'maker'
        require_fill: 是否要求填单成交
    
    Returns:
        labels: 标签Series (1=long, -1=short, 0=neutral)
    """
    # 计算未来相对收益
    r = (mid.shift(-horizon) - mid) / mid
    
    # 计算交易成本
    if mode == 'maker':
        # Maker模式：成本 = 0.5*spread + 0.0001 (返佣)
        cost = 0.5 * rel_spread + 0.0001
    else:
        # Taker模式：成本 = spread + 2*fee_rate
        cost = compute_cost(rel_spread, fee_rate)
    
    # 计算阈值
    threshold = alpha * cost
    
    # 生成基础标签
    long_mask = (r > threshold)
    short_mask = (r < -threshold)
    
    labels = pd.Series(0, index=mid.index)
    labels[long_mask] = 1
    labels[short_mask] = -1
    
    # 如果需要填单验证
    if require_fill:
        # 计算挂单价
        bid_price = mid * (1 - 0.5 * rel_spread)  # 买单挂单价
        ask_price = mid * (1 + 0.5 * rel_spread)  # 卖单挂单价
        
        # 计算未来价格范围
        future_min = pd.Series(index=mid.index, dtype=float)
        future_max = pd.Series(index=mid.index, dtype=float)
        
        for i in range(len(mid) - horizon):
            future_window = mid.iloc[i:i+horizon+1]
            future_min.iloc[i] = future_window.min()
            future_max.iloc[i] = future_window.max()
        
        # 检查填单条件
        long_fill_mask = (future_min <= bid_price)  # 买单能被成交
        short_fill_mask = (future_max >= ask_price)  # 卖单能被成交
        
        # 更新标签：只有既满足收益条件又能成交的才保留
        labels = pd.Series(0, index=mid.index)
        labels[(long_mask) & (long_fill_mask)] = 1
        labels[(short_mask) & (short_fill_mask)] = -1
    
    return labels

def get_label_stats(labels):
    """计算标签统计信息"""
    total = len(labels)
    long_count = (labels == 1).sum()
    short_count = (labels == -1).sum()
    neutral_count = (labels == 0).sum()
    
    return {
        'total': total,
        'long_count': long_count,
        'short_count': short_count,
        'neutral_count': neutral_count,
        'long_pct': (long_count / total) * 100,
        'short_pct': (short_count / total) * 100,
        'neutral_pct': (neutral_count / total) * 100
    }

def test_labeling():
    """单元测试"""
    # 创建测试数据
    np.random.seed(42)
    n = 1000
    mid_price = pd.Series(2400 + np.cumsum(np.random.randn(n) * 0.1))
    rel_spread = pd.Series(0.0001 + np.random.rand(n) * 0.0001)  # 0.01-0.02%
    
    # 测试标签生成
    labels = make_labels(mid_price, rel_spread, horizon=30, alpha=1.0)
    stats = get_label_stats(labels)
    
    print("测试结果:")
    print(f"总样本: {stats['total']}")
    print(f"Long: {stats['long_count']} ({stats['long_pct']:.1f}%)")
    print(f"Short: {stats['short_count']} ({stats['short_pct']:.1f}%)")
    print(f"Neutral: {stats['neutral_count']} ({stats['neutral_pct']:.1f}%)")
    
    return labels, stats

if __name__ == "__main__":
    test_labeling() 