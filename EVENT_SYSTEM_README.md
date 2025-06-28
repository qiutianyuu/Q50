# RexKing 事件系统使用指南

## 概述

RexKing 事件系统是一个基于事件检测和标签生成的量化交易信号系统。该系统能够识别市场中的关键事件，并基于这些事件生成交易信号标签。

## 系统架构

```
特征数据 → 事件检测 → 标签生成 → 交易信号
    ↓           ↓           ↓           ↓
features → detect_events → label_events → signals
```

## 核心组件

### 1. 事件检测器 (detect_events.py)

**功能**: 从特征数据中检测各种市场事件

**支持的事件类型**:
- **价格事件**: 价格突破、价格反转、新高新低
- **成交量事件**: 成交量异常、价量背离、放量突破
- **技术指标事件**: RSI超买超卖、布林带突破、MACD交叉
- **趋势事件**: 趋势强度、EMA交叉、趋势反转
- **鲸鱼事件**: 大额流入流出、鲸鱼活动异常、积累分发模式
- **市场状态事件**: 波动率状态、牛市熊市、横盘整理

**输出特征**:
- `event_strength`: 事件强度评分 (-1 到 1)
- `event_density`: 事件密度 (滚动窗口内事件数量)
- `event_consistency`: 事件一致性 (多空事件偏向)
- `bullish_event_count`: 看涨事件数量
- `bearish_event_count`: 看跌事件数量
- `neutral_event_count`: 中性事件数量

### 2. 标签生成器 (label_events.py)

**功能**: 基于检测到的事件生成交易信号标签

**支持的标签策略**:
- **event_strength**: 基于事件强度的标签生成
- **event_combination**: 基于事件组合的标签生成
- **event_sequential**: 基于事件时序的标签生成
- **ml_based**: 基于机器学习的标签生成

**输出标签**:
- `1`: 做多信号
- `0`: 做空信号
- `-1`: 不交易

## 快速开始

### 1. 环境准备

确保已安装必要的依赖:
```bash
pip install pandas numpy scikit-learn ta pathlib
```

### 2. 运行事件检测

```bash
python detect_events.py \
    --input data/features_15m_enhanced.parquet \
    --output data/events_15m.parquet
```

### 3. 生成交易标签

```bash
python label_events.py \
    --input data/events_15m.parquet \
    --output data/labels_15m_event_strength.parquet \
    --strategy event_strength \
    --min_strength 0.3 \
    --max_strength 0.8 \
    --min_density 3 \
    --hold_period 4
```

### 4. 使用完整工作流程

```bash
python run_event_system.py \
    --input data/features_15m_enhanced.parquet \
    --timeframe 15m \
    --strategy event_strength
```

### 5. 运行多种策略比较

```bash
python run_event_system.py \
    --input data/features_15m_enhanced.parquet \
    --timeframe 15m \
    --multi
```

## 参数配置

### 事件检测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `price_breakout_threshold` | 0.02 | 价格突破阈值 (2%) |
| `volume_spike_threshold` | 2.0 | 成交量异常阈值 (2倍均值) |
| `rsi_oversold` | 30 | RSI超卖阈值 |
| `rsi_overbought` | 70 | RSI超买阈值 |
| `whale_activity_threshold` | 2.0 | 鲸鱼活动z-score阈值 |
| `trend_strength_threshold` | 25 | ADX趋势强度阈值 |

### 标签生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_event_strength` | 0.3 | 最小事件强度 |
| `max_event_strength` | 0.8 | 最大事件强度 |
| `min_event_density` | 3 | 最小事件密度 |
| `hold_period` | 4 | 持仓周期(K线数) |
| `min_profit_threshold` | 0.001 | 最小收益阈值 |
| `max_loss_threshold` | -0.002 | 最大损失阈值 |

## 标签策略详解

### 1. 事件强度策略 (event_strength)

基于事件强度、密度和一致性生成标签：

```python
bullish_mask = (
    (event_strength >= min_strength) &
    (event_strength <= max_strength) &
    (event_density >= min_density) &
    (event_consistency >= min_consistency)
)
```

**适用场景**: 适合波动较大的市场，需要强信号确认

### 2. 事件组合策略 (event_combination)

基于特定事件组合生成标签：

```python
bullish_combinations = [
    (price_breakout == 1) & (volume_breakout == 1),
    (rsi_oversold == 1) & (price_reversal_up == 1),
    (macd_bullish_cross == 1) & (trend_strong == 1),
    (whale_large_inflow == 1) & (price_breakout == 1)
]
```

**适用场景**: 适合需要多重确认的市场环境

### 3. 事件时序策略 (event_sequential)

基于事件的时间序列模式生成标签：

```python
bullish_sequence = (
    (event_strength[t-2] < 0) &  # 前2个时间点事件强度为负
    (event_strength[t-1] > 0) &  # 前1个时间点事件强度为正
    (event_strength[t] > 0.3) &  # 当前时间点事件强度较高
    (event_density[t] >= 3)      # 当前事件密度较高
)
```

**适用场景**: 适合捕捉趋势转折点

## 测试和验证

### 运行测试

```bash
python test_event_system.py
```

测试脚本会验证：
- 事件检测功能
- 标签生成功能
- 数据一致性
- 结果质量

### 结果分析

系统会生成详细的统计报告：
- 事件分布统计
- 标签分布统计
- 收益分析
- 策略比较

## 最佳实践

### 1. 数据准备

确保输入特征数据包含必要的技术指标：
- RSI、MACD、布林带等基础指标
- 成交量相关指标
- 趋势指标 (EMA、ADX等)
- 鲸鱼数据 (可选)

### 2. 参数调优

根据市场环境调整参数：
- **高波动市场**: 提高事件强度阈值
- **低波动市场**: 降低事件强度阈值
- **趋势市场**: 使用事件组合策略
- **震荡市场**: 使用事件时序策略

### 3. 成本控制

系统内置成本感知机制：
- 考虑交易手续费
- 考虑滑点成本
- 考虑资金费率
- 设置最小收益阈值

### 4. 风险管理

- 设置最大损失阈值
- 控制信号频率
- 监控事件密度
- 定期回测验证

## 故障排除

### 常见问题

1. **缺少技术指标**
   ```
   错误: 缺少技术指标，请先运行特征工程
   解决: 确保输入数据包含必要的技术指标
   ```

2. **事件检测失败**
   ```
   错误: 事件检测失败
   解决: 检查输入数据格式和时间戳对齐
   ```

3. **标签生成失败**
   ```
   错误: 标签生成失败
   解决: 检查事件文件是否存在，调整参数阈值
   ```

4. **内存不足**
   ```
   错误: 内存不足
   解决: 分批处理数据或减少数据量
   ```

### 调试模式

启用详细日志输出：
```bash
python detect_events.py --input data/features.parquet --output data/events.parquet --debug
```

## 扩展开发

### 添加新事件类型

1. 在 `EventDetector` 类中添加新的事件检测方法
2. 在 `aggregate_events` 方法中更新事件分类
3. 更新配置参数

### 添加新标签策略

1. 在 `EventLabeler` 类中添加新的标签生成方法
2. 在 `generate_labels` 方法中添加策略选择
3. 更新命令行参数

### 集成机器学习模型

1. 准备训练数据
2. 训练模型
3. 在 `generate_ml_based_labels` 方法中集成模型

## 性能优化

### 数据处理优化

- 使用 Parquet 格式存储数据
- 批量处理大量数据
- 并行处理多个时间框架

### 计算优化

- 向量化计算
- 缓存中间结果
- 使用 NumPy 加速

## 联系和支持

如有问题或建议，请：
1. 查看测试报告
2. 检查日志输出
3. 验证数据格式
4. 调整参数配置

---

**注意**: 本系统仅供研究和学习使用，实际交易请谨慎评估风险。 