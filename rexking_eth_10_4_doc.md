# RexKing ETH 10.4 策略文档

**时间**: 2025年6月23日 17:15 EDT (05:15 HKT 6/24)  
**ETH价格**: 2554 USD  
**策略版本**: 10.4 (优化版)

## 📊 策略概览

RexKing ETH 10.4是一个多时间框架ETHUSDT交易策略，整合了4H、1H、15m技术指标和W1鲸鱼转账数据，采用动态止盈止损和Kelly仓位管理。

## 🎯 核心目标

- **信号频率**: 日0.3-0.5笔 (年化165-205条)
- **胜率目标**: 50-55%
- **年化收益**: 30-50%
- **最大回撤**: <10%
- **实盘资金**: $280-$700

## 📈 信号条件

### 4H主信号 (ADX趋势强度过滤)
```python
cond_4h = (
    (obv > obv_ma14) &           # OBV > 14期均线
    (ema20 > ema60) &            # 趋势向上
    (atr > 0.05% * close) &      # ATR > 0.05%
    (bb > 0.5%) &                # 布林带宽度 > 0.5%
    (close > ema20) &            # 价格在EMA20之上
    (adx > 15)                   # ADX趋势强度 > 15
)
```

### 1H Funding信号
```python
cond_1h = (funding < 0.005%)     # 资金费率 < 0.005%
```

### 15m突破+量能信号
```python
cond_15m = (
    breakout_15m &               # 突破前4H高点
    (volmean_15m.rolling(2).mean() > 0.03 * volume_4h_ma)  # 连续2根放量
)
```

### W1鲸鱼转账信号 (Dune+Etherscan)
```python
cond_w1 = (
    (w1_value > 1000) &          # 转账金额 > 1000 ETH
    (w1_zscore > 0.5) &          # Z-score > 0.5
    (w1_signal_rolling > 0)      # 12h滚动窗口信号
)
```

## 🎛️ 动态止盈止损

### 牛市止盈止损
```python
if atr_pct > 0.01:  # 波动率 > 1%
    tp_trigger = entry_price + 1.5 * atr  # 1.5×ATR止盈
    stop = entry_price - 0.7 * atr        # 0.7×ATR止损
```

### 震荡市止盈止损
```python
else:  # 波动率 < 1%
    tp_trigger = entry_price + 3 * atr    # 3×ATR止盈
    stop = entry_price - 0.7 * atr        # 0.7×ATR止损
```

### 跟踪止损
```python
trail_trigger = price - 2.5 * atr  # 2.5×ATR跟踪止损
```

## 💰 仓位管理

### Kelly仓位计算
```python
kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
pos_size = min(pos_size * kelly_fraction, MAX_POSITION_SIZE)
```

### 风险控制
- **单笔风险**: 0.5-1.5% ($1.4-$10.5)
- **最大仓位**: $28-$70
- **手续费**: 0.075% (VIP0)

## 📊 最新回测结果 (2023-2024)

### 信号统计
- **4H信号**: 1269条
- **Funding信号**: 3202条  
- **15m突破信号**: 1774条
- **W1信号**: 6条 (Dune+Etherscan整合)
- **联合信号**: 410条
- **年化信号频率**: 205条/年 (日0.56笔，达标！)

### 交易表现
- **总交易**: 177笔
- **胜率**: 39.0% (目标50-55%，需优化)
- **年化收益**: -0.47% (目标30-50%，需提升)
- **最大回撤**: 2.41% (安全！)
- **总盈亏**: -$6.55

### Walk-Forward验证
- **平均胜率**: 35.19%
- **平均Profit Factor**: 2.03
- **平均最大回撤**: 0.01%

### 蒙特卡洛模拟 (1000次)
- **平均总收益**: 0.08%
- **最大回撤 < 10%**: 100%
- **Profit Factor > 1.3**: 0%

## 🔧 W1数据集成

### Dune Analytics API
- **Transactions端点**: `/v1/evm/transactions/{address}`
- **Activity端点**: `/v1/evm/activity/{address}`
- **API密钥**: `sim_xZvnjKWCFpvVMhPAKK4idopF19hShv3f`
- **数据范围**: 2023-2025
- **CEX地址**: 15个主要交易所地址

### Etherscan API
- **API密钥**: `CM56ZD9KTV8K93U8EXP8P4E1CBIEJ1P5`
- **数据范围**: 2023-2025
- **过滤条件**: >1000 ETH转账

### 数据整合结果
- **原始W1信号**: 6条
- **滚动W1信号**: 17条
- **日化W1信号频率**: 0.00条/天 (需扩展地址)

## 🚀 实盘指引

### 资金配置
- **小资金**: $280-$700
- **单笔风险**: 0.5-1.5%
- **预期收益**: $2-$8/笔
- **月交易频率**: 3-6笔

### 风险控制
- **5分钟跌幅 > 3%**: 立即清仓
- **连续亏损3笔**: 暂停交易
- **日亏损 > 5%**: 停止当日交易

### 实盘注意事项
1. **滑点控制**: 使用限价单，避免市价单
2. **流动性**: 选择高流动性时段交易
3. **监控**: 实时监控W1信号和资金费率
4. **备份**: 准备备用API密钥

## 📈 优化方向

### 短期优化 (1-2周)
1. **胜率提升**: 调整ADX阈值至20，加强趋势过滤
2. **W1扩展**: 增加更多CEX地址，提升信号频率
3. **止盈优化**: 根据市场波动率动态调整止盈倍数

### 中期优化 (1个月)
1. **机器学习**: 集成XGBoost模型预测信号质量
2. **多币种**: 扩展到BTC、BNB等主流币种
3. **实盘测试**: 小资金验证策略稳定性

### 长期规划 (3个月)
1. **策略组合**: 开发多策略组合管理系统
2. **风险管理**: 实现动态风险预算分配
3. **自动化**: 构建完整的自动化交易系统

## 📋 技术栈

### 数据处理
- **Python**: pandas, numpy, scipy
- **数据库**: SQLite (缓存), CSV (数据存储)
- **API**: Dune Analytics, Etherscan, Binance

### 回测框架
- **Walk-Forward**: 滚动窗口验证
- **蒙特卡洛**: 1000次随机模拟
- **风险指标**: Sharpe Ratio, Sortino Ratio, Calmar Ratio

### 部署环境
- **操作系统**: macOS/Linux
- **Python版本**: 3.8+
- **依赖管理**: pip, requirements.txt

## 📞 联系方式

- **策略维护**: Cursor AI Assistant
- **数据更新**: 每日自动更新
- **问题反馈**: 通过GitHub Issues

---
