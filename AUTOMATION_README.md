# 微观策略自动化系统

## 概述

这是一个完整的微观结构策略自动化流水线，包含数据更新、标签生成、模型训练、回测和参数优化的全流程自动化。

## 系统架构

```
数据更新 → 标签生成 → 模型训练 → 回测 → 参数选择 → 配置更新
   ↓           ↓           ↓         ↓        ↓          ↓
data_pipeline → generate_labels → train_xgb → backtest → select_best → config/micro_best.yaml
```

## 核心组件

### 1. 数据管道 (`scripts/data_pipeline.py`)
- 检查最新数据文件
- 验证数据完整性
- 支持增量更新

### 2. 标签生成 (`generate_micro_labels.py`)
- 支持命令行参数：`--horizon`, `--alpha`, `--mode`, `--require-fill`
- 生成多时间窗口标签
- 输出带标签的特征文件

### 3. 模型训练 (`train_micro_xgb.py`)
- XGBoost 模型训练
- 支持 Optuna 超参数优化
- 输出训练好的模型文件

### 4. 回测引擎 (`micro_backtest_maker.py`)
- Maker 模式回测
- 支持自定义阈值和持仓时间
- 输出详细的回测结果 JSON

### 5. 参数选择 (`scripts/select_best.py`)
- 基于 Sharpe/收益/胜率选择最佳参数
- 输出 YAML 格式的配置文件

## 使用方法

### 本地测试
```bash
# 测试整个流水线
python scripts/test_automation.py

# 清理测试文件
python scripts/test_automation.py --cleanup
```

### 手动运行
```bash
# 1. 生成标签
python generate_micro_labels.py \
  --horizon 120 --alpha 0.3 --mode maker --require-fill \
  --output data/labels.parquet

# 2. 训练模型
python train_micro_xgb.py \
  --features data/labels.parquet \
  --model-out xgb_model.bin \
  --label-col label_h120_a0.3_maker

# 3. 执行回测
python micro_backtest_maker.py \
  --model xgb_model.bin \
  --features data/labels.parquet \
  --json-out backtest_results.json

# 4. 选择最佳参数
python scripts/select_best.py backtest_results.json --metric sharpe > best_params.yaml
```

### GitHub Actions 自动化

系统已配置 GitHub Actions 工作流 (`.github/workflows/microauto.yml`)：

- **触发方式**：
  - 手动触发 (`workflow_dispatch`)
  - 每日自动运行 (UTC 02:00)

- **执行步骤**：
  1. 安装依赖
  2. 生成标签
  3. 训练模型
  4. 执行回测
  5. 选择最佳参数
  6. 更新配置文件
  7. 提交到仓库

## 配置参数

### 标签参数
- `horizon`: 标签时间窗口 (默认: 120 步)
- `alpha`: 阈值参数 (默认: 0.3)
- `mode`: 标签模式 (maker/taker)
- `require_fill`: 是否需要填充验证

### 回测参数
- `threshold_long`: 做多阈值 (默认: 0.7)
- `threshold_short`: 做空阈值 (默认: 0.3)
- `holding_steps`: 持仓时间 (默认: 60 步)

### 优化指标
- `sharpe`: 夏普比率
- `total_return`: 总收益
- `win_rate`: 胜率

## 输出文件

### 中间文件
- `data/labels.parquet`: 带标签的特征文件
- `xgb_model.bin`: 训练好的 XGBoost 模型
- `backtest_results.json`: 回测结果

### 最终配置
- `config/micro_best.yaml`: 最佳参数配置

## 监控和告警

### GitHub Actions 日志
- 在 GitHub 仓库的 Actions 标签页查看运行日志
- 每次运行会生成 artifacts 包含所有结果文件

### 本地监控
```bash
# 查看最新配置
cat config/micro_best.yaml

# 查看回测结果
python -c "import json; data=json.load(open('backtest_results.json')); print(f'Sharpe: {data[0].get(\"sharpe\", 0):.3f}')"
```

## 扩展功能

### 1. 多资产支持
修改工作流添加 matrix 策略：
```yaml
strategy:
  matrix:
    asset: [ETH, BTC, SOL]
```

### 2. 多交易所支持
在回测脚本中添加交易所参数，支持跨交易所套利。

### 3. 实时监控
集成 Slack/Telegram 通知，实时推送策略表现。

### 4. 参数优化
使用 Optuna 进行更复杂的超参数搜索。

## 故障排除

### 常见问题

1. **数据文件不存在**
   - 检查 `data/` 目录下是否有特征文件
   - 运行 `python scripts/data_pipeline.py --check-only`

2. **模型训练失败**
   - 检查标签分布是否平衡
   - 确认特征文件格式正确

3. **回测结果异常**
   - 检查模型文件是否存在
   - 验证特征列名是否匹配

4. **GitHub Actions 失败**
   - 查看 Actions 日志
   - 检查依赖版本兼容性

### 调试模式
```bash
# 详细日志输出
python generate_micro_labels.py --horizon 120 --alpha 0.3 --verbose

# 单步调试
python -m pdb scripts/test_automation.py
```

## 性能优化

### 计算资源
- 当前配置适合 GitHub Actions 免费额度
- 如需更高性能，可迁移到 Coze 或自托管 Runner

### 并行处理
- 支持多资产并行回测
- 可配置 Optuna 并行试验

## 维护

### 定期任务
- 检查数据质量
- 更新依赖版本
- 备份重要结果

### 版本控制
- 配置文件变更会通过 Git 追踪
- 回测结果作为 artifacts 保存

---

**注意**: 这是一个实验性系统，请在生产环境中谨慎使用，并确保充分测试。 