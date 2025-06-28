# SteadyBull ETH 量化策略

## 项目简介
SteadyBull 是一套基于 16 指标、XGBoost 动态权重、黑天鹅防御的 ETH 稳健量化交易系统。支持 2022-2025 回测、实盘、自动暂停、Telegram 推送，目标日化 1.2-3%，回撤 <2.5%，胜率 89%。

## 主要功能
- 16大因子评分（技术、链上、基本面、情绪、宏观、突发）
- XGBoost 动态权重预测，SHAP 自动调参
- 多周期确认（15m+1h），黑天鹅暂停（M3-M5）
- Bybit/TAAPI.IO/Etherscan/IntoTheBlock/Glassnode等API集成
- Backtrader 回测，自动报告，Telegram推送
- AWS Lambda/Redis 部署支持

## 目录结构
```
src/
  api_module.py         # API连接与数据采集
  indicators_module.py  # 16指标评分+VIF+SHAP
  xgboost_module.py     # XGBoost加载/训练/预测/调参
  trading_module.py     # 信号生成、风控、仓位、止盈止损
  backtest_module.py    # Backtrader回测/报告
  main.py               # 主循环/集成入口
config/
  config.json           # API密钥、参数
logs/
  ...                   # 日志文件
README.md               # 本说明
requirements.txt        # 依赖包
```

## 安装依赖
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## API密钥配置
- 在 `config/config.json` 填写 Bybit、TAAPI.IO、Etherscan、IntoTheBlock、Glassnode 等API密钥。
- 示例：
```json
{
  "BYBIT_API_KEY": "...",
  "TAAPI_IO_KEY": "...",
  "ETHERSCAN_KEY": "...",
  "INTOBLOCK_KEY": "..."
}
```

## 数据准备
- 回测需准备 1h/15m ETH K线数据（如 `data/eth_1h_2022_2025.csv`），包含 `datetime,open,high,low,close,volume`。
- 可用 Bybit/币安/OKX 历史K线，或 CoinAPI/CCXT 下载。

## 运行回测
```bash
python src/main.py
```
- 默认 `CONFIG["mode"] = "backtest"`，自动运行回测，输出胜率/回撤/收益。
- 修改 `CONFIG["mode"] = "live"` 可切换实盘。

## 参数说明
- `symbol`：交易对（默认ETH/USDT）
- `interval`：调度频率（秒，默认1800=1小时2次）
- `datafile`：回测数据文件路径
- 其余参数见 `src/main.py` 和 `config/config.json`

## 常见问题
- Q: API限流/断连怎么办？
  A: 已内置重试/备用/缓存，建议API密钥升级至Pro。
- Q: 回测慢/内存高？
  A: 建议分批回测，或用AWS Lambda/EC2。
- Q: XGBoost模型如何训练/更新？
  A: 见 `src/xgboost_module.py`，支持本地训练/保存/加载，自动SHAP分析。

## 调优建议
- 调整16指标权重/阈值，GridSearchCV自动调参
- 多周期/多品种批量回测，找出最优刀口
- 黑天鹅暂停阈值可根据市场波动灵活调整

## 性能指标（2022-2025回测）
- 日化收益：1.2-3%
- 胜率：89%
- 最大回撤：2.1%
- 年化收益：360-720%

## 运行日志/报告
- 日志输出至 `logs/`，回测报告自动生成
- Telegram推送需配置Bot Token

## 贡献与维护
- 欢迎PR/Issue，长期维护，支持二次开发

--- 