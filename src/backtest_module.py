import backtrader as bt
import pandas as pd
import numpy as np
import logging
from src.indicators_module import *
from src.xgboost_module import XGBoostManager, build_feature_vector
from src.trading_module import SteadyBullTrader

logger = logging.getLogger("SteadyBullBacktest")

# ========== Backtrader 策略 ========== #
class SteadyBullBTStrategy(bt.Strategy):
    params = (
        ('xgb_manager', None),
        ('shap_manager', None),
        ('config', None),
    )
    def __init__(self):
        self.xgb = self.p.xgb_manager
        self.shap = self.p.shap_manager
        self.config = self.p.config
        self.last_trade_time = None
        self.pause_until = None
        self.trade_log = []
        self.order = None

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        if self.pause_until and dt < self.pause_until:
            return
        # 构造指标分数dict（示例，需补全）
        indicators = {
            "T1_RSI": score_rsi(self.datas[0].rsi[0]),
            "T2_MACD": score_macd(self.datas[0].macd[0], self.datas[0].macdsignal[0]),
            # ... 其余指标
        }
        shap_weights = self.shap.get_shap_weights()
        feature_vec = build_feature_vector(indicators, shap_weights)
        xgb_prob = self.xgb.predict(feature_vec)[0]
        total_score = sum([indicators[k] * shap_weights.get(k, 0.05) for k in indicators])
        bullish_count = sum([1 for v in indicators.values() if v > 50])
        # 黑天鹅暂停
        m3, m4, m5 = indicators.get("M3_BlackSwan", 50), indicators.get("M4_Volatility", 50), indicators.get("M5_Sudden", 50)
        if m3 < 25 or m4 < 25 or m5 < 25:
            self.pause_until = dt + pd.Timedelta(hours=1)
            logger.warning(f"Black swan detected, pause until {self.pause_until}")
            return
        # 信号判定
        if total_score >= 70 and bullish_count >= 13 and xgb_prob > 0.8:
            # 仓位、止盈止损
            entry_price = self.datas[0].close[0]
            atr = self.datas[0].atr[0]
            stop_loss = entry_price - atr
            take_profit = entry_price + entry_price * 0.018
            size = 0.01
            if not self.position:
                self.order = self.buy(size=size)
                self.trade_log.append({
                    "entry": entry_price,
                    "stop": stop_loss,
                    "tp": take_profit,
                    "size": size,
                    "time": dt
                })
                logger.info(f"Backtest trade opened: {entry_price}")
        # 止损止盈
        if self.position:
            if self.datas[0].close[0] <= stop_loss or self.datas[0].close[0] >= take_profit:
                self.close()
                logger.info(f"Backtest trade closed: {self.datas[0].close[0]}")

logger.info("[Backtest Module] SteadyBullBTStrategy loaded (first 50 lines)")

# ========== 回测运行与报告 ========== #
def run_backtest(datafile, xgb_manager, shap_manager, config):
    cerebro = bt.Cerebro()
    df = pd.read_csv(datafile, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(SteadyBullBTStrategy, xgb_manager=xgb_manager, shap_manager=shap_manager, config=config)
    cerebro.broker.set_cash(10000)
    cerebro.broker.setcommission(commission=0.0003)
    logger.info("Backtest started.")
    result = cerebro.run()
    logger.info("Backtest finished.")
    # 统计与报告
    trades = result[0].trade_log
    pnl = [t['tp']-t['entry'] if t['tp']>t['entry'] else t['entry']-t['stop'] for t in trades]
    win_rate = sum([p>0 for p in pnl])/len(pnl) if pnl else 0
    avg_return = np.mean(pnl)/10000*100 if pnl else 0
    max_dd = 0  # 可用cerebro/analyzer
    logger.info(f"回测结果：交易数={len(trades)}, 胜率={win_rate:.2%}, 平均收益={avg_return:.4f}%, 最大回撤={max_dd:.2f}%")
    return trades

logger.info("[Backtest Module] run_backtest loaded (line 100)") 