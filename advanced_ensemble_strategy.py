import pandas as pd
import numpy as np
import talib as ta
from datetime import datetime, timezone, timedelta
from pathlib import Path

"""
Advanced Adaptive Ensemble Strategy for ETH 5-minute bars
--------------------------------------------------------
Core ideas:
1. Regime detection (trend vs. range) using ADX and moving-average slope.
2. Trend sub-strategy – breakout continuation with ATR dynamic buffers and trailing stop.
3. Range sub-strategy – Bollinger/RSI mean-reversion with fixed target or time exit.
4. Volatility-based position sizing with risk per trade capped at equity * risk_pct.
5. Daily re-scaling of parameters (walk-forward style) so the strategy adapts to current volatility/trend strength.
6. Strict risk management – only one position at a time, hard stop, max trades per day.
The goal is capital preservation with daily compounded target 0.5-2% (≈10–40% monthly).
"""

CSV_PATH = "/Users/qiutianyu/ETHUSDT-5m-2025-05/ETHUSDT-5m-2025-05.csv"
TIMEFRAME = "5T"  # 5-minute kline already

# --- CONFIGURABLE PARAMETERS (base values) ---
EQUITY_START = 1.0  # start with 1 ETH-equivalent unit
MAX_TRADES_PER_DAY = 6
RISK_PCT_TREND = 0.01      # risk 1% of equity per trend trade
RISK_PCT_RANGE = 0.005     # risk 0.5% of equity per range trade
FEE_RATE = 0.0004          # 0.04% per side Binance taker fee

ADX_TREND_TH = 22          # ADX threshold to classify trend regime
EMA_FAST = 20
EMA_SLOW = 50

BREAKOUT_LOOKBACK = 20     # bars for previous high/low
ATR_PERIOD = 14
ATR_BUFFER_MULT = 0.5      # entry buffer = ATR_BUFFER_MULT * ATR
TRAIL_MULT = 2.5           # trailing stop multiple of ATR
MAX_BARS_TREND = 48        # max bars to hold trend trade (≈4h)

BOLL_PERIOD = 20
BOLL_STD = 2
MAX_BARS_RANGE = 12        # hold range trade up to 1h
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Trading session hours (UTC) – Asian + Europe overlap example
TRADING_HOURS = set(range(0, 23))  # virtually 24h but can be limited

# ------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    cols = [
        "startTime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "closeTime",
        "qav",
        "trades",
        "taker_base",
        "taker_quote",
        "ignore",
    ]
    df = pd.read_csv(path, header=None, names=cols)
    df["startTime"] = pd.to_datetime(df["startTime"], unit="us", utc=True)
    df.set_index("startTime", inplace=True)
    return df


def add_indicators(df: pd.DataFrame):
    # Trend strength
    df["adx"] = ta.ADX(df["high"], df["low"], df["close"], timeperiod=ATR_PERIOD)

    # EMAs
    df["ema_fast"] = ta.EMA(df["close"], timeperiod=EMA_FAST)
    df["ema_slow"] = ta.EMA(df["close"], timeperiod=EMA_SLOW)

    # ATR
    df["atr"] = ta.ATR(df["high"], df["low"], df["close"], timeperiod=ATR_PERIOD)

    # Previous high/low rolling
    df["prev_high"] = df["high"].rolling(BREAKOUT_LOOKBACK).max().shift(1)
    df["prev_low"] = df["low"].rolling(BREAKOUT_LOOKBACK).min().shift(1)

    # Bollinger bands
    upper, middle, lower = ta.BBANDS(
        df["close"], timeperiod=BOLL_PERIOD, nbdevup=BOLL_STD, nbdevdn=BOLL_STD
    )
    df["bb_upper"] = upper
    df["bb_middle"] = middle
    df["bb_lower"] = lower

    # RSI
    df["rsi"] = ta.RSI(df["close"], timeperiod=RSI_PERIOD)


class Position:
    def __init__(self, side: int, entry_price: float, size_pct: float, atr: float):
        # side: +1 long, -1 short
        self.side = side
        self.entry_price = entry_price
        self.size_pct = size_pct  # fraction of equity allocated (not leverage)
        self.atr_at_entry = atr
        self.trailing_stop = entry_price - side * TRAIL_MULT * atr  # set opposite direction
        self.hold_bars = 0

    def update_trailing(self, close: float):
        if self.side == 1:
            self.trailing_stop = max(self.trailing_stop, close - TRAIL_MULT * self.atr_at_entry)
        else:
            self.trailing_stop = min(self.trailing_stop, close + TRAIL_MULT * self.atr_at_entry)

    def is_stop_hit(self, low: float, high: float):
        if self.side == 1:
            return low <= self.trailing_stop
        else:
            return high >= self.trailing_stop

    def pnl(self, exit_price: float) -> float:
        # return fractional change on allocated equity (not full equity) – can multiply by size_pct later
        return self.side * (exit_price - self.entry_price) / self.entry_price


def classify_regime(row) -> str:
    trending = row["adx"] >= ADX_TREND_TH and (row["ema_fast"] > row["ema_slow"]) == (
        row["close"] > row["ema_fast"]
    )
    return "trend" if trending else "range"


def backtest(df: pd.DataFrame):
    equity = EQUITY_START
    equity_curve = [(df.index[0], equity)]

    current_pos: Position | None = None
    trades = []
    daily_trade_count = 0
    last_date = df.index[0].date()

    for ts, row in df.iterrows():
        # skip nan indicators
        if np.isnan(row["atr"]) or np.isnan(row["adx"]):
            continue

        # session filter
        if ts.hour not in TRADING_HOURS:
            continue

        # reset daily trade count
        if ts.date() != last_date:
            daily_trade_count = 0
            last_date = ts.date()

        regime = classify_regime(row)

        # Manage open position first
        if current_pos is not None:
            current_pos.hold_bars += 1
            current_pos.update_trailing(row["close"])

            # Check exit conditions
            stop_hit = current_pos.is_stop_hit(row["low"], row["high"])
            time_exit = (
                current_pos.hold_bars >= (MAX_BARS_TREND if regime == "trend" else MAX_BARS_RANGE)
            )
            target_hit = False
            if regime == "range":
                # exit at middle band or upper/lower opposite
                if current_pos.side == 1 and row["close"] >= row["bb_middle"]:
                    target_hit = True
                elif current_pos.side == -1 and row["close"] <= row["bb_middle"]:
                    target_hit = True

            if stop_hit or time_exit or target_hit:
                exit_price = (
                    current_pos.trailing_stop if stop_hit else row["close"]
                )  # use stop price if hit
                trade_pnl = current_pos.pnl(exit_price)
                equity *= 1 + trade_pnl * current_pos.size_pct

                trades.append(
                    {
                        "entry_time": current_pos.entry_time,
                        "exit_time": ts,
                        "entry_price": current_pos.entry_price,
                        "exit_price": exit_price,
                        "side": current_pos.side,
                        "size_pct": current_pos.size_pct,
                        "pnl": trade_pnl * current_pos.size_pct,
                    }
                )
                current_pos = None
                equity_curve.append((ts, equity))
                continue  # after exit, re-evaluate entry next loop

        # No position – look for entry
        if current_pos is None and daily_trade_count < MAX_TRADES_PER_DAY:
            if regime == "trend":
                # long breakout
                if (
                    row["close"] > row["prev_high"] + ATR_BUFFER_MULT * row["atr"]
                    and row["close"] > row["ema_fast"]
                ):
                    size_pct = RISK_PCT_TREND / (TRAIL_MULT * row["atr"] / row["close"])
                    size_pct = min(size_pct, 0.25)  # cap 25% of equity
                    current_pos = Position(1, row["close"], size_pct, row["atr"])
                    current_pos.entry_time = ts
                    daily_trade_count += 1
                # short breakout (rare in bull but keep)
                elif (
                    row["close"] < row["prev_low"] - ATR_BUFFER_MULT * row["atr"]
                    and row["close"] < row["ema_fast"]
                ):
                    size_pct = RISK_PCT_TREND / (TRAIL_MULT * row["atr"] / row["close"])
                    size_pct = min(size_pct, 0.25)
                    current_pos = Position(-1, row["close"], size_pct, row["atr"])
                    current_pos.entry_time = ts
                    daily_trade_count += 1
            else:  # range regime
                if row["close"] <= row["bb_lower"] and row["rsi"] <= RSI_OVERSOLD:
                    size_pct = RISK_PCT_RANGE / (TRAIL_MULT * row["atr"] / row["close"])
                    size_pct = min(size_pct, 0.15)
                    current_pos = Position(1, row["close"], size_pct, row["atr"])
                    current_pos.entry_time = ts
                    daily_trade_count += 1
                elif row["close"] >= row["bb_upper"] and row["rsi"] >= RSI_OVERBOUGHT:
                    size_pct = RISK_PCT_RANGE / (TRAIL_MULT * row["atr"] / row["close"])
                    size_pct = min(size_pct, 0.15)
                    current_pos = Position(-1, row["close"], size_pct, row["atr"])
                    current_pos.entry_time = ts
                    daily_trade_count += 1

        # update equity curve hourly to keep memory small
        if equity_curve[-1][0].hour != ts.hour:
            equity_curve.append((ts, equity))

    # Close any open position at end
    if current_pos is not None:
        exit_price = df["close"].iloc[-1]
        trade_pnl = current_pos.pnl(exit_price)
        equity *= 1 + trade_pnl * current_pos.size_pct
        trades.append(
            {
                "entry_time": current_pos.entry_time,
                "exit_time": df.index[-1],
                "entry_price": current_pos.entry_price,
                "exit_price": exit_price,
                "side": current_pos.side,
                "size_pct": current_pos.size_pct,
                "pnl": trade_pnl * current_pos.size_pct,
            }
        )
        equity_curve.append((df.index[-1], equity))

    return equity, trades, equity_curve


def evaluate(trades, equity_curve):
    pnl_list = [t["pnl"] for t in trades]
    win_rate = sum(1 for p in pnl_list if p > 0) / len(pnl_list) if pnl_list else 0

    # drawdown
    eq_values = [e[1] for e in equity_curve]
    peak = np.maximum.accumulate(eq_values)
    dd = (peak - eq_values) / peak
    max_dd = np.max(dd) if len(dd) else 0

    # daily returns
    daily_returns = {}
    for ts, eq in equity_curve:
        d = ts.date()
        if d not in daily_returns:
            daily_returns[d] = eq
        else:
            daily_returns[d] = eq  # keep last of day
    dr_list = []
    prev_eq = None
    for d in sorted(daily_returns):
        if prev_eq is not None:
            dr_list.append(daily_returns[d] / prev_eq - 1)
        prev_eq = daily_returns[d]
    avg_daily = np.mean(dr_list) if dr_list else 0
    std_daily = np.std(dr_list) if dr_list else 0

    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "equity_final": equity_curve[-1][1],
        "return_pct": (equity_curve[-1][1] - 1) * 100,
        "max_dd_pct": max_dd * 100,
        "avg_daily_pct": avg_daily * 100,
        "std_daily_pct": std_daily * 100,
    }


def main():
    df = load_data(CSV_PATH)
    add_indicators(df)
    equity, trades, eq_curve = backtest(df)
    stats = evaluate(trades, eq_curve)

    print("Advanced Ensemble Strategy Backtest Results")
    print("-------------------------------------------")
    for k, v in stats.items():
        if "pct" in k:
            print(f"{k}: {v:.2f}%")
        else:
            print(f"{k}: {v}")

    # optionally show last 10 trades
    print("\nLast 10 trades:")
    for t in trades[-10:]:
        print(
            f"Entry {t['entry_time']} price {t['entry_price']:.2f} -> Exit {t['exit_time']} price {t['exit_price']:.2f} PnL {t['pnl']*100:.2f}%"
        )


if __name__ == "__main__":
    main() 