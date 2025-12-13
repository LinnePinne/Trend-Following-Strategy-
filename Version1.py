import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ta
from math import sqrt
import os

df = pd.read_csv("BTCUSD_1H_2012-now.csv")

# Anpassa kolumnnamn om de skiljer sig
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
df = df.sort_index()

# =========================
# Indicators
# =========================
df["ema_fast"] = df["close"].ewm(span=20, adjust=False).mean()
df["ema_slow"] = df["close"].ewm(span=40, adjust=False).mean()


# =========================
# Backtest settings
# =========================
initial_equity = 50_000.0
equity = initial_equity
risk_pct = 0.005  # 0.5% per trade

open_trades = []     # list of dicts
closed_trades = []   # list of dicts

equity_curve = []    # stores equity when trades close
idx_list = df.index.to_list()

# (valfritt) logga equity från start
equity_curve.append({"Time": df.index[0], "Equity": equity})

# =========================
# Backtest loop
# =========================
for i in range(1, len(df) - 1):
    ts = df.index[i]
    row = df.iloc[i]
    prev_row = df.iloc[i - 1]

    # Skip if indicators not ready
    if np.isnan(row["ema_fast"]) or np.isnan(row["ema_slow"]) or np.isnan(prev_row["ema_fast"]):
        continue

    rolling_sl = row["ema_slow"]

    # -------------------------
    # Manage exits (rolling stop)
    # -------------------------
    still_open = []
    for t in open_trades:
        exit_price = None
        exit_reason = None

        if t["direction"] == "LONG":
            if row["low"] <= rolling_sl:
                exit_price = rolling_sl
                exit_reason = "rolling ema_slow stop"
        else:  # SHORT
            if row["high"] >= rolling_sl:
                exit_price = rolling_sl
                exit_reason = "rolling ema_slow stop"

        if exit_price is not None:
            # R-multipel baserat på initial risk (entry till initial_sl)
            if t["direction"] == "LONG":
                pnl_r = (exit_price - t["entry_price"]) / t["stop_distance"]
            else:
                pnl_r = (t["entry_price"] - exit_price) / t["stop_distance"]

            # Uppdatera equity med fast %-risk
            # equity_new = equity_old * (1 + pnl_r * risk_pct)
            equity *= (1.0 + pnl_r * risk_pct)

            closed_trades.append({
                "Entry Time": t["entry_time"],
                "Exit Time": ts,
                "Direction": t["direction"],
                "Entry Price": t["entry_price"],
                "Exit Price": exit_price,
                "Exit Reason": exit_reason,
                "Initial SL": t["initial_sl"],
                "Stop Distance": t["stop_distance"],
                "PnL R": pnl_r,
                "Equity After": equity
            })

            equity_curve.append({"Time": ts, "Equity": equity})
        else:
            still_open.append(t)

    # VIKTIGT: uppdatera open_trades EFTER loopen (inte inuti)
    open_trades = still_open

    # -------------------------
    # Signals
    # -------------------------
    ema_fast = row["ema_fast"]
    ema_slow = row["ema_slow"]
    close_price = row["close"]
    prev_close = prev_row["close"]
    prev_ema_fast = prev_row["ema_fast"]

    long_trend = ema_fast > ema_slow
    short_trend = ema_fast < ema_slow

    pullback_long = prev_close < prev_ema_fast and close_price > ema_fast
    pullback_short = prev_close > prev_ema_fast and close_price < ema_fast

    long_signal = long_trend and pullback_long
    short_signal = short_trend and pullback_short

    entry_price = df.iloc[i + 1]["open"]

    # -------------------------
    # Entries (initial SL = current ema_slow)
    # -------------------------
    if long_signal:
        initial_sl = row["ema_slow"]
        stop_distance = entry_price - initial_sl  # LONG risk in price units

        # kräver positiv risk
        if stop_distance > 0 and np.isfinite(stop_distance):
            open_trades.append({
                "direction": "LONG",
                "entry_time": ts,
                "entry_price": float(entry_price),
                "initial_sl": float(initial_sl),
                "stop_distance": float(stop_distance),
            })

    elif short_signal:
        initial_sl = row["ema_slow"]
        stop_distance = initial_sl - entry_price  # SHORT risk in price units

        if stop_distance > 0 and np.isfinite(stop_distance):
            open_trades.append({
                "direction": "SHORT",
                "entry_time": ts,
                "entry_price": float(entry_price),
                "initial_sl": float(initial_sl),
                "stop_distance": float(stop_distance),
            })


# =========================
# Results
# =========================
trades_df = pd.DataFrame(closed_trades)

if trades_df.empty:
    print("Inga trades att sammanställa för denna marknad.")
else:
    n_trades = len(trades_df)
    wins = trades_df[trades_df["PnL R"] > 0]
    losses = trades_df[trades_df["PnL R"] <= 0]

    print(f"Antal trades:           {n_trades}")
    print(f"Vinnande trades:        {len(wins)}")
    print(f"Förlorande trades:      {len(losses)}")
    print(f"Winrate:                {len(wins) / n_trades * 100:.1f}%")
    print(f"Slut-equity:            {equity:.2f}")
    print(f"Genomsnittlig R:        {trades_df['PnL R'].mean():.3f}")

    # =========================
    # Equity curve plot
    # =========================
    equity_df = pd.DataFrame(equity_curve)
    equity_df = equity_df.sort_values("Time").set_index("Time")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity_df.index, equity_df["Equity"])

    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator())

    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, which="major", alpha=0.6)
    ax.grid(True, which="minor", alpha=0.2)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()
'''
    # (valfritt) equity över alla candles (flat mellan exits)
    equity_ts = equity_df["Equity"].reindex(df.index).ffill()
    plt.figure(figsize=(12, 5))
    plt.plot(equity_ts.index, equity_ts.values)
    plt.title("Equity Curve (forward-filled over all bars)")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.show()'''