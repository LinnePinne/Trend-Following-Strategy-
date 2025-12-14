import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ta
from math import sqrt
import os

df = pd.read_csv("LNKUSD_1H_2012-now.csv")

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
closed_equity = initial_equity
risk_pct = 0.005  # 0.5% per trade

open_trades = []     # list of dicts
closed_trades = []   # list of dicts

closed_equity_curve = []    # stores equity when trades close
idx_list = df.index.to_list()

closed_equity_curve.append({"Time": df.index[0], "Equity": closed_equity})
open_equity_curve = []

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
            closed_equity += pnl_r * (initial_equity * risk_pct)

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
                "Equity After": closed_equity
            })

            closed_equity_curve.append({"Time": ts, "Equity": closed_equity})
        else:
            still_open.append(t)

    # ===== Mark-to-market equity =====
    unrealized_pnl = 0.0
    price = row["close"]  # mark-to-market på close

    for t in open_trades:
        if t["direction"] == "LONG":
            r_open = (price - t["entry_price"]) / t["stop_distance"]
        else:  # SHORT
            r_open = (t["entry_price"] - price) / t["stop_distance"]

        # fast risk i kronor
        unrealized_pnl += r_open * (initial_equity * risk_pct)

    mtm_equity = closed_equity + unrealized_pnl

    open_equity_curve.append({
        "Time": ts,
        "Equity": mtm_equity
    })

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

open_eq_df = pd.DataFrame(open_equity_curve)
open_eq_df = open_eq_df.sort_values("Time").set_index("Time")
# om flera rader per timestamp: behåll sista
open_eq = open_eq_df["Equity"].groupby(level=0).last()

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
    print(f"Slut-equity:            {closed_equity:.2f}")
    print(f"Genomsnittlig R:        {trades_df['PnL R'].mean():.3f}")

    # ---------- Expectancy (R) ----------
    expectancy_r = trades_df["PnL R"].mean()

    # ---------- Profit Factor (i R-termer) ----------
    gross_profit_r = trades_df.loc[trades_df["PnL R"] > 0, "PnL R"].sum()
    gross_loss_r = -trades_df.loc[trades_df["PnL R"] < 0, "PnL R"].sum()  # positivt tal
    profit_factor_r = np.inf if gross_loss_r == 0 else gross_profit_r / gross_loss_r

    # ---------- Equity df (för max drawdown) ----------
    equity_df = pd.DataFrame(closed_equity_curve).copy()
    equity_df["Time"] = pd.to_datetime(equity_df["Time"])
    equity_df = equity_df.sort_values("Time").set_index("Time")

    # ---------- Max Drawdown ----------
    eq = equity_df["Equity"].astype(float)
    # Om flera rader har samma timestamp, behåll sista (dvs equity efter sista exit på den baren)
    eq = eq.groupby(level=0).last()

    roll_max = eq.cummax()
    dd = (eq / roll_max) - 1.0
    max_dd = dd.min()
    trough_time = dd.idxmin()
    peak_time = roll_max.loc[:trough_time].idxmax()
    peak_eq = float(eq.loc[peak_time])
    trough_eq = float(eq.loc[trough_time])

    # tid från peak till trough (själva fallet)
    dd_drop_duration = trough_time - peak_time

    # recovery: första gång equity tar sig tillbaka till peak-nivån
    recovery_time = eq.loc[trough_time:][eq.loc[trough_time:] >= peak_eq].index.min()
    dd_recovery_duration = (recovery_time - peak_time) if pd.notna(recovery_time) else None

    # ---------- Sharpe på trade returns ----------
    # Trade-return definieras som R * risk_pct (t.ex. 0.005 => 0.5% risk)
    # Detta är standard när du modellerar return per trade via initial risk.
    risk_pct = 0.005  # sätt samma som i din backtest
    trade_rets = trades_df["PnL R"].astype(float) * risk_pct  # per trade-return i decimalform

    mean_ret = trade_rets.mean()
    std_ret = trade_rets.std(ddof=1)

    sharpe_per_trade = np.nan if std_ret == 0 else mean_ret / std_ret

    # Annualisering: trades per år baserat på exit-tidsstämpelspannet
    exit_times = pd.to_datetime(trades_df["Exit Time"])
    years_span = (exit_times.max() - exit_times.min()).total_seconds() / (365.25 * 24 * 3600)
    trades_per_year = len(trades_df) / years_span if years_span > 0 else np.nan

    sharpe_annual = (
        np.nan if (std_ret == 0 or not np.isfinite(trades_per_year))
        else (mean_ret / std_ret) * np.sqrt(trades_per_year)
    )

    # ---------- Utskrift ----------
    print(f"Expectancy (R):          {expectancy_r:.3f}")
    print(f"Profit Factor (R):       {profit_factor_r:.3f}")
    print(f"Max Drawdown:            {max_dd * 100:.2f}%")
    print(f"  Peak:                  {peak_time}  Equity: {peak_eq:.2f}")
    print(f"  Trough:                {trough_time}  Equity: {trough_eq:.2f}")
    print(f"  DD drop duration:      {dd_drop_duration}")
    print(f"  DD recovery duration:  {dd_recovery_duration if dd_recovery_duration is not None else 'Not recovered'}")
    print(f"Sharpe (per trade):      {sharpe_per_trade:.3f}")
    print(f"Trades per year:         {trades_per_year:.1f}")
    print(f"Sharpe (annualiserad):   {sharpe_annual:.3f}")

    #-------open equity curve----
    roll_max = open_eq.cummax()
    dd_open = (open_eq / roll_max) - 1.0
    max_dd_open = dd_open.min()
    trough_time = dd_open.idxmin()
    peak_time = roll_max.loc[:trough_time].idxmax()
    peak_eq = float(open_eq.loc[peak_time])
    trough_eq = float(open_eq.loc[trough_time])
    print(f"Open Max Drawdown:       {max_dd_open * 100:.2f}%")
    print(f"  Peak:                 {peak_time}  Equity: {peak_eq:.2f}")
    print(f"  Trough:               {trough_time}  Equity: {trough_eq:.2f}")

    #--------Cagr, Calmar, Sharpe(daily returns)----
    start = eq.index.min()
    end = eq.index.max()
    years = (end - start).total_seconds() / (365.25 * 24 * 3600)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    calmar = np.nan if max_dd == 0 else cagr / abs(max_dd)
    print(f"CAGR:                    {cagr * 100:.2f}%")
    print(f"Calmar (CAGR/MaxDD):     {calmar:.2f}")
    eq_daily = eq.resample("D").last().ffill()
    rets_daily = eq_daily.pct_change().dropna()
    sharpe_daily = rets_daily.mean() / rets_daily.std(ddof=1) * np.sqrt(252)
    print(f"Sharpe (daglig, ann.):   {sharpe_daily:.3f}")

    # =========================
    # Equity curve plot
    # =========================
equity_df = pd.DataFrame(closed_equity_curve)
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

plt.figure(figsize=(12,5))
plt.plot(open_eq.index, open_eq.values, label="Open (MTM) Equity")
plt.plot(eq.index, eq.values, label="Closed Equity", alpha=0.7)
plt.legend()
plt.title("Closed vs Open (Mark-to-Market) Equity")
plt.grid(True)
plt.show()