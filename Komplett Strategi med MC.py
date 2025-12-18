import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ta
from math import sqrt
import os
from scipy.optimize import minimize

# =========================
# Data loading
# =========================
INITIAL_EQUITY = 50_000.0
RISK_PCT = 0.005

EMA_FAST = 20
EMA_SLOW = 40

commission_roundturn = 0.00065   # entry + exit (round turn)
commission_side = commission_roundturn / 2

TP1_R = 20.0
TP1_FRACTION = 0.5  # 50% stängs vid +1R

# =========================
# Data loading
# =========================
def load_market(path, symbol):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # EMA på trading-TF
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    # =========================
    # HTF ADX regime filter
    # =========================
    df_htf = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

    adx = ta.trend.ADXIndicator(
        high=df_htf["high"],
        low=df_htf["low"],
        close=df_htf["close"],
        window=14
    )

    df_htf["adx"] = adx.adx()

    # forward-fill HTF ADX tillbaka till LTF
    df["adx_htf"] = df_htf["adx"].reindex(df.index, method="ffill")

    df["symbol"] = symbol
    return df

btc = load_market("BTCUSD_1H_2012-now.csv", "BTC")
eth = load_market("ETHUSD_1H_2012-now.csv", "ETH")

# =========================
# Sync timeline
# =========================
common_index = btc.index.intersection(eth.index)
btc = btc.loc[common_index]
eth = eth.loc[common_index]

markets = {
    "BTC": btc,
    "ETH": eth
}
index = common_index

# =========================
# Portfolio state
# =========================
closed_equity = INITIAL_EQUITY
open_trades = []
closed_trades = []

closed_equity_curve = [{"Time": index[0], "Equity": closed_equity}]
open_equity_curve = []

def cost_r_side(stop_distance, entry_price):
    stop_pct = stop_distance / entry_price
    if not np.isfinite(stop_pct) or stop_pct <= 0:
        return 0.0
    return commission_side / stop_pct

# =========================
# Backtest loop
# =========================
for i in range(1, len(index) - 1):
    ts = index[i]

    # -------- EXIT LOGIC (rolling ema_slow stop) --------
    still_open = []

    for t in open_trades:
        df = markets[t["symbol"]]
        row = df.iloc[i]
        if t.get("post_tp1", False):
            rolling_sl = row["ema_fast"]  # efter TP1
        else:
            rolling_sl = row["ema_slow"]  # före TP1

        exit_price = None
        exit_reason = None
        # =========================
        # Partial TP1: 50% at +1R
        # =========================
        if (not t["tp1_done"]) and (t["size"] > 0):
            if t["direction"] == "LONG":
                tp_price = t["entry_price"] + TP1_R * t["stop_distance"]
                hit = row["high"] >= tp_price
            else:  # SHORT
                tp_price = t["entry_price"] - TP1_R * t["stop_distance"]
                hit = row["low"] <= tp_price

            if hit:
                # TP1 gross R is TP1_R
                gross_r_tp1 = TP1_R

                stop_pct = t["stop_distance"] / t["entry_price"]
                exit_cost_r_tp1 = cost_r_side(t["stop_distance"], t["entry_price"])
                pnl_r_tp1 = gross_r_tp1 - exit_cost_r_tp1

                frac = min(TP1_FRACTION, t["size"])
                r_contrib = pnl_r_tp1 * frac

                # uppdatera equity (cash)
                closed_equity += r_contrib * (INITIAL_EQUITY * t["risk_pct"])

                # bokför partialen på traden
                t["realized_r"] += r_contrib
                t["size"] -= frac
                t["tp1_done"] = True
                t["post_tp1"] = True

                closed_equity_curve.append({"Time": ts, "Equity": closed_equity})

        if t["direction"] == "LONG":
            if row["low"] <= rolling_sl:
                exit_price = rolling_sl
                exit_reason = "rolling ema_slow stop"
        else:
            if row["high"] >= rolling_sl:
                exit_price = rolling_sl
                exit_reason = "rolling ema_slow stop"

        if exit_price is not None:
            if t["direction"] == "LONG":
                gross_r = (exit_price - t["entry_price"]) / t["stop_distance"]

                stop_pct = t["stop_distance"] / t["entry_price"]  # stop i procent
                exit_cost_r = cost_r_side(t["stop_distance"], t["entry_price"])
                pnl_r = gross_r - exit_cost_r

            else:
                gross_r = (t["entry_price"] - exit_price) / t["stop_distance"]

                stop_pct = t["stop_distance"] / t["entry_price"]
                exit_cost_r = cost_r_side(t["stop_distance"], t["entry_price"])
                pnl_r = gross_r - exit_cost_r

            # remaining fraction that exits now
            frac = t.get("size", 1.0)

            # cash update only for remaining fraction
            closed_equity += (pnl_r * frac) * (INITIAL_EQUITY * t["risk_pct"])

            # TOTAL R for this entry = realized partial R + final R contribution
            total_r = t.get("realized_r", 0.0) + (pnl_r * frac)

            closed_trades.append({
                "Symbol": t["symbol"],
                "Entry Time": t["entry_time"],
                "Exit Time": ts,
                "Direction": t["direction"],
                "Entry Price": t["entry_price"],
                "Exit Price": float(exit_price),
                "Exit Reason": exit_reason,
                "Stop Distance": t["stop_distance"],
                "PnL R": float(total_r),  # <-- VIKTIGT: total per entry
                "Risk Pct": float(t["risk_pct"]),
                "Equity After": float(closed_equity),
            })
            closed_equity_curve.append({"Time": ts, "Equity": closed_equity})
        else:
            still_open.append(t)

    open_trades = still_open

    # -------- MARK TO MARKET (open equity each bar) --------
    unrealized = 0.0
    for t in open_trades:
        df = markets[t["symbol"]]
        price = df.iloc[i]["close"]

        if t["direction"] == "LONG":
            r_open = (price - t["entry_price"]) / t["stop_distance"]
        else:
            r_open = (t["entry_price"] - price) / t["stop_distance"]

        unrealized += (r_open * t.get("size", 1.0)) * (INITIAL_EQUITY * t["risk_pct"])

    open_equity_curve.append({"Time": ts, "Equity": closed_equity + unrealized})

    # -------- ENTRY LOGIC --------
    for symbol, df in markets.items():
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if np.isnan(row["ema_fast"]) or np.isnan(row["ema_slow"]) or np.isnan(prev["ema_fast"]):
            continue

        long_trend = row["ema_fast"] > row["ema_slow"]
        short_trend = row["ema_fast"] < row["ema_slow"]

        pullback_long = prev["close"] < prev["ema_fast"] and row["close"] > row["ema_fast"]
        pullback_short = prev["close"] > prev["ema_fast"] and row["close"] < row["ema_fast"]

        # =========================
        # Regime filter
        # =========================
        if row["adx_htf"] < 40:
            continue
        long_signal = long_trend and pullback_long
        short_signal = short_trend and pullback_short

        entry_price = df.iloc[i + 1]["open"]

        if long_signal:
            sl = row["ema_slow"]
            stop_dist = entry_price - sl
            if stop_dist > 0 and np.isfinite(stop_dist):
                entry_cost_r = cost_r_side(stop_dist, entry_price)
                closed_equity -= entry_cost_r * (INITIAL_EQUITY * RISK_PCT)
                closed_equity_curve.append({"Time": ts, "Equity": closed_equity})
                open_trades.append({
                    "symbol": symbol,
                    "direction": "LONG",
                    "entry_time": ts,
                    "entry_price": float(entry_price),
                    "stop_distance": float(stop_dist),
                    "risk_pct": float(RISK_PCT),
                    "size": 1.0,
                    "tp1_done": False,
                    "realized_r": -entry_cost_r,
                    "post_tp1": False,
                })


        elif short_signal:
            sl = row["ema_slow"]
            stop_dist = sl - entry_price
            if stop_dist > 0 and np.isfinite(stop_dist):
                entry_cost_r = cost_r_side(stop_dist, entry_price)
                closed_equity -= entry_cost_r * (INITIAL_EQUITY * RISK_PCT)
                closed_equity_curve.append({"Time": ts, "Equity": closed_equity})
                open_trades.append({
                    "symbol": symbol,
                    "direction": "SHORT",
                    "entry_time": ts,
                    "entry_price": float(entry_price),
                    "stop_distance": float(stop_dist),
                    "risk_pct": float(RISK_PCT),
                    "size": 1.0,
                    "tp1_done": False,
                    "realized_r": -entry_cost_r,
                    "post_tp1": False,
                })

open_eq_df = pd.DataFrame(open_equity_curve)
open_eq_df = open_eq_df.sort_values("Time").set_index("Time")
# om flera rader per timestamp: behåll sista
open_eq = open_eq_df["Equity"].groupby(level=0).last()

# Convert closed_equity_curve to DataFrame and prepare equity series (eq)
equity_df = pd.DataFrame(closed_equity_curve)
equity_df = equity_df.sort_values("Time").set_index("Time")
eq = equity_df["Equity"].astype(float)
# Om flera rader har samma timestamp, behåll sista (dvs equity efter sista exit på den baren)
eq = eq.groupby(level=0).last()

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

    # ---------- Max Drawdown ----------
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
    print(
        f"  DD recovery duration:  {dd_recovery_duration if dd_recovery_duration is not None else 'Not recovered'}")
    print(f"Sharpe (per trade):      {sharpe_per_trade:.3f}")
    print(f"Trades per year:         {trades_per_year:.1f}")
    print(f"Sharpe (annualiserad):   {sharpe_annual:.3f}")

    # -------open equity curve----
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

    # --------Cagr, Calmar, Sharpe(daily returns)----
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

plt.figure(figsize=(12, 5))
plt.plot(open_eq.index, open_eq.values, label="Open (MTM) Equity")
plt.plot(eq.index, eq.values, label="Closed Equity", alpha=0.7)
plt.legend()
plt.title("Closed vs Open (Mark-to-Market) Equity")
plt.grid(True)
plt.show()


# =========================
# Equity per symbol, ERC
# =========================
symbol_equity = {}

for symbol in trades_df["Symbol"].unique():
    df_sym = trades_df[trades_df["Symbol"] == symbol].copy()
    df_sym = df_sym.sort_values("Exit Time")

    eq = INITIAL_EQUITY + (
        df_sym["PnL R"].cumsum() * INITIAL_EQUITY * RISK_PCT
    )

    eq.index = pd.to_datetime(df_sym["Exit Time"])
    symbol_equity[symbol] = eq

symbol_returns = {}

for symbol, eq in symbol_equity.items():
    eq_daily = eq.resample("D").last().ffill()
    rets = eq_daily.pct_change().dropna()
    symbol_returns[symbol] = rets
returns_df = pd.DataFrame(symbol_returns).dropna()

cov = returns_df.cov().values
n = cov.shape[0]

def portfolio_risk(w):
    return np.sqrt(w @ cov @ w)

def risk_contribution(w):
    port_var = w @ cov @ w
    mrc = cov @ w
    return w * mrc / port_var

def erc_objective(w):
    rc = risk_contribution(w)
    return np.sum((rc - rc.mean())**2)

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0,1)] * n
w0 = np.ones(n) / n

res = minimize(erc_objective, w0, bounds=bounds, constraints=constraints)
erc_weights = res.x
print(f"ERC weights: {erc_weights}")

#Tangecny
mu = returns_df.mean().values
cov = returns_df.cov().values
cov += np.eye(len(cov)) * 1e-6
inv_cov = np.linalg.inv(cov)
raw_w = inv_cov @ mu
tan_weights = raw_w / raw_w.sum()
alpha = 0.3  # tilt strength
weights = (1 - alpha) * erc_weights + alpha * tan_weights
weights /= weights.sum()
print(f"Tangecny weights: {weights}")

# =========================
# Monte Carlo simulation
# =========================
# använd faktiska trades
r_values = trades_df["PnL R"].values

N_TRADES = len(r_values)
N_SIM = 10_000

def run_mc_simulation(r_values, n_trades, n_sim, initial_equity, risk_pct):
    final_equities = []
    max_dds = []

    for _ in range(n_sim):
        # resample trades with replacement
        sampled_r = np.random.choice(r_values, size=n_trades, replace=True)

        equity = initial_equity
        peak = equity
        max_dd = 0.0

        for r in sampled_r:
            equity += r * (initial_equity * risk_pct)
            peak = max(peak, equity)
            dd = (equity / peak) - 1.0
            max_dd = min(max_dd, dd)

        final_equities.append(equity)
        max_dds.append(max_dd)

    return np.array(final_equities), np.array(max_dds)

final_eq, max_dd = run_mc_simulation(
    r_values,
    N_TRADES,
    N_SIM,
    INITIAL_EQUITY,
    RISK_PCT
)

print("Monte Carlo results ({} simulations)".format(N_SIM))
print(f"Median final equity:      {np.median(final_eq):.2f}")
print(f"5th percentile equity:   {np.percentile(final_eq, 5):.2f}")
print(f"Worst-case equity:       {final_eq.min():.2f}")

print(f"Median max DD:           {np.median(max_dd) * 100:.2f}%")
print(f"95th percentile max DD:  {np.percentile(max_dd, 5) * 100:.2f}%")
print(f"Worst max DD observed:   {max_dd.min() * 100:.2f}%")
