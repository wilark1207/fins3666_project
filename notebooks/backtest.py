# backtest.py
import numpy as np
import pandas as pd

# ---------- Helpers

def _ensure_index(df, ts_col="timestamp"):
    if ts_col in df.columns:
        df = df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).set_index(ts_col)
    df = df.sort_index()
    return df

def _logret(series):
    return np.log(series).diff()

# ---------- Signal & position construction

def make_signals_from_prob(prob_up: pd.Series, upper=0.55, lower=0.45) -> pd.Series:
    """
    Map model probabilities to discrete signals in {+1, 0, -1}.
    """
    sig = pd.Series(0, index=prob_up.index, dtype="int8")
    sig[prob_up > upper] = 1
    sig[prob_up < lower] = -1
    return sig

def make_positions_fixed_hold(
    raw_signal: pd.Series,
    hold_bars: int = 5,
    execute_next_bar: bool = True
) -> pd.Series:
    """
    Build a position time-series that:
      - Enters on the *next* bar after the signal (if execute_next_bar=True)
      - Holds exactly `hold_bars`
      - Closes automatically after hold; no mid-hold flips
    Later signals during an active hold are ignored; a new position can start only after the prior exits.
    """
    idx = raw_signal.index
    pos = pd.Series(0, index=idx, dtype="int8")

    active = 0           # current position (+1/-1/0)
    bars_left = 0        # holding bars remaining
    pending = 0          # pending position to start on next bar
    for i, t in enumerate(idx):
        # if a position is active, continue it
        if bars_left > 0:
            pos.iloc[i] = active
            bars_left -= 1
            if bars_left == 0:
                active = 0  # auto-close at end of hold
            continue

        # no active position now
        if pending != 0:
            # start pending position at this bar
            active = pending
            pos.iloc[i] = active
            bars_left = hold_bars - 1  # we count this bar as the first
            pending = 0
            continue

        s = int(raw_signal.iloc[i])

        if s != 0:
            if execute_next_bar:
                pending = s  # schedule for next bar
                pos.iloc[i] = 0
            else:
                active = s
                pos.iloc[i] = active
                bars_left = hold_bars - 1
        else:
            pos.iloc[i] = 0

    return pos

# ---------- P&L engine

def backtest_minute_open_to_open(
    df: pd.DataFrame,
    prob_col: str = "prob_up",
    upper: float = 0.55,
    lower: float = 0.45,
    hold_bars: int = 5,
    cost_bps_per_side: float = 0.0,
    slippage_bps_per_side: float = 0.0,
    ts_col_candidates=("timestamp_ny", "timestamp_utc", "timestamp"),
    open_col="open",
    close_col="close",
):
    """
    Strategy:
      - Build signals from CNN-LSTM probability.
      - Execute at next bar's open; hold exactly `hold_bars` minutes; then flat.
      - Returns accrue from open-to-open.
      - Costs applied on entries and exits (per side, in bps).

    Returns:
      dict with:
        df_perf: per-bar performance DataFrame
        trades:  trade ledger DataFrame
        metrics: summary dictionary
    """
    # 1) choose a timestamp column and set index
    df0 = df.copy()
    ts_col = next((c for c in ts_col_candidates if c in df0.columns), None)
    if ts_col is None:
        raise ValueError("No timestamp column found.")
    df0 = _ensure_index(df0, ts_col=ts_col)

    # sanity columns
    for col in [open_col, close_col, prob_col]:
        if col not in df0.columns:
            raise ValueError(f"Missing column: {col}")

    # 2) per-bar open-to-open log returns (execution price model)
    # r_t = log(open_t / open_{t-1})
    df0["ret_o2o"] = _logret(df0[open_col])

    # 3) signals & positions
    raw_sig = make_signals_from_prob(df0[prob_col], upper, lower)
    pos = make_positions_fixed_hold(raw_sig, hold_bars=hold_bars, execute_next_bar=True)

    # 4) per-bar strategy return = position_{t-1} * r_t
    # because the position becomes active at the bar open (we decided at t-1)
    pos_lag = pos.shift(1).fillna(0)
    strat_ret = (pos_lag * df0["ret_o2o"]).fillna(0.0)

    # 5) transaction costs/slippage (bps per side)
    # charge on position changes (entry and exit)
    # per-change cost in return terms ≈ bps/1e4
    per_side = (cost_bps_per_side + slippage_bps_per_side) / 1e4
    changes = pos.diff().fillna(pos.iloc[0])  # first bar change = pos0 - 0
    # any nonzero change triggers a side; absolute change can be 1 (enter/exit) or 2 (flip)
    # For flip (+1 -> -1), that's two sides (exit + entry).
    trans_cost = -abs(changes) * per_side
    # Costs are applied at the moment of change (at bar t open). They impact r_t.
    strat_ret_net = strat_ret + trans_cost

    # 6) equity curve
    equity = strat_ret_net.cumsum().apply(np.exp)  # start at 1.0
    df_perf = pd.DataFrame({
        "position": pos,
        "position_lag": pos_lag,
        "ret_o2o": df0["ret_o2o"],
        "strat_ret": strat_ret,
        "trans_cost": trans_cost,
        "strat_ret_net": strat_ret_net,
        "equity": equity,
    }, index=df0.index)

    # ---------- Trade ledger (entries at position changes from 0->±1, exits at ±1->0)
    entries = df_perf.index[(df_perf["position_lag"] == 0) & (df_perf["position"] != 0)]
    exits   = df_perf.index[(df_perf["position_lag"] != 0) & (df_perf["position"] == 0)]

    # Also handle flips (+1 -> -1 or -1 -> +1): treat as exit then immediate entry
    flips = df_perf.index[(df_perf["position_lag"] != 0) & (df_perf["position"] != 0) &
                          (np.sign(df_perf["position"]) != np.sign(df_perf["position_lag"]))]

    # Build a list of trades by walking the series
    trades = []
    curr_dir = 0
    entry_time = None
    entry_px = None

    opens = df0[open_col]
    for t in df_perf.index:
        p_prev = int(df_perf.loc[t, "position_lag"])
        p_now  = int(df_perf.loc[t, "position"])

        # flip or close
        if p_prev != 0 and (p_now == 0 or np.sign(p_now) != np.sign(p_prev)):
            exit_time = t
            exit_px = opens.loc[t]  # we modeled execution at open
            direction = int(np.sign(curr_dir))
            pnl = (np.log(exit_px) - np.log(entry_px)) * direction if (entry_px is not None) else 0.0
            trades.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "direction": direction,
                "entry_open": float(entry_px),
                "exit_open": float(exit_px),
                "ret_log": pnl,
                "ret_pct": np.exp(pnl) - 1.0,
                "holding_bars": int((exit_time - entry_time).total_seconds() // 60) if (entry_time and exit_time) else None
            })
            # after a flip, we’ll open a new one below

        # open (including after flip)
        if p_now != 0 and p_prev == 0:
            entry_time = t
            entry_px = opens.loc[t]
            curr_dir = p_now

        # flip: enters again immediately
        if p_prev != 0 and p_now != 0 and np.sign(p_now) != np.sign(p_prev):
            entry_time = t
            entry_px = opens.loc[t]
            curr_dir = p_now

    trades = pd.DataFrame(trades)

    # ---------- Metrics
    def max_drawdown(equity_series):
        peak = equity_series.cummax()
        dd = equity_series / peak - 1.0
        return float(dd.min()), dd

    ann_factor = 252 * 390  # ~ minutes per trading year
    mu = df_perf["strat_ret_net"].mean()
    sd = df_perf["strat_ret_net"].std(ddof=0)
    sharpe = (mu / sd) * np.sqrt(ann_factor) if sd > 0 else np.nan
    ann_vol = sd * np.sqrt(ann_factor)

    mdd, dd_series = max_drawdown(df_perf["equity"])
    exposure = (df_perf["position"] != 0).mean()

    if not trades.empty:
        hit_rate = (trades["ret_pct"] > 0).mean()
        avg_trade = trades["ret_pct"].mean()
        med_trade = trades["ret_pct"].median()
        avg_hold = trades["holding_bars"].mean()
    else:
        hit_rate = avg_trade = med_trade = avg_hold = np.nan

    metrics = {
        "bars": int(len(df_perf)),
        "exposure": float(exposure),
        "ann_return_geom": float(df_perf["strat_ret_net"].sum()),  # log-return sum (≈ geometric)
        "sharpe": float(sharpe),
        "ann_vol": float(ann_vol),
        "max_drawdown": float(mdd),
        "hit_rate": float(hit_rate),
        "avg_trade_ret": float(avg_trade),
        "median_trade_ret": float(med_trade),
        "avg_holding_bars": float(avg_hold),
        "trades": int(len(trades))
    }

    return {
        "df_perf": df_perf,
        "trades": trades,
        "metrics": metrics
    }
