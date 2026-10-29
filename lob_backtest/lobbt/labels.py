"""
Label construction.

The backtester owns labelling, not the ML person. Reason: the label definition
decides the purge gap in the walk-forward split and the horizon in the engine.
If those three disagree you get leakage, and leakage is invisible in the
metrics - it just makes everything look good.

Two labellers:

  horizon_labels   : sign of the forward mid move over k events or t ms,
                     with a deadband so "flat" means "didn't move enough to
                     pay for the spread"
  triple_barrier   : first-touch of an up barrier, a down barrier, or a time
                     limit. Better matched to how a strategy actually exits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DOWN, FLAT, UP = 0, 1, 2


def horizon_labels(
    df: pd.DataFrame,
    horizon_events: int | None = None,
    horizon_ms: float | None = None,
    deadband_bps: float = 0.0,
    price_col: str = "mid_price",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward return label.

    Returns (y, fwd_ret_bps, valid_mask).

    Exactly one of horizon_events / horizon_ms must be set.
      horizon_events : k events ahead. Simple, but a fixed event count is a
                       variable amount of clock time.
      horizon_ms     : wall-clock horizon. Preferred - it is what a latency
                       budget is measured in.

    deadband_bps is the dead zone. Set it to at least half the typical spread
    in bps, otherwise you are training the model to predict moves it could
    never profitably trade.
    """
    if (horizon_events is None) == (horizon_ms is None):
        raise ValueError("set exactly one of horizon_events or horizon_ms")

    px = df[price_col].to_numpy(float)
    n = len(px)
    fwd_idx = _forward_index(df, horizon_events, horizon_ms)

    valid = fwd_idx >= 0
    fwd_px = np.where(valid, px[np.clip(fwd_idx, 0, n - 1)], np.nan)
    ret_bps = (fwd_px - px) / px * 1e4

    y = np.full(n, FLAT, dtype=np.int8)
    y[ret_bps > deadband_bps] = UP
    y[ret_bps < -deadband_bps] = DOWN
    y[~valid] = -1
    return y, ret_bps, valid


def _forward_index(df, horizon_events, horizon_ms) -> np.ndarray:
    n = len(df)
    if horizon_events is not None:
        idx = np.arange(n) + int(horizon_events)
        idx[idx >= n] = -1
        return idx

    ts = df["ts_ns"].to_numpy("int64")
    target = ts + int(horizon_ms * 1e6)
    idx = np.searchsorted(ts, target, side="left")
    idx[idx >= n] = -1
    # don't let a label span a capture boundary
    if "session_id" in df.columns:
        sess = df["session_id"].to_numpy()
        ok = (idx >= 0) & (sess[np.clip(idx, 0, n - 1)] == sess)
        idx = np.where(ok, idx, -1)
    return idx


def triple_barrier(
    df: pd.DataFrame,
    up_bps: float,
    down_bps: float,
    max_horizon_ms: float,
    price_col: str = "mid_price",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    First-touch labelling.

    Returns (y, holding_ms, valid_mask). y is UP if the up barrier is hit
    first, DOWN if the down barrier is hit first, FLAT if the clock runs out.

    This is the honest label for a strategy with a take-profit and a stop,
    and it gives you the holding time distribution for free - which you need
    for the turnover and latency sections of the report.
    """
    px = df[price_col].to_numpy(float)
    ts = df["ts_ns"].to_numpy("int64")
    n = len(px)
    horizon_ns = int(max_horizon_ms * 1e6)

    y = np.full(n, -1, dtype=np.int8)
    hold = np.full(n, np.nan)
    end_idx = np.searchsorted(ts, ts + horizon_ns, side="right")

    for i in range(n):
        hi = min(end_idx[i], n)
        if hi <= i + 1:
            continue
        up_px = px[i] * (1 + up_bps / 1e4)
        dn_px = px[i] * (1 - down_bps / 1e4)
        seg = px[i + 1:hi]
        up_hit = np.flatnonzero(seg >= up_px)
        dn_hit = np.flatnonzero(seg <= dn_px)
        first_up = up_hit[0] if up_hit.size else np.inf
        first_dn = dn_hit[0] if dn_hit.size else np.inf
        if first_up == first_dn == np.inf:
            y[i] = FLAT
            hold[i] = (ts[hi - 1] - ts[i]) / 1e6
        elif first_up < first_dn:
            y[i] = UP
            hold[i] = (ts[i + 1 + int(first_up)] - ts[i]) / 1e6
        else:
            y[i] = DOWN
            hold[i] = (ts[i + 1 + int(first_dn)] - ts[i]) / 1e6

    return y, hold, y >= 0


def label_summary(y: np.ndarray) -> dict:
    """Class balance. Print this before you trust any accuracy number."""
    v = y[y >= 0]
    if not len(v):
        return {}
    out = {
        "n_labelled": int(len(v)),
        "down": float((v == DOWN).mean()),
        "flat": float((v == FLAT).mean()),
        "up": float((v == UP).mean()),
    }
    out["majority_class_accuracy"] = max(out["down"], out["flat"], out["up"])
    return out


def recommended_deadband_bps(df: pd.DataFrame, multiple: float = 0.5) -> float:
    """Half the median spread in bps. The minimum sensible deadband."""
    return float(df["spread_bps"].median() * multiple)
