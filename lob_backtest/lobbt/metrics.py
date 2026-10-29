"""
Metrics.

Two families, and they answer different questions. Report both.

  Forecast quality : is the model right?   (accuracy, macro-F1, AUC, Brier)
  Economic quality : does being right pay? (PnL after cost, Sharpe, DD)

A model can win on the first and lose on the second, and on 100 ms crypto data
it usually does. The gap between the two tables is the most interesting plot in
the deck, so we compute both off the same run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


# ---------------------------------------------------------------- forecast
def forecast_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """y_true in {0,1,2} with -1 for unlabelled. proba shape (n,3)."""
    m = (y_true >= 0) & np.isfinite(proba).all(axis=1)
    if m.sum() < 10:
        return {"n": int(m.sum())}
    yt, p = y_true[m], proba[m]
    yp = p.argmax(1)

    out = {
        "n": int(m.sum()),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_f1": float(f1_score(yt, yp, average="macro",
                                      zero_division=0)),
        "mcc": float(matthews_corrcoef(yt, yp)) if len(np.unique(yt)) > 1 else 0.0,
    }
    # majority-class baseline: the number your accuracy must beat
    counts = np.bincount(yt, minlength=3)
    out["majority_baseline_acc"] = float(counts.max() / counts.sum())
    out["edge_over_majority"] = out["accuracy"] - out["majority_baseline_acc"]

    # directional AUC: up vs down only, ignoring flat
    dmask = yt != 1
    if dmask.sum() > 10 and len(np.unique(yt[dmask])) == 2:
        score = p[dmask][:, 2] / np.clip(p[dmask][:, 0] + p[dmask][:, 2], 1e-9, None)
        try:
            out["directional_auc"] = float(roc_auc_score(yt[dmask] == 2, score))
        except ValueError:
            pass
    # calibration
    try:
        out["brier_up"] = float(brier_score_loss((yt == 2).astype(int), p[:, 2]))
    except ValueError:
        pass
    out["confusion"] = confusion_matrix(yt, yp, labels=[0, 1, 2]).tolist()
    return out


def rolling_accuracy(
    y_true: np.ndarray, proba: np.ndarray, window: int = 500
) -> pd.Series:
    """
    Rolling hit rate. This is the curve that makes the whole PS visible:
    the static model's line sags after the shift, the adaptive one recovers.
    """
    m = (y_true >= 0) & np.isfinite(proba).all(axis=1)
    hit = pd.Series(np.where(m, (proba.argmax(1) == y_true).astype(float),
                             np.nan))
    return hit.rolling(window, min_periods=window // 4).mean()


# ---------------------------------------------------------------- economic
def pnl_metrics(
    res, events_per_year: float | None = None, notional_per_unit: float | None = None
) -> dict:
    """
    PnL stats from a BacktestResult.

    events_per_year is inferred from the median inter-event time if not given.
    Sharpe on sub-second data is noisy by construction - quote it with the
    event frequency next to it, and prefer PnL-per-round-trip in bps as the
    headline number.
    """
    ev, tr = res.events, res.trades
    if ev.empty:
        return {}

    pnl = ev["pnl"].to_numpy(float)
    step = np.diff(pnl, prepend=pnl[0])
    dt_s = np.median(np.diff(ev["ts_ns"])) / 1e9 if len(ev) > 1 else 0.1
    if events_per_year is None:
        events_per_year = 365 * 24 * 3600 / max(dt_s, 1e-9)

    sd = step.std(ddof=1) if len(step) > 2 else 0.0
    sharpe = (step.mean() / sd * np.sqrt(events_per_year)) if sd > 0 else 0.0

    peak = np.maximum.accumulate(pnl)
    dd = pnl - peak
    notional = float(tr["notional"].sum()) if len(tr) else 0.0

    out = {
        "final_pnl_quote": float(pnl[-1]),
        "n_trades": int(len(tr)),
        "total_notional": notional,
        "total_fees": float(tr["fee"].sum()) if len(tr) else 0.0,
        "pnl_bps_of_notional": float(pnl[-1] / notional * 1e4) if notional else 0.0,
        "sharpe_annualised": float(sharpe),
        "max_drawdown_quote": float(dd.min()),
        "mean_step_pnl": float(step.mean()),
        "hit_rate_steps": float((step[step != 0] > 0).mean())
        if (step != 0).any() else 0.0,
        "avg_slip_bps": float(tr["slip_bps"].mean()) if len(tr) else 0.0,
        "turnover_per_min": float(len(tr) / (dt_s * len(ev) / 60))
        if len(ev) else 0.0,
        "time_in_market": float((ev["position"].abs() > 1e-12).mean()),
        "median_decide_us": float(ev["decide_us"].median()),
        "p99_decide_us": float(ev["decide_us"].quantile(0.99)),
        "median_event_gap_ms": float(dt_s * 1000),
    }
    if len(tr):
        hold = tr["event"].diff().dropna()
        out["median_events_between_trades"] = float(hold.median()) if len(hold) else 0.0
    return out


def segment_metrics(
    res, y: np.ndarray, proba: np.ndarray, breakpoints: list[int]
) -> pd.DataFrame:
    """
    Per-regime table. Pass the true or detected shift indices as breakpoints
    and get accuracy/PnL in each segment. This is the table that proves the
    model degraded, which is the whole claim of the problem statement.
    """
    bounds = [0] + sorted(breakpoints) + [len(res.events)]
    rows = []
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        if b - a < 20:
            continue
        fm = forecast_metrics(y[a:b], proba[a:b])
        seg = res.events.iloc[a:b]
        rows.append({
            "segment": k,
            "start": a,
            "end": b,
            "n": b - a,
            "accuracy": fm.get("accuracy", np.nan),
            "balanced_f1": fm.get("balanced_f1", np.nan),
            "edge_over_majority": fm.get("edge_over_majority", np.nan),
            "pnl": float(seg["pnl"].iloc[-1] - seg["pnl"].iloc[0]),
            "realised_vol": float(seg["mid"].pct_change().std() * 1e4),
        })
    return pd.DataFrame(rows)


def compare(results: dict[str, object], **kw) -> pd.DataFrame:
    """Side-by-side table of several runs. Index is the run name."""
    rows = {k: pnl_metrics(v, **kw) for k, v in results.items()}
    return pd.DataFrame(rows).T
