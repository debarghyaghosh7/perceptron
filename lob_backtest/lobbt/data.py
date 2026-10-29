"""
Loading and sanity-checking LOB feature files.

The loader is deliberately paranoid. Most silent backtest bugs are data bugs:
out-of-order timestamps, crossed books, duplicated snapshots, NaNs that quietly
become zeros. We fail loudly here instead of producing a pretty equity curve
that means nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Columns the engine itself needs. Everything else is a candidate feature.
REQUIRED = ["received_at_utc", "best_bid", "best_ask", "mid_price", "spread"]

# Columns that must never be fed to a model as features: they are either
# identifiers, raw prices (non-stationary), or forward-looking.
NON_FEATURES = {
    "received_at_utc",
    "receive_time_ns",
    "last_update_id",
    "best_bid",
    "best_ask",
    "mid_price",
    "microprice",
    "top20_changed",
}

# Reasonable default feature set for this schema: stationary, scale-free.
DEFAULT_FEATURES = [
    "spread_bps",
    "obi_1",
    "obi_5",
    "obi_10",
    "obi_20",
    "microprice_minus_mid_bps",
    "simple_return_1",
    "log_return_1",
    "rolling_vol_100_events",
    "rolling_vol_500_events",
    "elapsed_ms",
]


@dataclass
class DataQualityReport:
    n_rows: int
    duration_s: float
    median_gap_ms: float
    max_gap_ms: float
    n_crossed: int
    n_zero_spread: int
    n_duplicate_updates: int
    frac_mid_changed: float
    nan_counts: dict
    const_columns: list
    warnings: list

    def __str__(self) -> str:
        lines = [
            "Data quality report",
            "-------------------",
            f"rows                : {self.n_rows}",
            f"duration            : {self.duration_s:.1f} s "
            f"({self.duration_s/60:.1f} min)",
            f"median inter-event  : {self.median_gap_ms:.1f} ms",
            f"max inter-event gap : {self.max_gap_ms:.1f} ms",
            f"crossed books       : {self.n_crossed}",
            f"zero spreads        : {self.n_zero_spread}",
            f"duplicate update ids: {self.n_duplicate_updates}",
            f"events w/ mid move  : {self.frac_mid_changed:.3%}",
        ]
        if self.const_columns:
            lines.append(f"constant columns    : {self.const_columns}")
        if self.nan_counts:
            lines.append(f"NaNs                : {self.nan_counts}")
        if self.warnings:
            lines.append("")
            lines.append("WARNINGS")
            lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


def load_lob(
    path: str,
    features: list[str] | None = None,
    dropna: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str], DataQualityReport]:
    """
    Read a feature CSV and return (frame, feature_names, quality_report).

    The frame is sorted by time, de-duplicated, and has a monotonic
    RangeIndex. `feature_names` is the list of columns the model may use.
    """
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df["ts"] = pd.to_datetime(df["received_at_utc"], utc=True)
    if "receive_time_ns" in df.columns:
        df["ts_ns"] = df["receive_time_ns"].astype("int64")
    else:
        df["ts_ns"] = df["ts"].astype("int64")

    # Order and de-duplicate before anything else.
    df = df.sort_values("ts_ns", kind="mergesort").reset_index(drop=True)
    n_before = len(df)
    df = df.drop_duplicates(subset="ts_ns", keep="last").reset_index(drop=True)
    n_dup_ts = n_before - len(df)

    report = _quality(df, n_dup_ts)

    if features is None:
        features = [c for c in DEFAULT_FEATURES if c in df.columns]
    leaked = [f for f in features if f in NON_FEATURES]
    if leaked:
        raise ValueError(
            f"These columns are not allowed as features (non-stationary or "
            f"identifiers): {leaked}. Derive a scale-free version instead."
        )

    if dropna:
        n0 = len(df)
        df = df.dropna(subset=features).reset_index(drop=True)
        dropped = n0 - len(df)
        if dropped and verbose:
            print(f"[data] dropped {dropped} warmup rows with NaN features")

    if verbose:
        print(report)
        print(f"\n[data] using {len(features)} features: {features}\n")

    return df, list(features), report


def _quality(df: pd.DataFrame, n_dup_ts: int) -> DataQualityReport:
    if df.empty:
        raise ValueError(
            "no rows left after parsing. If this came from the synthetic "
            "generator, n is smaller than the rolling-volatility warmup."
        )
    gaps = df["ts_ns"].diff().dropna() / 1e6  # ms
    crossed = int((df["best_bid"] >= df["best_ask"]).sum())
    zero_spread = int((df["spread"] <= 0).sum())
    dup_updates = 0
    if "last_update_id" in df.columns:
        dup_updates = int(df["last_update_id"].duplicated().sum())

    mid_moved = float((df["mid_price"].diff().fillna(0) != 0).mean())
    nans = df.isna().sum()
    nans = {k: int(v) for k, v in nans.items() if v}

    numeric = df.select_dtypes(include=[np.number])
    const = [c for c in numeric.columns if numeric[c].nunique(dropna=True) <= 1]

    warn: list[str] = []
    dur = (df["ts_ns"].iloc[-1] - df["ts_ns"].iloc[0]) / 1e9
    if dur < 3600:
        warn.append(
            f"only {dur/60:.1f} minutes of data. A regime-shift claim needs "
            f"hours-to-days spanning at least one real volatility event."
        )
    if mid_moved < 0.05:
        warn.append(
            f"mid price moves on just {mid_moved:.2%} of events. Labels will be "
            f"overwhelmingly 'flat' and accuracy will look great for the wrong "
            f"reason. Either sample on book changes or label over a longer "
            f"horizon."
        )
    if const:
        warn.append(f"constant columns carry no signal: {const}")
    if crossed:
        warn.append(f"{crossed} crossed books - bad snapshots, investigate")
    if n_dup_ts:
        warn.append(f"{n_dup_ts} rows shared a timestamp and were dropped")
    if len(gaps) and gaps.max() > 10 * gaps.median():
        warn.append(
            f"largest inter-event gap is {gaps.max():.0f} ms vs median "
            f"{gaps.median():.0f} ms - possible feed outage"
        )

    return DataQualityReport(
        n_rows=len(df),
        duration_s=dur,
        median_gap_ms=float(gaps.median()) if len(gaps) else 0.0,
        max_gap_ms=float(gaps.max()) if len(gaps) else 0.0,
        n_crossed=crossed,
        n_zero_spread=zero_spread,
        n_duplicate_updates=dup_updates,
        frac_mid_changed=mid_moved,
        nan_counts=nans,
        const_columns=const,
        warnings=warn,
    )


def to_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Feature matrix as float64, C-contiguous, NaN/inf scrubbed to 0."""
    X = df[features].to_numpy(dtype=np.float64, copy=True)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(X)


def concat_sessions(paths: list[str], **kw):
    """
    Stitch several capture files into one timeline.

    Use this once you have more data. It inserts a `session_id` column so the
    label builder and the walk-forward splitter never span a capture boundary
    (returns across a gap are meaningless).
    """
    frames, feats, reports = [], None, []
    for i, p in enumerate(paths):
        d, f, r = load_lob(p, verbose=False, **kw)
        d["session_id"] = i
        frames.append(d)
        reports.append(r)
        feats = f
    out = pd.concat(frames, ignore_index=True).sort_values("ts_ns")
    return out.reset_index(drop=True), feats, reports
