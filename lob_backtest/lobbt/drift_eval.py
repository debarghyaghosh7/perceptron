"""
Scoring the drift detector itself.

The ML person builds the detector. Scoring it is a backtest job, because a
detector is only good relative to a cost: firing late means trading a stale
model, firing spuriously means throwing away a model that was fine.

Two regimes of evaluation:

  WITH ground truth (synthetic data, or a real session where you can point at
  a timestamped macro event): detection delay, false alarm rate, miss rate.

  WITHOUT ground truth (a plain capture): you cannot score detection directly.
  Use the degradation proxy - compare a frozen model against an adapting one
  and show the adapting one holds accuracy where the frozen one drops. That
  comparison is the evidence; the detector is only the mechanism.

Put both in the abstract's evaluation plan. Reviewers specifically ask whether
the comparison would actually prove the claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_detector(
    flags: list[int] | np.ndarray,
    true_shifts: list[int],
    tolerance_events: int = 500,
    n_events: int | None = None,
) -> dict:
    """
    Match each true shift to the first flag inside [shift, shift+tolerance].

    Returns detection delay stats, misses, and a false-alarm rate per 1000
    events for flags that match no shift.
    """
    flags = np.sort(np.asarray(flags, dtype=int))
    matched_flags: set[int] = set()
    delays: list[int] = []
    misses: list[int] = []

    for s in sorted(true_shifts):
        cand = flags[(flags >= s) & (flags <= s + tolerance_events)]
        cand = [c for c in cand if c not in matched_flags]
        if cand:
            matched_flags.add(cand[0])
            delays.append(int(cand[0] - s))
        else:
            misses.append(int(s))

    fa = [int(f) for f in flags if f not in matched_flags]
    out = {
        "n_true_shifts": len(true_shifts),
        "n_detected": len(delays),
        "n_missed": len(misses),
        "recall": len(delays) / len(true_shifts) if true_shifts else np.nan,
        "n_false_alarms": len(fa),
        "precision": len(delays) / len(flags) if len(flags) else np.nan,
        "missed_at": misses,
        "false_alarm_at": fa,
    }
    if delays:
        out.update(
            mean_delay_events=float(np.mean(delays)),
            median_delay_events=float(np.median(delays)),
            max_delay_events=int(np.max(delays)),
        )
    if n_events:
        out["false_alarms_per_1k_events"] = len(fa) / n_events * 1000
    return out


def detection_delay_ms(
    flags: list[int], true_shifts: list[int], ts_ns: np.ndarray,
    tolerance_events: int = 500,
) -> dict:
    """Same as above but in wall-clock ms, which is what the judges will ask."""
    s = score_detector(flags, true_shifts, tolerance_events, len(ts_ns))
    if "mean_delay_events" not in s:
        return s
    delays_ms = []
    flags_arr = np.sort(np.asarray(flags, int))
    used: set[int] = set()
    for sh in sorted(true_shifts):
        cand = [c for c in flags_arr
                if sh <= c <= sh + tolerance_events and c not in used]
        if cand:
            used.add(cand[0])
            delays_ms.append((ts_ns[cand[0]] - ts_ns[sh]) / 1e6)
    s["mean_delay_ms"] = float(np.mean(delays_ms))
    s["median_delay_ms"] = float(np.median(delays_ms))
    return s


def degradation_curve(
    static_acc: pd.Series, adaptive_acc: pd.Series, shift_idx: int,
    window: int = 500,
) -> dict:
    """
    The headline number when you have no ground truth.

    Compares rolling accuracy before and after a shift for a frozen model vs
    an adapting one. `recovery_events` is how long the adaptive model takes to
    return to within 1 point of its own pre-shift accuracy.
    """
    def seg(s, a, b):
        v = s.iloc[a:b].dropna()
        return float(v.mean()) if len(v) else np.nan

    pre_a, pre_b = max(0, shift_idx - 5 * window), shift_idx
    post_a, post_b = shift_idx, min(len(static_acc), shift_idx + 5 * window)

    s_pre, s_post = seg(static_acc, pre_a, pre_b), seg(static_acc, post_a, post_b)
    a_pre, a_post = seg(adaptive_acc, pre_a, pre_b), seg(adaptive_acc, post_a, post_b)

    out = {
        "static_pre": s_pre, "static_post": s_post,
        "static_drop": s_pre - s_post,
        "adaptive_pre": a_pre, "adaptive_post": a_post,
        "adaptive_drop": a_pre - a_post,
        "degradation_avoided": (s_pre - s_post) - (a_pre - a_post),
    }

    tail = adaptive_acc.iloc[shift_idx:]
    rec = tail[tail >= a_pre - 0.01]
    out["recovery_events"] = int(rec.index[0] - shift_idx) if len(rec) else None
    return out


def flags_to_regimes(flags: list[int], n: int) -> np.ndarray:
    """Turn detector flags into an integer regime id per event, for colouring."""
    r = np.zeros(n, dtype=int)
    for k, f in enumerate(sorted(flags), start=1):
        r[f:] = k
    return r
