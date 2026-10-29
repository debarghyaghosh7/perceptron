"""
Walk-forward splitting with purge and embargo.

Why not KFold: with a forward-looking label, the training row at index i
overlaps in time with the test row at index i+1. Random folds therefore train
on the answer. This is the single most common reason a hackathon LOB model
scores 0.75 in the notebook and 0.50 at the demo.

What we do instead:

  train:  [t0 ............ t1]
  purge:              <-- horizon --> dropped from the end of train
  embargo:                           <-- gap --> dropped from start of test
  test:                                          [t2 ...... t3]

Purge removes training rows whose label window reaches into the test set.
Embargo removes test rows immediately after train, where serial correlation
in the features still effectively leaks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray

    def __repr__(self):
        return (
            f"Fold {self.fold_id}: train[{self.train_idx[0]}:"
            f"{self.train_idx[-1]}] n={len(self.train_idx)} | "
            f"test[{self.test_idx[0]}:{self.test_idx[-1]}] "
            f"n={len(self.test_idx)}"
        )


def walk_forward(
    df: pd.DataFrame,
    n_folds: int = 5,
    label_horizon_ms: float = 1000.0,
    embargo_ms: float = 1000.0,
    mode: str = "expanding",       # "expanding" or "rolling"
    train_frac: float = 0.5,
    min_train: int = 500,
) -> list[Fold]:
    """
    Build time-ordered folds.

    mode="expanding": train window grows each fold. Use for the headline
                      result - it is what a live system would have.
    mode="rolling"  : fixed-length train window. Use to show the model needs
                      recent data, which is itself evidence of regime shift.
    """
    ts = df["ts_ns"].to_numpy("int64")
    n = len(df)
    purge_ns = int(label_horizon_ms * 1e6)
    embargo_ns = int(embargo_ms * 1e6)

    first_test = max(int(n * train_frac), min_train)
    if first_test >= n:
        raise ValueError(
            f"not enough rows ({n}) for train_frac={train_frac} and "
            f"min_train={min_train}"
        )
    bounds = np.linspace(first_test, n, n_folds + 1).astype(int)

    folds = []
    for k in range(n_folds):
        t_start, t_end = bounds[k], bounds[k + 1]
        if t_end - t_start < 20:
            continue

        test_idx = np.arange(t_start, t_end)
        # embargo: push test start forward in clock time
        emb_cut = np.searchsorted(ts, ts[t_start] + embargo_ns, "left")
        test_idx = test_idx[test_idx >= emb_cut] if emb_cut > t_start else test_idx
        if len(test_idx) < 20:
            continue

        train_end = t_start
        # purge: drop train rows whose label reaches past train_end
        purge_cut = np.searchsorted(ts, ts[train_end - 1] - purge_ns, "left")
        train_start = 0 if mode == "expanding" else max(
            0, purge_cut - (bounds[1] - bounds[0])
        )
        train_idx = np.arange(train_start, purge_cut)
        if len(train_idx) < min_train // 2:
            continue

        folds.append(Fold(k, train_idx, test_idx))

    if not folds:
        raise ValueError(
            "no usable folds. Dataset is too short for this horizon/embargo. "
            "Reduce n_folds, or get more data."
        )
    return folds


def assert_no_leakage(
    folds: list[Fold], df: pd.DataFrame, label_horizon_ms: float
) -> None:
    """
    Hard check. Raises if any training label window overlaps its test window.
    Call this in CI. It is the only guarantee the metrics mean anything.
    """
    ts = df["ts_ns"].to_numpy("int64")
    h = int(label_horizon_ms * 1e6)
    for f in folds:
        train_label_end = ts[f.train_idx[-1]] + h
        test_start = ts[f.test_idx[0]]
        if train_label_end > test_start:
            raise AssertionError(
                f"LEAKAGE in fold {f.fold_id}: last training label closes "
                f"{(train_label_end - test_start)/1e6:.1f} ms after the test "
                f"window opens. Increase purge."
            )
        if f.train_idx[-1] >= f.test_idx[0]:
            raise AssertionError(f"fold {f.fold_id}: index overlap")


def single_split(
    df: pd.DataFrame, train_frac: float = 0.6, label_horizon_ms: float = 1000.0,
    embargo_ms: float = 1000.0,
) -> Fold:
    """One train/test cut. Use for quick iteration, never for the final number."""
    return walk_forward(
        df, n_folds=1, label_horizon_ms=label_horizon_ms,
        embargo_ms=embargo_ms, train_frac=train_frac, min_train=100,
    )[0]
