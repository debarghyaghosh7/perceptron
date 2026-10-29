"""
Contracts between the three workstreams.

Everyone codes against these. Nobody imports anyone else's module directly.

  ML person      -> implements Model
  Strategy person-> implements Strategy
  ML person      -> implements DriftDetector (or strategy person, your call)
  Backtest (me)  -> consumes all three, owns the event loop

Rules that the engine enforces, so you don't have to worry about them:
  * You never see the future. `fit` only gets the training slice; `predict`
    only gets one row at a time, in order.
  * Your decision at event i is executed at event j > i, after latency.
  * You never touch prices or PnL. The engine does fills and accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# What the engine hands you on each event
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketState:
    """One point-in-time snapshot. This is all you are allowed to see."""

    index: int                 # event number, 0-based
    ts_ns: int                 # exchange receive time, nanoseconds
    best_bid: float
    best_ask: float
    mid: float
    spread: float
    features: np.ndarray       # the feature vector for this event
    feature_names: Sequence[str]

    def f(self, name: str) -> float:
        """Grab one feature by name. Handy for rule-based strategies."""
        return float(self.features[self.feature_names.index(name)])


@dataclass
class Decision:
    """What a strategy returns. Position is in base units (BTC)."""

    target_position: float
    order_type: str = "market"      # "market" or "limit"
    limit_price: float | None = None
    reason: str = ""                # free text, lands in the trade log
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# The three things you implement
# --------------------------------------------------------------------------

@runtime_checkable
class Model(Protocol):
    """Short-horizon forecasting model. ML person owns this."""

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> "Model":
        """Train on a training slice only. Called once per walk-forward fold."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return shape (n, 3) of probabilities for [down, flat, up].

        Must work for n == 1 (live path) and n > 1 (batch scoring path).
        If your model is a regressor, wrap it so it still returns 3 columns.
        """
        ...

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "Model":
        """
        Optional. Called by the engine when the drift detector fires and the
        adaptation policy is "refit". If your model can't do incremental
        updates, just call fit() on the window you're given and return self.
        """
        ...


@runtime_checkable
class DriftDetector(Protocol):
    """Decides when the market has changed. ML person owns this."""

    name: str

    def reset(self, reference: np.ndarray) -> None:
        """Set/replace the reference window the detector compares against."""
        ...

    def update(self, x: np.ndarray, error: float | None = None) -> bool:
        """
        One event in. Return True on the event where drift is declared.

        `x` is the feature vector. `error` is the model's loss on the previous
        event if the engine has it (None during warmup), so you can build a
        feature-space detector, an error-based detector, or both.
        """
        ...

    @property
    def statistic(self) -> float:
        """Current test statistic. Logged every event for the plots."""
        ...


@runtime_checkable
class Strategy(Protocol):
    """Turns a forecast into a position. Strategy person owns this."""

    name: str

    def reset(self) -> None:
        ...

    def on_event(
        self,
        state: MarketState,
        proba: np.ndarray | None,
        drift_flag: bool,
        position: float,
    ) -> Decision:
        """
        Called once per event, in order.

        proba       : the model's [down, flat, up] for this event, or None
                      while the model is warming up
        drift_flag  : True on the event the detector fired
        position    : your current position in base units
        """
        ...


# --------------------------------------------------------------------------
# Convenience base classes, so nobody has to write boilerplate
# --------------------------------------------------------------------------

class BaseModel:
    name = "base"

    def fit(self, X, y, sample_weight=None):
        return self

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        out = np.zeros((len(X), 3))
        out[:, 1] = 1.0
        return out

    def partial_fit(self, X, y):
        return self.fit(X, y)


class BaseStrategy:
    name = "base"

    def reset(self):
        pass

    def on_event(self, state, proba, drift_flag, position):
        return Decision(target_position=0.0)


class BaseDetector:
    name = "base"
    _stat = 0.0

    def reset(self, reference):
        pass

    def update(self, x, error=None):
        return False

    @property
    def statistic(self):
        return self._stat


__all__ = [
    "MarketState",
    "Decision",
    "Model",
    "Strategy",
    "DriftDetector",
    "BaseModel",
    "BaseStrategy",
    "BaseDetector",
]
