"""
Reference implementations.

These exist so that (a) the harness runs today, before the other two
workstreams land anything, and (b) every real result has something to beat.
The baselines are intentionally dumb. If the real model cannot beat
MajorityModel + ThresholdStrategy after costs, the result is not a result.

Replace by passing your own object with the same methods. Nothing here is
imported by the engine.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from .interfaces import BaseDetector, BaseModel, BaseStrategy, Decision

# ---------------------------------------------------------------- models


class MajorityModel(BaseModel):
    """Always predicts the training majority class. The accuracy floor."""

    name = "majority"

    def fit(self, X, y, sample_weight=None):
        c = np.bincount(np.asarray(y)[np.asarray(y) >= 0], minlength=3)
        self.p = c / max(c.sum(), 1)
        return self

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        return np.tile(getattr(self, "p", np.array([0.1, 0.8, 0.1])), (len(X), 1))


class LogitModel(BaseModel):
    """Multinomial logistic regression on standardised features."""

    name = "logit"

    def __init__(self, C: float = 1.0, max_iter: int = 400):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self._mk = lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=max_iter,
                               class_weight="balanced"),
        )
        self.pipe = self._mk()
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        m = y >= 0
        self.pipe = self._mk()
        self.pipe.fit(X[m], y[m])
        self.classes_ = self.pipe[-1].classes_
        return self

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        raw = self.pipe.predict_proba(X)
        out = np.zeros((len(X), 3))
        for j, c in enumerate(self.classes_):
            out[:, int(c)] = raw[:, j]
        return out

    def partial_fit(self, X, y):
        return self.fit(X, y)


class GBMModel(BaseModel):
    """
    Gradient boosting. Strong tabular baseline - hand this to the ML person as
    the bar their fancy model has to clear.
    """

    name = "gbm"

    def __init__(self, **kw):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self._kw = dict(max_depth=4, max_iter=150,
                        learning_rate=0.08, l2_regularization=1.0)
        self._kw.update(kw)
        self._cls = HistGradientBoostingClassifier
        self.clf = self._cls(**self._kw)

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        m = y >= 0
        self.clf = self._cls(**self._kw)
        self.clf.fit(X[m], y[m], sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        raw = self.clf.predict_proba(X)
        out = np.zeros((len(X), 3))
        for j, c in enumerate(self.clf.classes_):
            out[:, int(c)] = raw[:, j]
        return out

    def partial_fit(self, X, y):
        return self.fit(X, y)


# ------------------------------------------------------------- detectors


class RollingKSDetector(BaseDetector):
    """
    Two-sample KS test on one feature, reference window vs current window.

    The problem statement names this explicitly, so it is the baseline, not
    the contribution. Beat it with: multivariate (per-feature p-values with a
    Bonferroni or Benjamini-Hochberg correction), or an error-based detector
    like ADWIN/Page-Hinkley, or both in an OR gate.
    """

    name = "rolling_ks"

    def __init__(self, feature_idx: int = 0, window: int = 300,
                 alpha: float = 1e-3, cooldown: int = 500,
                 stride: int = 25):
        self.fi, self.w, self.alpha = feature_idx, window, alpha
        self.cooldown, self.stride = cooldown, stride
        self.ref = np.array([])
        self.buf: list[float] = []
        self.since = 10**9
        self._stat = 0.0
        self._t = 0

    def reset(self, reference):
        ref = np.atleast_2d(reference)
        self.ref = ref[:, self.fi].astype(float)
        self.buf, self.since, self._t = [], 0, 0

    def update(self, x, error=None):
        self._t += 1
        self.since += 1
        self.buf.append(float(np.atleast_1d(x)[self.fi]))
        if len(self.buf) > self.w:
            self.buf.pop(0)
        if (len(self.buf) < self.w or len(self.ref) < 30
                or self._t % self.stride or self.since < self.cooldown):
            return False
        stat, p = ks_2samp(self.ref, np.asarray(self.buf))
        self._stat = float(stat)
        if p < self.alpha:
            self.ref = np.asarray(self.buf).copy()   # adopt new regime
            self.since = 0
            return True
        return False


class PageHinkleyDetector(BaseDetector):
    """
    Error-based detector. Watches the model's own loss for a sustained upward
    shift. Catches degradation that leaves the feature distribution alone -
    the KS test is blind to that, which is a clean novelty point.
    """

    name = "page_hinkley"

    def __init__(self, delta: float = 0.005, threshold: float = 8.0,
                 alpha: float = 0.999, cooldown: int = 500):
        self.delta, self.threshold = delta, threshold
        self.alpha, self.cooldown = alpha, cooldown
        self.reset(None)

    def reset(self, reference=None):
        self.mean = 0.0
        self.n = 0
        self.cum = 0.0
        self.min_cum = 0.0
        self.since = 10**9
        self._stat = 0.0

    def update(self, x, error=None):
        self.since += 1
        if error is None:
            return False
        self.n += 1
        self.mean += (error - self.mean) / self.n
        self.cum = self.cum * self.alpha + (error - self.mean - self.delta)
        self.min_cum = min(self.min_cum, self.cum)
        self._stat = self.cum - self.min_cum
        if self.n > 100 and self._stat > self.threshold and self.since > self.cooldown:
            self.reset()
            self.since = 0
            return True
        return False


class MultiFeatureKSDetector(BaseDetector):
    """
    KS on every feature, Benjamini-Hochberg across them. Fires when any
    feature's corrected p-value clears alpha. Strictly stronger than the
    single-feature baseline and costs three lines of code - use it as the
    "goes beyond starter resources" claim.
    """

    name = "multi_ks"

    def __init__(self, window: int = 300, alpha: float = 1e-3,
                 cooldown: int = 500, stride: int = 25):
        self.w, self.alpha = window, alpha
        self.cooldown, self.stride = cooldown, stride
        self.ref = None
        self.buf: list[np.ndarray] = []
        self.since = 10**9
        self._stat = 0.0
        self._t = 0
        self.last_features: list[int] = []

    def reset(self, reference):
        self.ref = np.atleast_2d(reference).astype(float)
        self.buf, self.since, self._t = [], 0, 0

    def update(self, x, error=None):
        self._t += 1
        self.since += 1
        self.buf.append(np.atleast_1d(x).astype(float))
        if len(self.buf) > self.w:
            self.buf.pop(0)
        if (self.ref is None or len(self.buf) < self.w
                or self._t % self.stride or self.since < self.cooldown):
            return False
        cur = np.vstack(self.buf)
        ps, stats = [], []
        for j in range(cur.shape[1]):
            if np.ptp(self.ref[:, j]) == 0 and np.ptp(cur[:, j]) == 0:
                ps.append(1.0); stats.append(0.0); continue
            s, p = ks_2samp(self.ref[:, j], cur[:, j])
            ps.append(p); stats.append(s)
        ps = np.asarray(ps)
        self._stat = float(np.max(stats))
        order = np.argsort(ps)
        m = len(ps)
        bh = ps[order] <= self.alpha * (np.arange(1, m + 1) / m)
        if bh.any():
            self.last_features = order[:int(np.flatnonzero(bh).max()) + 1].tolist()
            self.ref = cur.copy()
            self.since = 0
            return True
        return False


# ------------------------------------------------------------ strategies


class FlatStrategy(BaseStrategy):
    """Does nothing. PnL must be exactly zero - a sanity check on the engine."""
    name = "flat"


class BuyAndHoldStrategy(BaseStrategy):
    name = "buy_and_hold"

    def __init__(self, size: float = 1.0):
        self.size = size

    def on_event(self, state, proba, drift_flag, position):
        return Decision(self.size, reason="hold")


class RandomStrategy(BaseStrategy):
    """
    Trades at the same rate as the real strategy but with no signal. The cost
    drag it shows is the bar: the real strategy must beat random by more than
    the fee it pays.
    """

    name = "random"

    def __init__(self, size=1.0, p_trade=0.01, seed=0):
        self.size, self.p = size, p_trade
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.rng = np.random.default_rng(0)

    def on_event(self, state, proba, drift_flag, position):
        if self.rng.random() < self.p:
            return Decision(self.rng.choice([-1.0, 0.0, 1.0]) * self.size,
                            reason="random")
        return Decision(position)


class ThresholdStrategy(BaseStrategy):
    """
    The obvious signal-to-position rule, and a fair baseline for the strategy
    person to beat.

    Long when p(up) - p(down) > enter, flat when it falls back inside exit.
    The band stops it from churning on noise, which on 100 ms data is where
    all the money goes.
    """

    name = "threshold"

    def __init__(self, size=1.0, enter=0.15, exit=0.05,
                 max_hold_events: int | None = None,
                 flatten_on_drift: bool = False):
        self.size, self.enter, self.exit = size, enter, exit
        self.max_hold = max_hold_events
        self.flatten_on_drift = flatten_on_drift
        self.held = 0

    def reset(self):
        self.held = 0

    def on_event(self, state, proba, drift_flag, position):
        if proba is None:
            return Decision(0.0, reason="warmup")
        if self.flatten_on_drift and drift_flag:
            self.held = 0
            return Decision(0.0, reason="drift_flatten")

        edge = float(proba[2] - proba[0])
        self.held = self.held + 1 if position != 0 else 0
        if self.max_hold and self.held >= self.max_hold:
            self.held = 0
            return Decision(0.0, reason="max_hold")

        if position == 0:
            if edge > self.enter:
                return Decision(self.size, reason=f"enter_long edge={edge:.3f}")
            if edge < -self.enter:
                return Decision(-self.size, reason=f"enter_short edge={edge:.3f}")
            return Decision(0.0, reason="no_signal")

        if position > 0 and edge < self.exit:
            return Decision(0.0, reason="exit_long")
        if position < 0 and edge > -self.exit:
            return Decision(0.0, reason="exit_short")
        return Decision(position, reason="hold")


class OracleStrategy(BaseStrategy):
    """
    Trades the true label. Not a baseline - a ceiling. Its PnL is the most
    money the label definition can possibly produce, so if the oracle itself
    barely makes money after costs then the label or the horizon is wrong and
    no model will save it. Run this first. It takes thirty seconds and it has
    saved entire projects.
    """

    name = "oracle"

    def __init__(self, y: np.ndarray, size: float = 1.0):
        self.y, self.size = y, size

    def on_event(self, state, proba, drift_flag, position):
        lab = self.y[state.index]
        if lab == 2:
            return Decision(self.size, reason="oracle_up")
        if lab == 0:
            return Decision(-self.size, reason="oracle_down")
        return Decision(0.0, reason="oracle_flat")
