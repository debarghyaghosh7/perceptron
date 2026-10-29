"""
Correctness tests for the harness.

These are the tests that make the backtest trustworthy. Run them before every
demo. A backtester nobody has tested is just an optimistic opinion.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lobbt import *  # noqa: E402
from lobbt.interfaces import Decision  # noqa: E402


@pytest.fixture(scope="module")
def synth():
    df, shifts = make_regimes(n=6000, seed=3)
    feats = [f for f in DEFAULT_FEATURES if f in df.columns]
    y, ret, _ = horizon_labels(df, horizon_ms=500, deadband_bps=0.3)
    return df, feats, y, shifts


# ------------------------------------------------------------------ engine
def test_flat_strategy_pnl_is_exactly_zero(synth):
    df, feats, y, _ = synth
    r = Backtester().run(df, feats, strategy=FlatStrategy())
    assert abs(r.meta["final_pnl"]) < 1e-12
    assert len(r.trades) == 0


def test_position_cap_is_respected(synth):
    df, feats, y, _ = synth

    class Greedy(BaseStrategy):
        name = "greedy"

        def on_event(self, state, proba, flag, position):
            return Decision(999.0)

    cfg = BacktestConfig(max_position=0.4, latency_ms=1)
    r = Backtester(cfg).run(df, feats, strategy=Greedy())
    assert r.events["position"].abs().max() <= 0.4 + 1e-12


def test_costs_make_a_churning_strategy_lose(synth):
    """A strategy with no edge that trades constantly must lose exactly fees."""
    df, feats, y, _ = synth

    class Churn(BaseStrategy):
        name = "churn"

        def __init__(self):
            self.k = 0

        def reset(self):
            self.k = 0

        def on_event(self, state, proba, flag, position):
            self.k += 1
            return Decision(0.1 if self.k % 2 else 0.0)

    cfg = BacktestConfig(latency_ms=1,
                         costs=CostModel(taker_fee_bps=2.0, slippage_ticks=1))
    r = Backtester(cfg).run(df, feats, strategy=Churn())
    assert r.meta["final_pnl"] < 0, "churning with fees must lose money"
    assert len(r.trades) > 100


def test_latency_delays_execution(synth):
    """With latency, the fill event must be strictly after the decision event."""
    df, feats, y, _ = synth
    r = Backtester(BacktestConfig(latency_ms=50.0)).run(
        df, feats, strategy=BuyAndHoldStrategy())
    assert len(r.trades)
    assert (r.trades["event"] > r.trades["decided_at"]).all()
    assert r.trades["latency_events"].min() >= 1


def test_zero_latency_fills_next_event(synth):
    df, feats, y, _ = synth
    r = Backtester(BacktestConfig(latency_ms=0.0)).run(
        df, feats, strategy=BuyAndHoldStrategy())
    assert r.trades["latency_events"].max() <= 2


def test_market_buy_fills_at_or_above_ask(synth):
    """No fill may be better than the touch. This is the anti-fantasy test."""
    df, feats, y, _ = synth
    r = Backtester(BacktestConfig(latency_ms=5)).run(
        df, feats, strategy=RandomStrategy(p_trade=0.05, seed=1), y=y)
    tr = r.trades
    bid = df["best_bid"].to_numpy()
    ask = df["best_ask"].to_numpy()
    buys = tr[tr["side"] > 0]
    sells = tr[tr["side"] < 0]
    assert (buys["price"].to_numpy() >= ask[buys["event"]] - 1e-9).all()
    assert (sells["price"].to_numpy() <= bid[sells["event"]] + 1e-9).all()


def test_slippage_is_adverse_not_favourable(synth):
    df, feats, y, _ = synth
    cfg = BacktestConfig(latency_ms=5, costs=CostModel(slippage_ticks=2.0))
    r = Backtester(cfg).run(df, feats,
                            strategy=RandomStrategy(p_trade=0.05, seed=2), y=y)
    assert r.trades["slip_bps"].mean() > 0, "slippage must cost, not pay"


# ------------------------------------------------------------------ labels
def test_labels_never_see_the_future_of_the_split(synth):
    df, feats, y, _ = synth
    folds = walk_forward(df, n_folds=3, label_horizon_ms=500, embargo_ms=500,
                         min_train=300)
    assert_no_leakage(folds, df, 500)        # raises on failure


def test_leakage_detector_actually_catches_leakage(synth):
    """Negative control: a deliberately leaky split must be rejected."""
    df, feats, y, _ = synth
    bad = [Fold(0, np.arange(0, 1000), np.arange(1000, 1500))]
    with pytest.raises(AssertionError):
        assert_no_leakage(bad, df, label_horizon_ms=60_000)


def test_horizon_labels_are_directionally_correct():
    df, _ = make_regimes(n=2000, seed=5)
    y, ret, valid = horizon_labels(df, horizon_events=10, deadband_bps=0.0)
    mid = df["mid_price"].to_numpy()
    for i in (50, 500, 1200):
        if y[i] < 0:
            continue
        actual = mid[i + 10] - mid[i]
        if y[i] == UP:
            assert actual > 0
        elif y[i] == DOWN:
            assert actual < 0
        else:
            assert abs(actual) < 1e-9


def test_triple_barrier_respects_barriers():
    df, _ = make_regimes(n=1500, seed=6)
    y, hold, valid = triple_barrier(df, up_bps=5, down_bps=5,
                                    max_horizon_ms=3000)
    assert set(np.unique(y[y >= 0])) <= {0, 1, 2}
    assert np.nanmax(hold) <= 3100


# ---------------------------------------------------------------- detector
def test_detector_fires_on_a_real_shift():
    df, shifts = make_regimes(n=30_000, seed=9)
    feats = [f for f in DEFAULT_FEATURES if f in df.columns]
    X = to_matrix(df, feats)
    det = MultiFeatureKSDetector(window=400, alpha=1e-4, cooldown=800)
    det.reset(X[:2000])
    flags = [i for i in range(2000, len(X)) if det.update(X[i])]
    s = score_detector(flags, shifts, tolerance_events=3000, n_events=len(X))
    assert s["n_detected"] >= 1, f"no shift detected; flags={flags[:20]}"


def test_detector_is_quiet_on_stationary_data():
    """False alarm control. A detector that always fires is worthless."""
    df, _ = make_regimes(
        n=20_000, seed=4,
        regimes=[dict(frac=1.0, vol_bps=1.0, spread_ticks=1.0,
                      obi_strength=0.3, drift_bps=0.0, intensity=1.0)])
    feats = [f for f in DEFAULT_FEATURES if f in df.columns]
    X = to_matrix(df, feats)
    det = MultiFeatureKSDetector(window=400, alpha=1e-6, cooldown=1000)
    det.reset(X[:2000])
    flags = [i for i in range(2000, len(X)) if det.update(X[i])]
    rate = len(flags) / len(X) * 1000
    assert rate < 1.0, f"{rate:.2f} false alarms per 1k events is too many"


# ------------------------------------------------------------------ oracle
def test_oracle_beats_every_real_strategy(synth):
    """Sanity on the ceiling: nothing may out-earn perfect foresight."""
    df, feats, y, _ = synth
    bt = Backtester(BacktestConfig(latency_ms=1))
    X = to_matrix(df, feats)
    m = y[:3000] >= 0
    mdl = LogitModel().fit(X[:3000][m], y[:3000][m])
    orc = bt.run(df, feats, strategy=OracleStrategy(y))
    real = bt.run(df, feats, model=mdl, strategy=ThresholdStrategy(), y=y)
    assert orc.meta["final_pnl"] >= real.meta["final_pnl"]


# -------------------------------------------------------------------- data
def test_loader_rejects_price_columns_as_features():
    df, _ = make_regimes(n=1500)
    p = "/tmp/_t.csv"
    df.drop(columns=["ts", "ts_ns"]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="not allowed as features"):
        load_lob(p, features=["mid_price", "obi_1"], verbose=False)


def test_loader_flags_short_data():
    df, _ = make_regimes(n=2000, event_ms=100)
    p = "/tmp/_t2.csv"
    df.drop(columns=["ts", "ts_ns"]).to_csv(p, index=False)
    _, _, qr = load_lob(p, verbose=False)
    assert any("minutes of data" in w for w in qr.warnings)


def test_determinism(synth):
    df, feats, y, _ = synth
    a = Backtester(BacktestConfig(seed=1)).run(
        df, feats, strategy=RandomStrategy(p_trade=0.02, seed=1), y=y)
    b = Backtester(BacktestConfig(seed=1)).run(
        df, feats, strategy=RandomStrategy(p_trade=0.02, seed=1), y=y)
    assert a.meta["final_pnl"] == b.meta["final_pnl"]
