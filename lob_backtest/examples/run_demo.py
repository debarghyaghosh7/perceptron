"""
End-to-end demo: the exact shape of the Round 2 result.

Two experiments:

  A) SYNTHETIC, shifts known -> score the detector properly (delay, recall,
     false alarms) and show frozen vs adaptive accuracy.
  B) REAL capture -> run the full pipeline on the actual BTCUSDT file so the
     harness is proven against real data, costs and all.

Run:  python examples/run_demo.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lobbt import *  # noqa: E402

OUT = "outputs"
HORIZON_MS = 500.0


def batch_proba(model, X):
    return np.asarray(model.predict_proba(X))


# ===========================================================================
def experiment_synthetic():
    print("=" * 72)
    print("A) SYNTHETIC DATA - detector scoring with known ground truth")
    print("=" * 72)

    df, true_shifts = make_regimes(n=24_000, seed=11)
    feats = [f for f in DEFAULT_FEATURES if f in df.columns]
    y, ret_bps, _ = horizon_labels(df, horizon_ms=HORIZON_MS,
                                   deadband_bps=0.3)
    print("labels:", {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in label_summary(y).items()})
    print("true shifts at events:", true_shifts)

    X = to_matrix(df, feats)
    train_end = true_shifts[0] - 1000          # train inside regime 0 only
    m = y[:train_end] >= 0

    # ---- frozen model -----------------------------------------------------
    frozen = GBMModel().fit(X[:train_end][m], y[:train_end][m])
    p_frozen = batch_proba(frozen, X)

    # ---- adaptive model: refit whenever the detector fires ---------------
    cfg = BacktestConfig(latency_ms=5.0, max_position=1.0,
                         warmup_events=train_end,
                         costs=CostModel(taker_fee_bps=1.0, slippage_ticks=0.5))
    bt = Backtester(cfg)

    adaptive = GBMModel().fit(X[:train_end][m], y[:train_end][m])
    det = MultiFeatureKSDetector(window=400, alpha=1e-4, cooldown=800)
    res_adapt = bt.run(df, feats, model=adaptive,
                       strategy=ThresholdStrategy(enter=0.18, exit=0.05),
                       detector=det, y=y, adapt="refit", adapt_window=4000)

    # re-score adaptively-updated model offline for the accuracy curve
    p_adapt = _replay_adaptive(X, y, feats, train_end, res_adapt.meta["drift_events"])

    res_frozen = bt.run(df, feats, strategy=ThresholdStrategy(enter=0.18, exit=0.05),
                        y=y, proba_matrix=p_frozen)
    res_flat = bt.run(df, feats, strategy=FlatStrategy())
    res_oracle = bt.run(df, feats, strategy=OracleStrategy(y), y=y)

    flags = res_adapt.meta["drift_events"]
    print(f"\ndetector fired {len(flags)} times at {flags[:12]}")
    dsc = detection_delay_ms(flags, true_shifts, df["ts_ns"].to_numpy(),
                             tolerance_events=2000)
    for k, v in dsc.items():
        print(f"  {k:28s} {v}")

    acc_f = rolling_accuracy(y, p_frozen, window=1500)
    acc_a = rolling_accuracy(y, p_adapt, window=1500)
    deg = degradation_curve(acc_f, acc_a, true_shifts[0], window=1500)
    print("\ndegradation around first shift:")
    for k, v in deg.items():
        print(f"  {k:24s} {v if v is None else round(float(v), 4)}")

    runs = {
        "flat (sanity)": pnl_metrics(res_flat),
        "frozen model": pnl_metrics(res_frozen),
        "adaptive model": pnl_metrics(res_adapt),
        "oracle (ceiling)": pnl_metrics(res_oracle),
    }
    print("\n", pd.DataFrame(runs).T[
        ["final_pnl_quote", "n_trades", "total_fees",
         "pnl_bps_of_notional", "sharpe_annualised"]].round(3))

    four_panel(
        df, static_acc=acc_f, adaptive_acc=acc_a,
        detector_stat=res_adapt.events["drift_stat"],
        flags=flags, true_shifts=true_shifts,
        pnl={"frozen": res_frozen.events["pnl"],
             "adaptive": res_adapt.events["pnl"],
             "oracle": res_oracle.events["pnl"]},
        path=f"{OUT}/synthetic_four_panel.png",
        title="Synthetic regimes - frozen vs adaptive",
    )
    trade_diagnostics(res_adapt, f"{OUT}/synthetic_trades.png")

    segs = segment_metrics(res_frozen, y, p_frozen, true_shifts)
    print("\nfrozen model, per regime:\n", segs.round(4).to_string(index=False))

    write_report(
        f"{OUT}/synthetic", runs,
        forecast=forecast_metrics(y[train_end:], p_frozen[train_end:]),
        detector=dsc, segments=segs,
        notes=[
            "Synthetic data with injected shifts; used ONLY to score the "
            "detector, where ground-truth shift indices exist.",
            "Regime 2 flips the sign of the OBI-return relationship, which is "
            "why the frozen model goes below chance rather than merely flat.",
            f"Oracle PnL is the ceiling for this label "
            f"({HORIZON_MS:.0f} ms horizon); any model result must be read "
            f"against it.",
        ],
    )
    return deg


def _replay_adaptive(X, y, feats, train_end, flags):
    """Offline reconstruction of the adaptive model's out-of-sample probas."""
    p = np.full((len(X), 3), np.nan)
    bounds = [train_end] + [f for f in flags if f > train_end] + [len(X)]
    m0 = y[:train_end] >= 0
    mdl = GBMModel().fit(X[:train_end][m0], y[:train_end][m0])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        p[a:b] = mdl.predict_proba(X[a:b])
        lo = max(0, b - 4000)
        mm = y[lo:b] >= 0
        if mm.sum() > 100 and b < len(X):
            mdl = GBMModel().fit(X[lo:b][mm], y[lo:b][mm])
    return p


# ===========================================================================
def experiment_real(path="data/BTCUSDT_features_full.csv"):
    print("\n" + "=" * 72)
    print("B) REAL CAPTURE - full pipeline, honest numbers")
    print("=" * 72)

    df, feats, qr = load_lob(path)
    db = recommended_deadband_bps(df)
    y, ret_bps, _ = horizon_labels(df, horizon_ms=HORIZON_MS, deadband_bps=db)
    print(f"deadband = {db:.5f} bps (half median spread)")
    print("labels:", {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in label_summary(y).items()})

    X = to_matrix(df, feats)
    cost = CostModel(taker_fee_bps=1.0, slippage_ticks=0.5)
    print(f"round-trip cost = "
          f"{cost.round_trip_cost_bps(df.mid_price.median(), df.spread.median()):.3f} bps")
    print(f"label move size  = {np.nanpercentile(np.abs(ret_bps), 90):.3f} bps "
          f"(90th pct |forward return|)")

    folds = walk_forward(df, n_folds=4, label_horizon_ms=HORIZON_MS,
                         embargo_ms=HORIZON_MS, train_frac=0.5, min_train=400)
    assert_no_leakage(folds, df, HORIZON_MS)
    print(f"\n{len(folds)} leakage-free folds:")
    for f in folds:
        print("  ", f)

    # walk-forward out-of-sample probabilities
    p_oos = np.full((len(df), 3), np.nan)
    for f in folds:
        m = y[f.train_idx] >= 0
        if m.sum() < 100:
            continue
        mdl = GBMModel().fit(X[f.train_idx][m], y[f.train_idx][m])
        p_oos[f.test_idx] = mdl.predict_proba(X[f.test_idx])

    fm = forecast_metrics(y, p_oos)
    print("\nwalk-forward forecast quality:")
    for k, v in fm.items():
        if k != "confusion":
            print(f"  {k:24s} {round(v, 4) if isinstance(v, float) else v}")

    test_start = folds[0].test_idx[0]
    cfg = BacktestConfig(latency_ms=5.0, warmup_events=test_start, costs=cost)
    bt = Backtester(cfg)
    m = y[:test_start] >= 0
    mdl = GBMModel().fit(X[:test_start][m], y[:test_start][m])

    p_frozen = mdl.predict_proba(X)
    runs, pnls = {}, {}
    specs = [
        ("flat (sanity)", None, FlatStrategy(), None, "none", None),
        ("buy & hold", None, BuyAndHoldStrategy(), None, "none", None),
        ("random", None, RandomStrategy(p_trade=0.005), None, "none", None),
        ("model + threshold", None, ThresholdStrategy(enter=0.15, exit=0.05),
         None, "none", p_frozen),
        ("model + ks-adapt", GBMModel().fit(X[:test_start][m], y[:test_start][m]),
         ThresholdStrategy(enter=0.15, exit=0.05),
         MultiFeatureKSDetector(window=200, alpha=1e-3, cooldown=300),
         "refit", None),
        ("oracle (ceiling)", None, OracleStrategy(y), None, "none", None),
    ]
    for name, model, strat, det, adapt, pm in specs:
        r = bt.run(df, feats, model=model, strategy=strat, detector=det,
                   y=y, adapt=adapt, adapt_window=1500, proba_matrix=pm)
        runs[name] = pnl_metrics(r)
        pnls[name] = r.events["pnl"]
        print(f"  {name:20s} pnl={runs[name]['final_pnl_quote']:+9.4f}  "
              f"trades={runs[name]['n_trades']:4d}  "
              f"fees={runs[name]['total_fees']:.4f}  "
              f"flags={r.meta['n_drift_flags']}")

    assert abs(runs["flat (sanity)"]["final_pnl_quote"]) < 1e-9, \
        "flat strategy must produce exactly zero PnL"

    four_panel(
        df, static_acc=rolling_accuracy(y, p_oos, window=300),
        detector_stat=None, flags=[], true_shifts=[],
        pnl={k: v for k, v in pnls.items() if k != "oracle (ceiling)"},
        path=f"{OUT}/real_four_panel.png",
        title="BTCUSDT capture - walk-forward",
    )

    write_report(
        f"{OUT}/real", runs, forecast=fm, data_report=qr,
        notes=[
            f"Capture is only {qr.duration_s/60:.1f} minutes and the mid moves "
            f"on {qr.frac_mid_changed:.2%} of events. This is one calm regime: "
            f"it cannot support a regime-shift claim and is used here only to "
            f"validate the pipeline.",
            "spread_bps is effectively constant (always one tick), so spread "
            "prediction is degenerate on this sample. Predict direction, not "
            "spread, until wider-spread data is captured.",
            "Oracle PnL bounds what the label can earn. Compare every model "
            "result against it before claiming an edge.",
            "Costs: 1.0 bps taker fee, 0.5 tick slippage, 5 ms latency.",
        ],
    )


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    experiment_synthetic()
    experiment_real()
    print(f"\nwrote figures and reports to ./{OUT}/")
