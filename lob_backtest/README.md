# lobbt — backtesting harness (Perceptron Hackathon 2026, PS 4)

My workstream. The ML person and the strategy person code against
`lobbt/interfaces.py` and never touch anything else.

```
pip install -r requirements.txt
python -m pytest tests/ -q          # 17 tests, all must pass before a demo
python examples/run_demo.py         # writes outputs/
```

## Division of labour

| Owner | Implements | Contract |
|---|---|---|
| ML | `Model` (`fit`, `predict_proba` → (n,3) for down/flat/up, `partial_fit`) | `interfaces.py` |
| ML | `DriftDetector` (`reset`, `update` → bool, `statistic`) | `interfaces.py` |
| Strategy | `Strategy` (`on_event` → `Decision(target_position)`) | `interfaces.py` |
| Me | labels, splits, costs, fills, latency, PnL, metrics, report | everything else |

Nobody imports anyone else's module. Swap in your class, the harness runs.

## What the harness guarantees

- **No lookahead.** `predict_proba` sees one row at a time, in order.
- **Latency.** A decision at event *i* fills at the first event ≥ `ts[i] + latency_ms`. Tested.
- **Costs.** Taker fee, slippage ticks, optional impact. Fills never beat the touch — tested both sides.
- **Leakage-proof splits.** Walk-forward with purge + embargo; `assert_no_leakage` raises, and there's a negative-control test proving the check works.
- **Baselines you must beat:** flat (PnL must be exactly 0), buy-and-hold, random-at-same-turnover, majority-class, and **oracle** — trades the true label, so it's the ceiling the label can earn.

## Modules

`data.py` loader + quality report · `labels.py` horizon & triple-barrier · `splits.py` walk-forward · `costs.py` fees/fills · `engine.py` event loop · `metrics.py` forecast + economic · `drift_eval.py` detection delay / FAR / degradation · `baselines.py` reference models, detectors, strategies · `synthetic.py` LOB with known shifts · `report.py` four-panel figure + markdown report

## Two findings from the first run — read these

**1. The real capture cannot support this problem statement.**
8.9 minutes, mid moves on 0.21% of events, spread pinned at exactly 1 tick the whole time. Labels come out 99.1% flat, so walk-forward accuracy is 0.975 while the majority-class baseline is 0.989 — the model is *worse than guessing "flat"*. And the **oracle loses money** (−50.9 quote): round-trip cost is 2.0 bps but the 90th-percentile forward move is ~0.000 bps. Perfect foresight can't pay the fees. No model fixes that; only more data does.

What to capture: several hours to days, ideally spanning a macro print or a liquidity event, and sample **on book change** (the `changed_only` file) rather than every 100 ms. Also drop `spread_bps` while it's constant — it's a dead feature, and `spread` prediction is degenerate on this sample. Predict direction.

**2. The synthetic path already demonstrates the claim.**
Frozen model across three injected regimes: accuracy **0.648 → 0.503 → 0.384**. The third regime flips the sign of the OBI→return relationship, so the frozen model goes *below chance* rather than merely flat — which is exactly the "silent degradation" the PS describes. Detector: 2/2 shifts caught, 0 missed, mean delay 599 events, 0.77 false alarms per 1k events. Use synthetic to score the detector (ground truth exists there), real data for the strategy result, and say so explicitly in the abstract.

## For the abstract (my sections)

Evaluation plan: forecast quality *and* economic quality reported separately, every result against five baselines plus the oracle ceiling, detector scored on delay/recall/FAR against injected shifts, per-regime breakdown table, walk-forward with purge+embargo.

Novelty beyond starter resources: the rolling KS test is named in the PS, so it's the baseline, not the contribution. `MultiFeatureKSDetector` (per-feature KS + Benjamini–Hochberg) and `PageHinkleyDetector` (error-based, catches degradation the feature-space test is blind to) are both in `baselines.py` — OR-gating the two is a cheap, defensible contribution.

Risks to state: short/calm data; fee drag exceeding the edge at this horizon; detector false alarms causing over-refitting; queue position unobservable in L2, so any maker fill is an assumption (`LimitFillModel.queue_factor`).
