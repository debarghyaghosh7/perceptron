# Backtest report

_generated 2026-10-12 13:44 UTC_

## Data
```
Data quality report
-------------------
rows                : 5313
duration            : 536.3 s (8.9 min)
median inter-event  : 100.0 ms
max inter-event gap : 205.6 ms
crossed books       : 0
zero spreads        : 0
duplicate update ids: 0
events w/ mid move  : 0.207%
NaNs                : {'mid_change': 1, 'simple_return_1': 1, 'log_return_1': 1, 'rolling_vol_100_events': 100, 'rolling_vol_500_events': 500, 'elapsed_ms': 1}

WARNINGS
  ! only 8.9 minutes of data. A regime-shift claim needs hours-to-days spanning at least one real volatility event.
  ! mid price moves on just 0.21% of events. Labels will be overwhelmingly 'flat' and accuracy will look great for the wrong reason. Either sample on book changes or label over a longer horizon.
```

## Economic results

|                   |   final_pnl_quote |   pnl_bps_of_notional |   n_trades |   total_fees |   sharpe_annualised |   max_drawdown_quote |   time_in_market |   avg_slip_bps |   median_decide_us |
|:------------------|------------------:|----------------------:|-----------:|-------------:|--------------------:|---------------------:|-----------------:|---------------:|-------------------:|
| flat (sanity)     |            0      |                0      |          0 |       0      |              0      |               0      |           0      |         0      |              5.173 |
| buy & hold        |           -4.5844 |               -0.552  |          1 |       8.3044 |            -61.8504 |             -12.3244 |           0.4986 |         0.0012 |              5.274 |
| random            |          -82.6297 |               -0.8291 |         10 |      99.6597 |           -591.007  |             -91.4656 |           0.293  |         0.0012 |              5.494 |
| model + threshold |        -2028.79   |               -1.0012 |        244 |    2026.36   |          -4103.11   |           -2028.79   |           0.0708 |         0.0012 |              6.476 |
| model + ks-adapt  |         -315.946  |               -1.0012 |         38 |     315.576  |          -1583.88   |            -315.946  |           0.0131 |         0.0012 |             32.756 |
| oracle (ceiling)  |          -50.8758 |               -0.6126 |         10 |      83.0458 |           -415.61   |             -54.1894 |           0.0054 |         0.0012 |             11.366 |

## Forecast quality

|                       |     value |
|:----------------------|----------:|
| n                     | 2379      |
| accuracy              |    0.9748 |
| balanced_f1           |    0.3291 |
| mcc                   |   -0.0096 |
| majority_baseline_acc |    0.9891 |
| edge_over_majority    |   -0.0143 |
| directional_auc       |    0.3333 |
| brier_up              |    0.0084 |

Confusion matrix (rows = true down/flat/up):

```
[[   0   18    0]
 [  19 2319   15]
 [   0    8    0]]
```

## Notes and caveats

- Capture is only 8.9 minutes and the mid moves on 0.21% of events. This is one calm regime: it cannot support a regime-shift claim and is used here only to validate the pipeline.
- spread_bps is effectively constant (always one tick), so spread prediction is degenerate on this sample. Predict direction, not spread, until wider-spread data is captured.
- Oracle PnL bounds what the label can earn. Compare every model result against it before claiming an edge.
- Costs: 1.0 bps taker fee, 0.5 tick slippage, 5 ms latency.
