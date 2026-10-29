# Backtest report

_generated 2026-10-12 13:44 UTC_

## Economic results

|                  |   final_pnl_quote |   pnl_bps_of_notional |   n_trades |   total_fees |   sharpe_annualised |   max_drawdown_quote |   time_in_market |   avg_slip_bps |   median_decide_us |
|:-----------------|------------------:|----------------------:|-----------:|-------------:|--------------------:|---------------------:|-----------------:|---------------:|-------------------:|
| flat (sanity)    |              0    |                0      |          0 |         0    |               0     |                 0    |           0      |         0      |              7.763 |
| frozen model     |          -8469.81 |               -1.553  |        656 |      5453.94 |            -534.774 |             -8692.33 |           0.683  |         0.0019 |             10.862 |
| adaptive model   |         -57968    |               -1.016  |       6783 |     57054.6  |           -4120.71  |            -57980.4  |           0.4708 |         0.0022 |           3344.3   |
| oracle (ceiling) |          31262.7  |                0.6728 |       3607 |     46469.6  |            1832.06  |             -1767.82 |           0.6613 |         0.0021 |              7.713 |

## Forecast quality

|                       |      value |
|:----------------------|-----------:|
| n                     | 16591      |
| accuracy              |     0.4348 |
| balanced_f1           |     0.2368 |
| mcc                   |     0.0091 |
| majority_baseline_acc |     0.4994 |
| edge_over_majority    |    -0.0646 |
| directional_auc       |     0.5034 |
| brier_up              |     0.386  |

Confusion matrix (rows = true down/flat/up):

```
[[6973   98  168]
 [ 912   40  115]
 [7983  102  200]]
```

## Drift detector

|                            |        value |
|:---------------------------|-------------:|
| n_true_shifts              |     2        |
| n_detected                 |     2        |
| n_missed                   |     0        |
| recall                     |     1        |
| n_false_alarms             |    18        |
| precision                  |     0.1      |
| mean_delay_events          |   599        |
| median_delay_events        |   599        |
| max_delay_events           |   599        |
| false_alarms_per_1k_events |     0.765957 |
| mean_delay_ms              | 39690.7      |
| median_delay_ms            | 39690.7      |

## Per-regime breakdown

|   segment |   start |   end |    n |   accuracy |   balanced_f1 |   edge_over_majority |      pnl |   realised_vol |
|----------:|--------:|------:|-----:|-----------:|--------------:|---------------------:|---------:|---------------:|
|         0 |       0 |  7900 | 7900 |     0.6476 |        0.6279 |               0.28   | -2833.02 |         0.372  |
|         1 |    7900 | 15100 | 7200 |     0.5031 |        0.2292 |              -0.0029 |  1997.82 |         3.5228 |
|         2 |   15100 | 23500 | 8400 |     0.3842 |        0.1902 |              -0.1487 | -7619.16 |         1.1488 |

## Notes and caveats

- Synthetic data with injected shifts; used ONLY to score the detector, where ground-truth shift indices exist.
- Regime 2 flips the sign of the OBI-return relationship, which is why the frozen model goes below chance rather than merely flat.
- Oracle PnL is the ceiling for this label (500 ms horizon); any model result must be read against it.
