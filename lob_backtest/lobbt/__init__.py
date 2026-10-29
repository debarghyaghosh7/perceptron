"""
lobbt - backtesting harness for regime-adaptive LOB forecasting.

Perceptron Hackathon 2026, Problem 4.

    from lobbt import *

    df, feats, qr = load_lob("data/BTCUSDT_features_full.csv")
    y, ret, _ = horizon_labels(df, horizon_ms=500,
                               deadband_bps=recommended_deadband_bps(df))
    folds = walk_forward(df, n_folds=5, label_horizon_ms=500)
    assert_no_leakage(folds, df, 500)

    bt = Backtester(BacktestConfig(latency_ms=5))
    res = bt.run(df, feats, model=GBMModel(), strategy=ThresholdStrategy(),
                 detector=MultiFeatureKSDetector(), y=y, adapt="refit")
    print(pnl_metrics(res))
"""

__version__ = "0.3.0"

from .baselines import (
    BuyAndHoldStrategy,
    FlatStrategy,
    GBMModel,
    LogitModel,
    MajorityModel,
    MultiFeatureKSDetector,
    OracleStrategy,
    PageHinkleyDetector,
    RandomStrategy,
    RollingKSDetector,
    ThresholdStrategy,
)
from .costs import CostModel, LimitFillModel
from .data import (
    DEFAULT_FEATURES,
    DataQualityReport,
    concat_sessions,
    load_lob,
    to_matrix,
)
from .drift_eval import (
    degradation_curve,
    detection_delay_ms,
    flags_to_regimes,
    score_detector,
)
from .engine import BacktestConfig, BacktestResult, Backtester
from .interfaces import (
    BaseDetector,
    BaseModel,
    BaseStrategy,
    Decision,
    DriftDetector,
    MarketState,
    Model,
    Strategy,
)
from .labels import (
    DOWN,
    FLAT,
    UP,
    horizon_labels,
    label_summary,
    recommended_deadband_bps,
    triple_barrier,
)
from .metrics import (
    compare,
    forecast_metrics,
    pnl_metrics,
    rolling_accuracy,
    segment_metrics,
)
from .report import four_panel, trade_diagnostics, write_report
from .splits import Fold, assert_no_leakage, single_split, walk_forward
from .synthetic import make_regimes

__all__ = [n for n in dir() if not n.startswith("_")]
