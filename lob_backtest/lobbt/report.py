"""
Plots and the run report.

The four-panel figure is the one that goes in the slides: price with the shift
markers, rolling accuracy of static vs adaptive, the detector statistic with
its threshold, and cumulative PnL. One figure, and the whole claim is visible.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {
    "static": "#B4654A",
    "adaptive": "#2F6F6B",
    "price": "#4A5568",
    "flag": "#C0734F",
    "truth": "#2D3748",
    "grid": "#E2E8F0",
}


def _style(ax, title=None, ylabel=None):
    ax.grid(True, color=PALETTE["grid"], lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, loc="left", pad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)


def four_panel(
    df: pd.DataFrame,
    static_acc: pd.Series | None = None,
    adaptive_acc: pd.Series | None = None,
    detector_stat: pd.Series | None = None,
    flags: list[int] | None = None,
    true_shifts: list[int] | None = None,
    pnl: dict[str, pd.Series] | None = None,
    threshold: float | None = None,
    path: str = "outputs/four_panel.png",
    title: str = "Regime-adaptive LOB forecasting",
):
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1.5, 2]})
    x = np.arange(len(df))

    ax = axes[0]
    ax.plot(x, df["mid_price"].to_numpy(), color=PALETTE["price"], lw=0.8)
    _style(ax, "Mid price", "price")

    ax = axes[1]
    if static_acc is not None:
        ax.plot(x[:len(static_acc)], static_acc.to_numpy(),
                color=PALETTE["static"], lw=1.4, label="frozen model")
    if adaptive_acc is not None:
        ax.plot(x[:len(adaptive_acc)], adaptive_acc.to_numpy(),
                color=PALETTE["adaptive"], lw=1.4, label="adaptive model")
    ax.axhline(1 / 3, color="#9AA5B1", lw=0.8, ls=":", label="chance (3-class)")
    _style(ax, "Rolling accuracy", "hit rate")
    ax.legend(fontsize=8, frameon=False, ncol=3)

    ax = axes[2]
    if detector_stat is not None:
        ax.plot(x[:len(detector_stat)], detector_stat.to_numpy(),
                color=PALETTE["truth"], lw=1.0)
    if threshold is not None:
        ax.axhline(threshold, color=PALETTE["flag"], lw=0.9, ls="--",
                   label="threshold")
        ax.legend(fontsize=8, frameon=False)
    _style(ax, "Drift statistic", "stat")

    ax = axes[3]
    if pnl:
        for k, s in pnl.items():
            ax.plot(np.arange(len(s)), s.to_numpy(), lw=1.3, label=k)
        ax.axhline(0, color="#9AA5B1", lw=0.8)
        ax.legend(fontsize=8, frameon=False, ncol=max(1, len(pnl)))
    _style(ax, "Cumulative PnL after costs", "quote ccy")
    ax.set_xlabel("event index", fontsize=9)

    for a in axes:
        for f in (flags or []):
            a.axvline(f, color=PALETTE["flag"], lw=0.8, alpha=0.65)
        for s in (true_shifts or []):
            a.axvline(s, color=PALETTE["truth"], lw=1.2, ls="--", alpha=0.8)

    fig.suptitle(title, fontsize=12, x=0.09, ha="left", y=0.985)
    fig.text(0.09, 0.962,
             "dashed = true regime shift   |   solid vertical = detector fired",
             fontsize=8, color="#718096", ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def trade_diagnostics(res, path="outputs/trades.png"):
    """Slippage, inter-trade spacing, decision latency, position over time."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    tr, ev = res.trades, res.events

    if len(tr):
        ax[0, 0].hist(tr["slip_bps"], bins=40, color=PALETTE["static"],
                      alpha=0.85)
        _style(ax[0, 0], "Realised slippage vs mid", "trades")
        ax[0, 0].set_xlabel("bps", fontsize=8)

        gaps = tr["event"].diff().dropna()
        if len(gaps):
            ax[0, 1].hist(gaps, bins=40, color=PALETTE["adaptive"], alpha=0.85)
        _style(ax[0, 1], "Events between trades", "count")
    else:
        for a in (ax[0, 0], ax[0, 1]):
            a.text(0.5, 0.5, "no trades", ha="center", transform=a.transAxes)
            _style(a)

    ax[1, 0].plot(ev["position"].to_numpy(), color=PALETTE["price"], lw=0.7)
    _style(ax[1, 0], "Position", "base units")

    ax[1, 1].hist(ev["decide_us"], bins=50, color=PALETTE["truth"], alpha=0.8)
    _style(ax[1, 1], "Per-event decision time", "events")
    ax[1, 1].set_xlabel("microseconds", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def write_report(
    out_dir: str,
    runs: dict,
    forecast: dict | None = None,
    detector: dict | None = None,
    segments: pd.DataFrame | None = None,
    data_report=None,
    notes: list[str] | None = None,
) -> str:
    """Markdown run report plus a machine-readable results.json."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"# Backtest report", f"", f"_generated {stamp}_", ""]

    if data_report is not None:
        L += ["## Data", "```", str(data_report), "```", ""]

    L += ["## Economic results", ""]
    tbl = pd.DataFrame(runs).T
    keep = [c for c in ["final_pnl_quote", "pnl_bps_of_notional", "n_trades",
                        "total_fees", "sharpe_annualised",
                        "max_drawdown_quote", "time_in_market",
                        "avg_slip_bps", "median_decide_us"] if c in tbl.columns]
    L += [tbl[keep].round(4).to_markdown(), ""]

    if forecast:
        L += ["## Forecast quality", ""]
        f = {k: v for k, v in forecast.items() if k != "confusion"}
        L += [pd.Series(f).to_frame("value").round(4).to_markdown(), ""]
        if "confusion" in forecast:
            L += ["Confusion matrix (rows = true down/flat/up):", "",
                  "```", str(np.array(forecast["confusion"])), "```", ""]

    if detector:
        L += ["## Drift detector", ""]
        d = {k: v for k, v in detector.items()
             if not isinstance(v, (list, dict))}
        L += [pd.Series(d).to_frame("value").to_markdown(), ""]

    if segments is not None and len(segments):
        L += ["## Per-regime breakdown", "",
              segments.round(4).to_markdown(index=False), ""]

    if notes:
        L += ["## Notes and caveats", ""] + [f"- {n}" for n in notes] + [""]

    md = "\n".join(L)
    p = os.path.join(out_dir, "report.md")
    with open(p, "w") as fh:
        fh.write(md)

    def _j(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump({"runs": runs, "forecast": forecast, "detector": detector},
                  fh, indent=2, default=_j)
    return p
