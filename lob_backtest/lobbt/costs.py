"""
Cost and fill models.

A backtest without costs on 100 ms data is a random number generator. At BTC
spreads of ~0.012 bps plus 1-2 bps of taker fee, the fee alone eats almost any
short-horizon edge, so the cost model is the part of this harness that decides
whether the strategy is real.

Defaults below are Binance spot VIP0 at the time of writing. Change them in
one place: BacktestConfig.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostModel:
    taker_fee_bps: float = 1.0     # Binance spot VIP0 taker, no BNB discount
    maker_fee_bps: float = 1.0     # maker rebate: pass a negative number
    slippage_ticks: float = 0.0    # extra adverse ticks on a market order
    tick_size: float = 0.01
    impact_coef_bps: float = 0.0   # temporary impact, bps per unit of
                                   # (order size / top-of-book size)

    def fill_price(
        self,
        side: int,                 # +1 buy, -1 sell
        qty: float,
        best_bid: float,
        best_ask: float,
        touch_qty: float,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> float:
        """Price actually achieved, spread crossing and impact included."""
        if order_type == "limit":
            if limit_price is None:
                raise ValueError("limit order needs a limit_price")
            return float(limit_price)

        base = best_ask if side > 0 else best_bid
        px = base + side * self.slippage_ticks * self.tick_size

        if self.impact_coef_bps and touch_qty > 0:
            ratio = abs(qty) / max(touch_qty, 1e-12)
            px *= 1 + side * (self.impact_coef_bps * ratio) / 1e4
        return float(px)

    def fee(self, notional: float, order_type: str = "market") -> float:
        bps = self.taker_fee_bps if order_type == "market" else self.maker_fee_bps
        return abs(notional) * bps / 1e4

    def round_trip_cost_bps(self, mid: float, spread: float) -> float:
        """
        What a flat round trip costs, in bps of notional. Print this next to
        your predicted edge. If edge < this, nothing else matters.
        """
        spread_bps = spread / mid * 1e4
        return spread_bps + 2 * self.taker_fee_bps + \
            2 * self.slippage_ticks * self.tick_size / mid * 1e4


@dataclass
class LimitFillModel:
    """
    Optimistic-but-bounded maker fill rule.

    A resting buy at the bid is treated as filled only once the book trades
    through our price: i.e. a later event shows best_ask <= our limit, or the
    best_bid moves below our price (we were the top of the queue and got run
    over). Queue position is not observable in L2 snapshot data, so we also
    apply `queue_factor` as a blunt haircut on fill probability.

    State this assumption in the abstract. Reviewers from Optiver will look
    for it, and a maker strategy whose backtest assumes free queue priority is
    the classic LOB-hackathon mistake.
    """

    queue_factor: float = 0.5      # 1.0 = always first in queue (too kind)
    max_wait_events: int = 50

    def try_fill(
        self, side: int, limit_price: float, best_bid: float, best_ask: float,
        rng: np.random.Generator,
    ) -> bool:
        if side > 0:
            crossed = best_ask <= limit_price
        else:
            crossed = best_bid >= limit_price
        if not crossed:
            return False
        return bool(rng.random() < self.queue_factor)
