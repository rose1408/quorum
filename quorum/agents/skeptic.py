from __future__ import annotations

import random
from typing import Callable, Optional

from ..backtest import NO_COSTS, CostModel, drift_adjusted_r, expectancy, simulate_detail
from ..types import Proposal, Verdict, Vote


class Skeptic:
    """Argues against every idea. Asks "what if this edge is just luck?"

    The Validator checks the numbers. The Skeptic checks whether the numbers
    deserve to be trusted: is the edge stable, does it hang on one lucky
    trade, and could chance explain it?
    """

    name = "skeptic"

    def __init__(self, holdout: float = 0.4, warmup: int = 40, min_trades: int = 20,
                 resamples: int = 1000, max_luck: float = 0.05, seed: int = 11):
        self.holdout = holdout
        self.warmup = warmup
        self.min_trades = min_trades
        self.resamples = resamples
        self.max_luck = max_luck
        self.seed = seed

    def review(self, proposal: Optional[Proposal], candles: list,
               signal_fn: Callable[[list], Optional[Proposal]],
               costs: CostModel = NO_COSTS, neighbors=()) -> Verdict:
        start = max(self.warmup, int(len(candles) * (1 - self.holdout)))
        detail = simulate_detail(candles, signal_fn, start, None, costs)
        # judged after removing the market's own drift, so being long in a bull run is not mistaken for skill
        return self.assess(drift_adjusted_r(detail, candles, start))

    def assess(self, results: list) -> Verdict:
        n = len(results)
        if n < self.min_trades:
            return Verdict(self.name, Vote.REJECT, f"only {n} trades, not enough evidence to judge")
        challenges = []

        half = n // 2
        first, second = expectancy(results[:half]), expectancy(results[half:])
        if (first > 0) != (second > 0):
            challenges.append(f"edge flips between halves ({first:.2f}R then {second:.2f}R)")

        rest = list(results)
        rest.remove(max(results))
        if expectancy(results) > 0 and expectancy(rest) <= 0:
            challenges.append("edge depends on a single trade")

        rng = random.Random(self.seed)
        no_edge = sum(1 for _ in range(self.resamples)
                      if sum(rng.choice(results) for _ in range(n)) <= 0)
        luck = no_edge / self.resamples
        if luck > self.max_luck:
            challenges.append(f"{luck:.0%} of resamples show no edge, could be luck")

        streak = worst = 0
        equity = peak = drawdown = 0.0
        for r in results:
            streak = streak + 1 if r <= 0 else 0
            worst = max(worst, streak)
            equity += r
            peak = max(peak, equity)
            drawdown = max(drawdown, peak - equity)

        if challenges:
            return Verdict(self.name, Vote.REJECT, "; ".join(challenges))
        return Verdict(self.name, Vote.APPROVE,
                       f"could not break it: luck chance {luck:.0%}, worst losing streak {worst}, "
                       f"max drawdown {drawdown:.1f}R")
