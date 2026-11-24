from __future__ import annotations

import random
from typing import Callable, Optional

from ..backtest import expectancy, simulate
from ..types import Proposal, Verdict, Vote


class Skeptic:
    """Argues against every idea. Asks "what if this edge is just luck?"

    The Validator checks the numbers. The Skeptic checks whether the numbers
    deserve to be trusted: is the edge stable, and could chance explain it?
    """

    name = "skeptic"

    def __init__(self, holdout: float = 0.4, warmup: int = 40, min_trades: int = 20,
                 resamples: int = 1000, max_luck: float = 0.2, seed: int = 11):
        self.holdout = holdout
        self.warmup = warmup
        self.min_trades = min_trades
        self.resamples = resamples
        self.max_luck = max_luck
        self.seed = seed

    def review(self, proposal: Proposal, candles: list,
               signal_fn: Callable[[list], Optional[Proposal]]) -> Verdict:
        start = max(self.warmup, int(len(candles) * (1 - self.holdout)))
        return self.assess(simulate(candles, signal_fn, start))

    def assess(self, results: list) -> Verdict:
        n = len(results)
        if n < self.min_trades:
            return Verdict(self.name, Vote.REJECT, f"only {n} trades, not enough evidence to judge")
        challenges = []

        half = n // 2
        first, second = expectancy(results[:half]), expectancy(results[half:])
        if (first > 0) != (second > 0):
            challenges.append(f"edge flips between halves ({first:.2f}R then {second:.2f}R)")

        rng = random.Random(self.seed)
        no_edge = sum(1 for _ in range(self.resamples)
                      if sum(rng.choice(results) for _ in range(n)) <= 0)
        luck = no_edge / self.resamples
        if luck > self.max_luck:
            challenges.append(f"{luck:.0%} of resamples show no edge, could be luck")

        streak = worst = 0
        for r in results:
            streak = streak + 1 if r <= 0 else 0
            worst = max(worst, streak)

        if challenges:
            return Verdict(self.name, Vote.REJECT, "; ".join(challenges))
        return Verdict(self.name, Vote.APPROVE,
                       f"could not break it: luck chance {luck:.0%}, worst losing streak {worst}")
