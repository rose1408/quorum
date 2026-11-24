from __future__ import annotations

from typing import Callable, Optional

from ..backtest import expectancy, risk_of, simulate
from ..types import Proposal, Verdict, Vote


class Validator:
    """The tester. Tries to break an idea before it goes anywhere.

    It replays the strategy over a holdout slice of the history (the most
    recent part), one trade at a time, and judges the result in R multiples.
    Keep the holdout data out of any tuning of the strategy.
    """

    name = "validator"

    def __init__(self, holdout: float = 0.4, warmup: int = 40, min_trades: int = 20, min_expectancy_r: float = 0.0):
        self.holdout = holdout
        self.warmup = warmup
        self.min_trades = min_trades
        self.min_expectancy_r = min_expectancy_r

    def review(self, proposal: Proposal, candles: list,
               signal_fn: Callable[[list], Optional[Proposal]]) -> Verdict:
        if risk_of(proposal) <= 0:
            return Verdict(self.name, Vote.REJECT, "stop is on the wrong side of entry")
        start = max(self.warmup, int(len(candles) * (1 - self.holdout)))
        results = simulate(candles, signal_fn, start)
        trades = len(results)
        if trades < self.min_trades:
            return Verdict(self.name, Vote.REJECT, f"only {trades} holdout trades, need {self.min_trades}")
        exp = expectancy(results)
        win_rate = sum(1 for r in results if r > 0) / trades
        summary = f"{trades} trades, win rate {win_rate:.0%}, expectancy {exp:.2f}R"
        if exp <= self.min_expectancy_r:
            return Verdict(self.name, Vote.REJECT, "holdout replay not profitable: " + summary)
        return Verdict(self.name, Vote.APPROVE, "holdout replay passed: " + summary)
