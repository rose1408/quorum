from __future__ import annotations

from typing import Callable, Optional

from ..types import Candle, Proposal, Side, Verdict, Vote


class Validator:
    """The tester. Tries to break an idea before it goes anywhere.

    It replays the strategy over a holdout slice of the history (the most
    recent part), one trade at a time, and judges the result in R multiples.
    Stops are checked before targets inside a candle, which is the pessimistic
    choice. Keep the holdout data out of any tuning of the strategy.
    """

    name = "validator"

    def __init__(self, holdout: float = 0.4, warmup: int = 40, min_trades: int = 20, min_expectancy_r: float = 0.0):
        self.holdout = holdout
        self.warmup = warmup
        self.min_trades = min_trades
        self.min_expectancy_r = min_expectancy_r

    @staticmethod
    def _risk(p: Proposal) -> float:
        return p.entry - p.stop if p.side == Side.LONG else p.stop - p.entry

    def _walk(self, p: Proposal, candles: list, start: int):
        risk = self._risk(p)
        reward = abs(p.target - p.entry)
        for j in range(start, len(candles)):
            c = candles[j]
            if p.side == Side.LONG:
                if c.low <= p.stop:
                    return -1.0, j
                if c.high >= p.target:
                    return reward / risk, j
            else:
                if c.high >= p.stop:
                    return -1.0, j
                if c.low <= p.target:
                    return reward / risk, j
        return None

    def review(self, proposal: Proposal, candles: list,
               signal_fn: Callable[[list], Optional[Proposal]]) -> Verdict:
        if self._risk(proposal) <= 0:
            return Verdict(self.name, Vote.REJECT, "stop is on the wrong side of entry")
        n = len(candles)
        i = max(self.warmup, int(n * (1 - self.holdout)))
        results = []
        while i < n - 1:
            p = signal_fn(candles[: i + 1])
            if p is None or self._risk(p) <= 0:
                i += 1
                continue
            out = self._walk(p, candles, i + 1)
            if out is None:
                break
            r, exit_idx = out
            results.append(r)
            i = exit_idx + 1
        trades = len(results)
        if trades < self.min_trades:
            return Verdict(self.name, Vote.REJECT, f"only {trades} holdout trades, need {self.min_trades}")
        expectancy = sum(results) / trades
        win_rate = sum(1 for r in results if r > 0) / trades
        summary = f"{trades} trades, win rate {win_rate:.0%}, expectancy {expectancy:.2f}R"
        if expectancy <= self.min_expectancy_r:
            return Verdict(self.name, Vote.REJECT, "holdout replay not profitable: " + summary)
        return Verdict(self.name, Vote.APPROVE, "holdout replay passed: " + summary)
