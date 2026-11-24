from __future__ import annotations

from typing import Callable, Optional

from ..backtest import NO_COSTS, CostModel, expectancy, risk_of, simulate, simulate_trades
from ..types import Proposal, Verdict, Vote


class Validator:
    """The tester. Tries to break an idea before it goes anywhere.

    It replays the strategy over a holdout slice of the history (the most
    recent part), one trade at a time, with trading costs taken off, and
    judges the result in R multiples. Checks:

    - enough trades and positive expectancy after costs
    - walk-forward: the holdout is cut into windows, and enough of them must be profitable
    - robustness: nearby parameter settings must mostly be profitable too

    The harness only ever shows the strategy candles up to the current bar.
    Keep the holdout data out of any tuning of the strategy.
    """

    name = "validator"

    def __init__(self, holdout: float = 0.4, warmup: int = 40, min_trades: int = 20,
                 min_expectancy_r: float = 0.0, windows: int = 4,
                 min_positive_windows: float = 0.5, min_robust: float = 0.5):
        self.holdout = holdout
        self.warmup = warmup
        self.min_trades = min_trades
        self.min_expectancy_r = min_expectancy_r
        self.windows = windows
        self.min_positive_windows = min_positive_windows
        self.min_robust = min_robust

    def review(self, proposal: Optional[Proposal], candles: list,
               signal_fn: Callable[[list], Optional[Proposal]],
               costs: CostModel = NO_COSTS, neighbors=()) -> Verdict:
        if proposal is not None and risk_of(proposal) <= 0:
            return Verdict(self.name, Vote.REJECT, "stop is on the wrong side of entry")
        n = len(candles)
        start = max(self.warmup, int(n * (1 - self.holdout)))
        trades = simulate_trades(candles, signal_fn, start, None, costs)
        results = [r for _, r in trades]
        count = len(results)
        if count < self.min_trades:
            return Verdict(self.name, Vote.REJECT, f"only {count} holdout trades, need {self.min_trades}")

        exp = expectancy(results)
        win_rate = sum(1 for r in results if r > 0) / count if count else 0.0
        summary = f"{count} trades, win rate {win_rate:.0%}, expectancy {exp:.2f}R after costs"
        fails = []
        if exp <= self.min_expectancy_r:
            fails.append("not profitable: " + summary)

        span = max(1, n - start)
        buckets = [[] for _ in range(self.windows)]
        for i, r in trades:
            buckets[min(self.windows - 1, (i - start) * self.windows // span)].append(r)
        active = [b for b in buckets if b]
        positive = sum(1 for b in active if expectancy(b) > 0)
        if len(active) >= 2 and positive / len(active) < self.min_positive_windows:
            fails.append(f"edge not consistent: only {positive} of {len(active)} windows profitable")

        robust = None
        if neighbors:
            robust = sum(1 for nb in neighbors if expectancy(simulate(candles, nb, start, None, costs)) > 0)
            if robust / len(neighbors) < self.min_robust:
                fails.append(f"fragile: only {robust} of {len(neighbors)} nearby settings profitable")

        wins = sum(r for r in results if r > 0)
        losses = -sum(r for r in results if r <= 0)
        equity = peak = drawdown = 0.0
        for r in results:
            equity += r
            peak = max(peak, equity)
            drawdown = max(drawdown, peak - equity)
        metrics = {"trades": count, "expectancy_r": exp, "win_rate": win_rate,
                   "profit_factor": wins / losses if losses > 0 else None, "max_drawdown_r": drawdown,
                   "windows_positive": positive, "windows_active": len(active),
                   "robust_positive": robust, "robust_total": len(neighbors)}
        if fails:
            return Verdict(self.name, Vote.REJECT, "; ".join(fails), metrics=metrics)
        extra = f", {positive} of {len(active)} windows profitable"
        if robust is not None:
            extra += f", {robust} of {len(neighbors)} nearby settings profitable"
        return Verdict(self.name, Vote.APPROVE, "holdout replay passed: " + summary + extra, metrics=metrics)
