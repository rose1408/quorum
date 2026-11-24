from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Optional

from ..backtest import NO_COSTS, CostModel, expectancy, simulate


@dataclass(frozen=True)
class ExperimentResult:
    params: dict
    train_trades: int
    train_expectancy: float
    valid_trades: int
    valid_expectancy: float
    overfit: bool


@dataclass(frozen=True)
class ExperimentReport:
    results: tuple
    best: Optional[ExperimentResult]
    notes: tuple


class ResearchAnalyst:
    """Runs parameter experiments and recommends. Never changes the live strategy.

    It only touches the older part of the history. The recent holdout slice is
    left alone so the Validator and Skeptic still judge on data nobody tuned on.
    """

    name = "research_analyst"

    DEFAULT_GRID = {"stop_atr": [1.5, 2.0, 3.0], "reward_risk": [1.5, 2.0, 3.0]}

    def __init__(self, grid: Optional[dict] = None, holdout: float = 0.4, warmup: int = 40,
                 train_frac: float = 0.7, min_trades: int = 10):
        self.grid = grid or self.DEFAULT_GRID
        self.holdout = holdout
        self.warmup = warmup
        self.train_frac = train_frac
        self.min_trades = min_trades

    def run(self, symbol: str, candles: list, observer, make_analyst: Callable,
            costs: CostModel = NO_COSTS) -> ExperimentReport:
        region = candles[: int(len(candles) * (1 - self.holdout))]
        split = int(len(region) * self.train_frac)
        if split <= self.warmup:
            return ExperimentReport((), None, ("not enough history to run experiments",))

        keys = list(self.grid)
        results = []
        for values in product(*(self.grid[k] for k in keys)):
            params = dict(zip(keys, values))
            analyst = make_analyst(**params)

            def signal_fn(history, analyst=analyst):
                return analyst.propose(observer.observe(symbol, history, check_data=False))

            train = simulate(region, signal_fn, self.warmup, split, costs)
            valid = simulate(region, signal_fn, split, len(region), costs)
            t_exp, v_exp = expectancy(train), expectancy(valid)
            overfit = t_exp > 0 and v_exp < 0.5 * t_exp
            results.append(ExperimentResult(params, len(train), t_exp, len(valid), v_exp, overfit))

        results.sort(key=lambda r: r.train_expectancy, reverse=True)
        best = next((r for r in results
                     if r.train_trades >= self.min_trades and r.valid_trades >= self.min_trades
                     and r.valid_expectancy > 0 and not r.overfit), None)
        profitable = sum(1 for r in results if r.valid_expectancy > 0)
        notes = [f"tested {len(results)} combinations, so the best one looks better than it really is",
                 f"{profitable} of {len(results)} combinations were profitable on the validation slice"]
        if best is None:
            notes.append("no combination held up on the validation slice, keep current settings")
        return ExperimentReport(tuple(results), best, tuple(notes))
