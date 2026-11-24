from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..backtest import NO_COSTS, CostModel
from .research_analyst import ExperimentResult, ResearchAnalyst
from .strategy_analyst import FAMILIES

DEFAULT_GRIDS = {
    "trend": {"stop_atr": [1.5, 2.0, 3.0], "reward_risk": [1.5, 2.0, 3.0]},
    "breakout": {"stop_atr": [1.5, 2.0], "reward_risk": [1.5, 2.0, 3.0]},
    "mean_reversion": {"entry_stretch": [2.0, 2.5, 3.0], "stop_atr": [1.5, 2.0]},
}


@dataclass(frozen=True)
class LabCandidate:
    family: str
    result: ExperimentResult
    score: float     # the lower of training and validation expectancy
    eligible: bool   # enough trades, profitable on both slices, not flagged as overfit


@dataclass(frozen=True)
class LabReport:
    candidates: tuple  # every combination tested, best score first
    chosen: Optional[LabCandidate]
    tested: int
    notes: tuple


class StrategyLab:
    """Searches strategy families and settings for one market and picks a candidate.

    Like the Research Analyst, it only uses the older part of the history. A
    candidate must be profitable on both its training and validation slices
    (after costs) and is ranked by the weaker of the two. The recent holdout
    stays untouched for the Validator and Skeptic. It recommends and never trades.
    """

    name = "strategy_lab"

    def __init__(self, grids: Optional[dict] = None, holdout: float = 0.4, warmup: int = 40,
                 train_frac: float = 0.7, min_trades: int = 10):
        self.grids = grids or DEFAULT_GRIDS
        self.holdout = holdout
        self.warmup = warmup
        self.train_frac = train_frac
        self.min_trades = min_trades

    def build(self, symbol: str, candles: list, observer, costs: CostModel = NO_COSTS,
              structure: str = "unknown") -> LabReport:
        candidates, tested = [], 0
        for family, grid in self.grids.items():
            cls = FAMILIES[family]
            research = ResearchAnalyst(grid, self.holdout, self.warmup, self.train_frac, self.min_trades)
            report = research.run(symbol, candles, observer, lambda cls=cls, **p: cls(**p), costs)
            for r in report.results:
                tested += 1
                eligible = (r.train_trades >= self.min_trades and r.valid_trades >= self.min_trades
                            and r.train_expectancy > 0 and r.valid_expectancy > 0 and not r.overfit)
                candidates.append(LabCandidate(family, r, min(r.train_expectancy, r.valid_expectancy), eligible))
        candidates.sort(key=lambda c: c.score, reverse=True)
        chosen = next((c for c in candidates if c.eligible), None)

        notes = []
        if tested == 0:
            notes.append("not enough history to search for strategies")
        else:
            notes.append(f"searched {tested} strategy and setting combinations, so the best one looks "
                         "better than it really is; the untouched holdout is the real test")
            eligible_count = sum(1 for c in candidates if c.eligible)
            notes.append(f"{eligible_count} of {tested} passed training and validation")
            if chosen is None:
                notes.append("no combination held up, so there is nothing to trade here")
        if structure == "random":
            notes.append("this market looks like a random walk, so any candidate is probably noise")
        return LabReport(tuple(candidates), chosen, tested, tuple(notes))
