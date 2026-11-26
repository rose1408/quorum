from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .agents import (DataQualityAgent, MarketObserver, PaperMonitor, ResearchAnalyst, RiskLimits, RiskManager, Skeptic,
                     TradeJournal, Validator)
from .agents.portfolio_risk import PortfolioLimits
from .agents.strategy_analyst import Strategy, TrendFollow
from .backtest import CostModel
from .coordinator import Coordinator

DEMO_COSTS = CostModel(spread=0.03, slippage=0.01)


@dataclass(frozen=True)
class QuorumConfig:
    """Every knob in one place so a scan is reproducible.

    Costs default to zero, which is optimistic. Set them to what your broker
    really charges before trusting any result.
    """

    costs: CostModel = CostModel()
    risk: RiskLimits = RiskLimits()
    portfolio: PortfolioLimits = PortfolioLimits()
    holdout: float = 0.4          # most recent share of history kept for the final exam
    min_trades: int = 20
    min_expectancy_r: float = 0.05  # must beat this after costs
    max_luck: float = 0.05        # the Skeptic rejects if more than this share of resamples show no edge
    random_max_luck: float = 0.01  # stricter bar when the structure test says the market is a random walk
    grids: Optional[dict] = None  # strategy lab search space, None means the built-in one


def build_coordinator(config: Optional[QuorumConfig] = None, strategy: Optional[Strategy] = None,
                      with_state: bool = True, max_luck: Optional[float] = None,
                      quality: Optional[DataQualityAgent] = None) -> Coordinator:
    """A full team for one market. `with_state` adds the Paper Monitor and Trade Journal."""
    config = config or QuorumConfig(costs=DEMO_COSTS)
    return Coordinator(
        MarketObserver(),
        strategy or TrendFollow(),
        Validator(holdout=config.holdout, min_trades=config.min_trades, min_expectancy_r=config.min_expectancy_r),
        RiskManager(config.risk),
        skeptic=Skeptic(holdout=config.holdout, min_trades=config.min_trades,
                        max_luck=config.max_luck if max_luck is None else max_luck),
        monitor=PaperMonitor() if with_state else None,
        journal=TradeJournal() if with_state else None,
        researcher=ResearchAnalyst(holdout=config.holdout),
        quality=quality,
        costs=config.costs,
    )
