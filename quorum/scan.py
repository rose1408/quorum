from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Optional

from .agents import (AccountState, DataQualityAgent, FlowAnalyst, FlowReport, LabReport, MarketFlow,
                     MarketObserver, PortfolioPlan, PortfolioRiskManager, StrategyLab)
from .agents.strategy_analyst import FAMILIES
from .config import QuorumConfig, build_coordinator
from .types import Decision, Vote


@dataclass(frozen=True)
class SymbolAnalysis:
    symbol: str
    flow: Optional[FlowReport]
    lab: Optional[LabReport]
    strategy: Optional[str]   # family the lab chose
    params: Optional[dict]
    decision: Decision
    warnings: tuple
    exam: tuple = ()          # holdout verdicts for the chosen strategy, setup or not

    @property
    def exam_passed(self) -> bool:
        return bool(self.exam) and all(v.vote == Vote.APPROVE for v in self.exam)


@dataclass(frozen=True)
class ScanResult:
    analyses: tuple       # one per market, in the order given
    flow: MarketFlow
    plan: PortfolioPlan


def analyze_symbol(symbol: str, candles: list, account: AccountState, config: QuorumConfig,
                   reference: Optional[list] = None) -> SymbolAnalysis:
    """The full pipeline for one market: data check, flow, strategy search, then the team's decision."""
    data = DataQualityAgent().check(candles, reference)
    warnings = tuple(data.warnings)
    if not data.ok:
        d = Decision(None, False, (), 0.0, "data check failed: " + "; ".join(data.issues), warnings)
        return SymbolAnalysis(symbol, None, None, None, None, d, warnings)

    observer = MarketObserver()
    flow = FlowAnalyst().analyze(symbol, candles, observer)
    lab = StrategyLab(grids=config.grids, holdout=config.holdout).build(
        symbol, candles, observer, config.costs, flow.structure)
    if lab.chosen is None:
        d = Decision(None, False, (), 0.0, "no strategy held up in research", warnings)
        return SymbolAnalysis(symbol, flow, lab, None, None, d, warnings)

    strategy = FAMILIES[lab.chosen.family](**lab.chosen.result.params)
    # a market that looks like a random walk has almost no chance of a real edge, so demand much stronger proof
    luck = config.random_max_luck if flow.structure == "random" else config.max_luck
    team = build_coordinator(config, strategy, with_state=False, max_luck=luck)
    exam = team.exam(symbol, candles)
    decision = team.decide(symbol, candles, account, reference)
    return SymbolAnalysis(symbol, flow, lab, lab.chosen.family, dict(lab.chosen.result.params), decision, warnings, exam)


def _run_one(args) -> SymbolAnalysis:
    return analyze_symbol(*args)


def scan_markets(markets: dict, account: AccountState, config: Optional[QuorumConfig] = None,
                 workers: int = 1, references: Optional[dict] = None) -> ScanResult:
    """Analyze several markets at once, then look across them.

    With `workers` above 1 each market runs in its own process, so three
    markets take about as long as one. Results do not depend on the worker count.
    """
    config = config or QuorumConfig()
    references = references or {}
    jobs = [(s, c, account, config, references.get(s)) for s, c in markets.items()]
    analyses = None
    if workers > 1 and len(jobs) > 1:
        try:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
                analyses = list(pool.map(_run_one, jobs))
        except (BrokenProcessPool, OSError, PermissionError):
            analyses = None  # cannot start processes here, fall back to one at a time
    if analyses is None:
        analyses = [_run_one(j) for j in jobs]

    reports = {a.symbol: a.flow for a in analyses if a.flow is not None}
    flow = FlowAnalyst().compare(reports, {s: markets[s] for s in reports})
    plan = PortfolioRiskManager(config.portfolio).allocate(analyses, flow, account)
    return ScanResult(tuple(analyses), flow, plan)
