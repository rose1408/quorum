from __future__ import annotations

from typing import Optional

from .agents import (AccountState, DataQualityAgent, ExperimentReport, MarketObserver, PaperMonitor,
                     ResearchAnalyst, RiskManager, Skeptic, StrategyAnalyst, TradeJournal, Validator)
from .backtest import NO_COSTS, CostModel
from .ledger import PaperLedger
from .types import Decision, Vote


class Coordinator:
    """The AI Research Coordinator. The only component that talks to the agents.

    Order of play: Paper Monitor halt check, Data Quality, Market Observer,
    Strategy Analyst, then the voting agents. Every voting agent present
    (Validator, Risk Manager, and the Skeptic if configured) must approve.
    Any one of them can veto. The result is advice, never an order.

    One CostModel is shared by every agent so they all judge with the same assumptions.
    """

    def __init__(self, observer: MarketObserver, analyst: StrategyAnalyst,
                 validator: Validator, risk: RiskManager, ledger: Optional[PaperLedger] = None, *,
                 skeptic: Optional[Skeptic] = None, monitor: Optional[PaperMonitor] = None,
                 journal: Optional[TradeJournal] = None, researcher: Optional[ResearchAnalyst] = None,
                 quality: Optional[DataQualityAgent] = None, costs: CostModel = NO_COSTS):
        self.observer = observer
        self.analyst = analyst
        self.validator = validator
        self.risk = risk
        self.ledger = ledger
        self.skeptic = skeptic
        self.monitor = monitor
        self.journal = journal
        self.researcher = researcher
        self.quality = quality or DataQualityAgent()
        self.costs = costs

    def _signal_fn(self, symbol: str, analyst):
        def signal_fn(history):
            return analyst.propose(self.observer.observe(symbol, history, check_data=False))
        return signal_fn

    def decide(self, symbol: str, candles: list, account: AccountState, reference: Optional[list] = None) -> Decision:
        if self.monitor is not None:
            report = self.monitor.report()
            if report.halt:
                return self._done(Decision(None, False, (), 0.0,
                                           "paper monitor halted trading: " + "; ".join(report.notes)))

        data = self.quality.check(candles, reference)
        warnings = tuple(data.warnings)
        if not data.ok:
            return self._done(Decision(None, False, (), 0.0, "data check failed: " + "; ".join(data.issues), warnings))

        snapshot = self.observer.observe(symbol, candles, check_data=False)
        if not snapshot.data_ok:
            return self._done(Decision(None, False, (), 0.0, "data check failed: " + "; ".join(snapshot.notes), warnings))

        proposal = self.analyst.propose(snapshot)
        if proposal is None:
            return self._done(Decision(None, False, (), 0.0, f"no setup in {snapshot.regime.value} market", warnings))

        signal_fn = self._signal_fn(symbol, self.analyst)
        variants = self.analyst.variants() if hasattr(self.analyst, "variants") else []
        neighbors = [self._signal_fn(symbol, a) for a in variants]

        verdicts = [
            self.validator.review(proposal, candles, signal_fn, costs=self.costs, neighbors=neighbors),
            self.risk.review(proposal, account, costs=self.costs),
        ]
        if self.skeptic is not None:
            verdicts.append(self.skeptic.review(proposal, candles, signal_fn, costs=self.costs))
        verdicts = tuple(verdicts)

        approved = all(v.vote == Vote.APPROVE for v in verdicts)
        if approved:
            reason = "quorum reached"
            size = verdicts[1].size or 0.0
        else:
            reason = "vetoed: " + "; ".join(f"{v.agent}: {v.reason}" for v in verdicts if v.vote == Vote.REJECT)
            size = 0.0
        decision = Decision(proposal, approved, verdicts, size, reason, warnings)
        if approved:
            if self.journal is not None:
                self.journal.log_decision(decision, snapshot.regime)
            metrics = verdicts[0].metrics
            if self.monitor is not None and self.monitor.expected_r is None and metrics:
                self.monitor.set_expected(metrics["expectancy_r"])  # baseline for drift checks
        return self._done(decision)

    def research(self, symbol: str, candles: list) -> ExperimentReport:
        if self.researcher is None:
            raise ValueError("no research analyst configured")
        base = self.analyst.params
        analyst_type = type(self.analyst)
        return self.researcher.run(symbol, candles, self.observer,
                                   lambda **params: analyst_type(**{**base, **params}), costs=self.costs)

    def review_journal(self) -> list:
        if self.journal is None:
            raise ValueError("no trade journal configured")
        return self.journal.review()

    def _done(self, decision: Decision) -> Decision:
        if self.ledger is not None:
            self.ledger.record(decision)
        return decision
