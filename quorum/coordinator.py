from __future__ import annotations

from typing import Optional

from .agents import (AccountState, ExperimentReport, MarketObserver, PaperMonitor, ResearchAnalyst,
                     RiskManager, Skeptic, StrategyAnalyst, TradeJournal, Validator)
from .ledger import PaperLedger
from .types import Decision, Vote


class Coordinator:
    """The AI Research Coordinator. The only component that talks to the agents.

    Quorum rule: every voting agent present (Validator, Risk Manager, and the
    Skeptic if configured) must approve. Any one of them can veto.
    The result is advice, never an order.
    """

    def __init__(self, observer: MarketObserver, analyst: StrategyAnalyst,
                 validator: Validator, risk: RiskManager, ledger: Optional[PaperLedger] = None, *,
                 skeptic: Optional[Skeptic] = None, monitor: Optional[PaperMonitor] = None,
                 journal: Optional[TradeJournal] = None, researcher: Optional[ResearchAnalyst] = None):
        self.observer = observer
        self.analyst = analyst
        self.validator = validator
        self.risk = risk
        self.ledger = ledger
        self.skeptic = skeptic
        self.monitor = monitor
        self.journal = journal
        self.researcher = researcher

    def decide(self, symbol: str, candles: list, account: AccountState) -> Decision:
        if self.monitor is not None:
            report = self.monitor.report()
            if report.halt:
                return self._done(Decision(None, False, (), 0.0,
                                           "paper monitor halted trading: " + "; ".join(report.notes)))

        snapshot = self.observer.observe(symbol, candles)
        if not snapshot.data_ok:
            return self._done(Decision(None, False, (), 0.0, "data check failed: " + "; ".join(snapshot.notes)))

        proposal = self.analyst.propose(snapshot)
        if proposal is None:
            return self._done(Decision(None, False, (), 0.0, f"no setup in {snapshot.regime.value} market"))

        def signal_fn(history):
            return self.analyst.propose(self.observer.observe(symbol, history))

        verdicts = [
            self.validator.review(proposal, candles, signal_fn),
            self.risk.review(proposal, account),
        ]
        if self.skeptic is not None:
            verdicts.append(self.skeptic.review(proposal, candles, signal_fn))
        verdicts = tuple(verdicts)

        approved = all(v.vote == Vote.APPROVE for v in verdicts)
        if approved:
            reason = "quorum reached"
            size = verdicts[1].size or 0.0
        else:
            reason = "vetoed: " + "; ".join(f"{v.agent}: {v.reason}" for v in verdicts if v.vote == Vote.REJECT)
            size = 0.0
        decision = Decision(proposal, approved, verdicts, size, reason)
        if approved and self.journal is not None:
            self.journal.log_decision(decision, snapshot.regime)
        return self._done(decision)

    def research(self, symbol: str, candles: list) -> ExperimentReport:
        if self.researcher is None:
            raise ValueError("no research analyst configured")
        analyst_type = type(self.analyst)
        return self.researcher.run(symbol, candles, self.observer, lambda **params: analyst_type(**params))

    def review_journal(self) -> list:
        if self.journal is None:
            raise ValueError("no trade journal configured")
        return self.journal.review()

    def _done(self, decision: Decision) -> Decision:
        if self.ledger is not None:
            self.ledger.record(decision)
        return decision
