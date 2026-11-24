from __future__ import annotations

from typing import Optional

from .agents import AccountState, MarketObserver, RiskManager, StrategyAnalyst, Validator
from .ledger import PaperLedger
from .types import Decision, Vote


class Coordinator:
    """The only component that talks to the agents.

    Quorum rule: the Validator AND the Risk Manager must both approve.
    Either one can veto. The result is advice, never an order.
    """

    def __init__(self, observer: MarketObserver, analyst: StrategyAnalyst,
                 validator: Validator, risk: RiskManager, ledger: Optional[PaperLedger] = None):
        self.observer = observer
        self.analyst = analyst
        self.validator = validator
        self.risk = risk
        self.ledger = ledger

    def decide(self, symbol: str, candles: list, account: AccountState) -> Decision:
        snapshot = self.observer.observe(symbol, candles)
        if not snapshot.data_ok:
            return self._done(Decision(None, False, (), 0.0, "data check failed: " + "; ".join(snapshot.notes)))

        proposal = self.analyst.propose(snapshot)
        if proposal is None:
            return self._done(Decision(None, False, (), 0.0, f"no setup in {snapshot.regime.value} market"))

        def signal_fn(history):
            return self.analyst.propose(self.observer.observe(symbol, history))

        verdicts = (
            self.validator.review(proposal, candles, signal_fn),
            self.risk.review(proposal, account),
        )
        approved = all(v.vote == Vote.APPROVE for v in verdicts)
        if approved:
            reason = "quorum reached"
            size = verdicts[1].size or 0.0
        else:
            reason = "vetoed: " + "; ".join(f"{v.agent}: {v.reason}" for v in verdicts if v.vote == Vote.REJECT)
            size = 0.0
        return self._done(Decision(proposal, approved, verdicts, size, reason))

    def _done(self, decision: Decision) -> Decision:
        if self.ledger is not None:
            self.ledger.record(decision)
        return decision
