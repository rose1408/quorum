from __future__ import annotations

from typing import Optional

from ..types import MarketSnapshot, Proposal, Regime, Side


class StrategyAnalyst:
    """Turns a market snapshot into a trade idea, or nothing.

    The rule below is a PLACEHOLDER (trend-follow with an ATR stop) so the
    pipeline runs end to end. The real strategy gets ported in here later.
    """

    name = "strategy_analyst"

    def __init__(self, stop_atr: float = 2.0, reward_risk: float = 2.0):
        self.stop_atr = stop_atr
        self.reward_risk = reward_risk

    def propose(self, snapshot: MarketSnapshot) -> Optional[Proposal]:
        if not snapshot.data_ok or snapshot.atr <= 0:
            return None
        entry = snapshot.last_close
        dist = snapshot.atr * self.stop_atr
        if snapshot.regime == Regime.TRENDING_UP:
            return Proposal(snapshot.symbol, Side.LONG, entry, entry - dist,
                            entry + dist * self.reward_risk, "uptrend: fast MA above slow MA")
        if snapshot.regime == Regime.TRENDING_DOWN:
            return Proposal(snapshot.symbol, Side.SHORT, entry, entry + dist,
                            entry - dist * self.reward_risk, "downtrend: fast MA below slow MA")
        return None
