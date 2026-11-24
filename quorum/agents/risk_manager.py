from __future__ import annotations

from dataclasses import dataclass

from ..types import Proposal, Side, Verdict, Vote


@dataclass
class AccountState:
    equity: float
    daily_pnl: float = 0.0
    open_positions: int = 0
    kill_switch: bool = False


@dataclass(frozen=True)
class RiskLimits:
    risk_per_trade: float = 0.01  # fraction of equity risked per trade
    max_daily_loss: float = 0.03  # fraction of equity
    max_open_positions: int = 1
    min_reward_risk: float = 1.5


class RiskManager:
    """Owns position size and the hard limits. Can always say no."""

    name = "risk_manager"

    def __init__(self, limits: RiskLimits = RiskLimits()):
        self.limits = limits

    def review(self, p: Proposal, account: AccountState) -> Verdict:
        L = self.limits
        no = lambda why: Verdict(self.name, Vote.REJECT, why)
        if account.kill_switch:
            return no("kill switch is on")
        if account.equity <= 0:
            return no("no equity")
        if account.daily_pnl <= -L.max_daily_loss * account.equity:
            return no("daily loss limit reached")
        if account.open_positions >= L.max_open_positions:
            return no("max open positions reached")
        risk = p.entry - p.stop if p.side == Side.LONG else p.stop - p.entry
        reward = p.target - p.entry if p.side == Side.LONG else p.entry - p.target
        if risk <= 0 or reward <= 0:
            return no("stop or target on the wrong side of entry")
        if reward / risk < L.min_reward_risk:
            return no(f"reward/risk {reward / risk:.2f} below {L.min_reward_risk}")
        size = account.equity * L.risk_per_trade / risk
        return Verdict(self.name, Vote.APPROVE, f"risking {L.risk_per_trade:.1%} of equity", size)
