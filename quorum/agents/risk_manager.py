from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..backtest import NO_COSTS, CostModel, cost_r
from ..types import Proposal, Side, Verdict, Vote


@dataclass
class AccountState:
    equity: float
    daily_pnl: float = 0.0
    open_positions: int = 0
    kill_switch: bool = False
    peak_equity: Optional[float] = None
    consecutive_losses: int = 0


@dataclass(frozen=True)
class RiskLimits:
    risk_per_trade: float = 0.01   # fraction of equity risked per trade
    max_daily_loss: float = 0.03   # fraction of equity
    max_drawdown: float = 0.10     # fraction below peak equity
    max_consecutive_losses: int = 4
    max_open_positions: int = 1
    min_reward_risk: float = 1.5
    max_cost_r: float = 0.25       # costs may not eat more than this share of the risk
    max_notional_multiple: Optional[float] = None  # cap on size * entry vs equity, off by default
    min_size: Optional[float] = None   # smallest tradable size on the broker, in units of the instrument
    size_step: Optional[float] = None  # sizes come in multiples of this; rounded down so risk is never exceeded


class RiskManager:
    """Owns position size and the hard limits. Hard-coded rules, never AI-decided. Can always say no."""

    name = "risk_manager"

    def __init__(self, limits: RiskLimits = RiskLimits()):
        self.limits = limits

    def review(self, p: Proposal, account: AccountState, costs: CostModel = NO_COSTS) -> Verdict:
        L = self.limits
        no = lambda why: Verdict(self.name, Vote.REJECT, why)
        if account.kill_switch:
            return no("kill switch is on")
        if account.equity <= 0:
            return no("no equity")
        if account.daily_pnl <= -L.max_daily_loss * account.equity:
            return no("daily loss limit reached")
        if account.peak_equity is not None and account.equity <= account.peak_equity * (1 - L.max_drawdown):
            return no(f"drawdown limit of {L.max_drawdown:.0%} reached")
        if account.consecutive_losses >= L.max_consecutive_losses:
            return no(f"{account.consecutive_losses} losses in a row, cooling off")
        if account.open_positions >= L.max_open_positions:
            return no("max open positions reached")
        risk = p.entry - p.stop if p.side == Side.LONG else p.stop - p.entry
        if risk <= 0:
            return no("stop on the wrong side of entry")
        if p.target is not None:  # trades that exit on a clock or a stop have no target to check
            reward = p.target - p.entry if p.side == Side.LONG else p.entry - p.target
            if reward <= 0:
                return no("target on the wrong side of entry")
            if reward / risk < L.min_reward_risk:
                return no(f"reward/risk {reward / risk:.2f} below {L.min_reward_risk}")
        c = cost_r(p, costs)
        if c > L.max_cost_r:
            return no(f"costs take {c:.0%} of the risk, limit is {L.max_cost_r:.0%}")
        size = account.equity * L.risk_per_trade / risk
        note = f"risking {L.risk_per_trade:.1%} of equity"
        if L.max_notional_multiple is not None:
            cap = account.equity * L.max_notional_multiple / p.entry
            if size > cap:
                size, note = cap, note + ", size capped by exposure limit"
        if L.size_step:
            size = math.floor(size / L.size_step + 1e-9) * L.size_step
        if L.min_size is not None and size < L.min_size:
            return no(f"the right size for this stop is below the smallest tradable size of {L.min_size:g}; "
                      "the account is too small for this stop distance at this risk per trade")
        return Verdict(self.name, Vote.APPROVE, note, size)
