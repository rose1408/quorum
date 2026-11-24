from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .types import Proposal, Side


@dataclass(frozen=True)
class CostModel:
    """Trading costs in price units. Zero by default, which is optimistic."""

    spread: float = 0.0    # full spread paid once per round trip
    slippage: float = 0.0  # per side
    fee: float = 0.0       # per round trip

    def round_trip(self) -> float:
        return self.spread + 2 * self.slippage + self.fee


NO_COSTS = CostModel()


def risk_of(p: Proposal) -> float:
    return p.entry - p.stop if p.side == Side.LONG else p.stop - p.entry


def cost_r(p: Proposal, costs: CostModel) -> float:
    """Round-trip cost as a fraction of the planned risk."""
    r = risk_of(p)
    return costs.round_trip() / r if r > 0 else float("inf")


def walk(p: Proposal, candles: list, start: int, costs: CostModel = NO_COSTS):
    """Follow a trade from `start`. Returns (net R multiple, exit index) or None if unresolved.

    Stops are checked before targets inside a candle, which is the pessimistic choice.
    Costs come off every trade, win or lose.
    """
    risk = risk_of(p)
    reward = abs(p.target - p.entry)
    c = costs.round_trip() / risk
    for j in range(start, len(candles)):
        k = candles[j]
        if p.side == Side.LONG:
            if k.low <= p.stop:
                return -1.0 - c, j
            if k.high >= p.target:
                return reward / risk - c, j
        else:
            if k.high >= p.stop:
                return -1.0 - c, j
            if k.low <= p.target:
                return reward / risk - c, j
    return None


def simulate_trades(candles: list, signal_fn: Callable[[list], Optional[Proposal]],
                    start: int, end: Optional[int] = None, costs: CostModel = NO_COSTS) -> list:
    """Replay a strategy one trade at a time between `start` and `end`.

    The signal only ever sees candles up to the current bar. Trades still open
    at `end` are dropped. Returns a list of (entry index, net R multiple).
    """
    end = len(candles) if end is None else end
    view = candles[:end]
    trades, i = [], start
    while i < end - 1:
        p = signal_fn(view[: i + 1])
        if p is None or risk_of(p) <= 0:
            i += 1
            continue
        out = walk(p, view, i + 1, costs)
        if out is None:
            break
        r, exit_idx = out
        trades.append((i, r))
        i = exit_idx + 1
    return trades


def simulate(candles: list, signal_fn: Callable[[list], Optional[Proposal]],
             start: int, end: Optional[int] = None, costs: CostModel = NO_COSTS) -> list:
    return [r for _, r in simulate_trades(candles, signal_fn, start, end, costs)]


def expectancy(results: list) -> float:
    return sum(results) / len(results) if results else 0.0
