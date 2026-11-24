from __future__ import annotations

from typing import Callable, Optional

from .types import Proposal, Side


def risk_of(p: Proposal) -> float:
    return p.entry - p.stop if p.side == Side.LONG else p.stop - p.entry


def walk(p: Proposal, candles: list, start: int):
    """Follow a trade from `start`. Returns (R multiple, exit index) or None if unresolved.

    Stops are checked before targets inside a candle, which is the pessimistic choice.
    """
    risk = risk_of(p)
    reward = abs(p.target - p.entry)
    for j in range(start, len(candles)):
        c = candles[j]
        if p.side == Side.LONG:
            if c.low <= p.stop:
                return -1.0, j
            if c.high >= p.target:
                return reward / risk, j
        else:
            if c.high >= p.stop:
                return -1.0, j
            if c.low <= p.target:
                return reward / risk, j
    return None


def simulate(candles: list, signal_fn: Callable[[list], Optional[Proposal]],
             start: int, end: Optional[int] = None) -> list:
    """Replay a strategy one trade at a time between `start` and `end`.

    The signal only ever sees candles up to the current bar. Trades still open
    at `end` are dropped. Returns the list of R multiples.
    """
    end = len(candles) if end is None else end
    view = candles[:end]
    results, i = [], start
    while i < end - 1:
        p = signal_fn(view[: i + 1])
        if p is None or risk_of(p) <= 0:
            i += 1
            continue
        out = walk(p, view, i + 1)
        if out is None:
            break
        r, exit_idx = out
        results.append(r)
        i = exit_idx + 1
    return results


def expectancy(results: list) -> float:
    return sum(results) / len(results) if results else 0.0
