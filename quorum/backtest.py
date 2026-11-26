from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
from itertools import islice
from typing import Callable, Optional

from .types import Proposal, Side


@dataclass(frozen=True)
class CostModel:
    """Trading costs in price units. Zero by default, which is optimistic.

    Beyond the basics it can model two things that matter for CFDs:
    news time, when spreads and slippage balloon, and overnight financing,
    charged for every 24 hours a position is held.
    """

    spread: float = 0.0    # full spread paid once per round trip
    slippage: float = 0.0  # per side
    fee: float = 0.0       # per round trip
    overnight_pct: float = 0.0     # financing per 24h held, as a fraction of the entry price
    news_times: tuple = ()         # sorted event times (unix seconds)
    news_window_s: int = 900       # spread and slippage are inflated this long either side of an event
    news_multiplier: float = 1.0

    def round_trip(self, ts: Optional[int] = None) -> float:
        mult = self.news_multiplier if (ts is not None and self._near_news(ts)) else 1.0
        return (self.spread + 2 * self.slippage) * mult + self.fee

    def _near_news(self, ts: int) -> bool:
        if not self.news_times:
            return False
        i = bisect_left(self.news_times, ts - self.news_window_s)
        return i < len(self.news_times) and self.news_times[i] <= ts + self.news_window_s

    def holding_cost(self, entry_price: float, seconds: float) -> float:
        return self.overnight_pct * entry_price * max(0.0, seconds) / 86400

    def with_news(self, times, multiplier: float, window_s: int = 900) -> "CostModel":
        return replace(self, news_times=tuple(sorted(times)), news_multiplier=multiplier, news_window_s=window_s)


NO_COSTS = CostModel()


class Prefix:
    """The first n candles of a list, without copying them. What a strategy is allowed to see at one moment."""

    __slots__ = ("_c", "_n")

    def __init__(self, candles: list, n: int):
        self._c = candles
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, k):
        if isinstance(k, slice):
            a, b, step = k.indices(self._n)
            return self._c[a:b:step]
        if k < 0:
            k += self._n
        if not 0 <= k < self._n:
            raise IndexError("candle index out of range")
        return self._c[k]

    def __iter__(self):
        return islice(self._c, self._n)


@dataclass(frozen=True)
class Trade:
    entry: int   # index of the candle the signal fired on
    exit: int    # index of the candle the trade ended on
    r: float     # net result in R multiples, after costs
    risk: float  # price distance from entry to stop
    side: Side


def risk_of(p: Proposal) -> float:
    return p.entry - p.stop if p.side == Side.LONG else p.stop - p.entry


def cost_r(p: Proposal, costs: CostModel) -> float:
    """Round-trip cost as a fraction of the planned risk (financing not included)."""
    r = risk_of(p)
    return costs.round_trip(p.ts) / r if r > 0 else float("inf")


def walk(p: Proposal, candles: list, start: int, costs: CostModel = NO_COSTS):
    """Follow a trade from candle `start`. Returns (net R multiple, exit index) or None if unresolved.

    - Stops are checked before targets inside a candle, the pessimistic choice.
    - A candle that opens beyond the stop (a gap) fills at its open, so gaps cost more than 1R.
    - A time exit fills at the open of the first candle that starts at or after `exit_ts`.
    - Costs come off every trade: spread, slippage, and financing for the time held.
    """
    risk = risk_of(p)
    long = p.side == Side.LONG
    entry_ts = candles[start - 1].ts if start > 0 else None
    base = costs.round_trip(entry_ts)
    for j in range(start, len(candles)):
        k = candles[j]
        held = (k.ts - entry_ts) if entry_ts is not None else 0
        cost = (base + costs.holding_cost(p.entry, held)) / risk
        if p.exit_ts is not None and k.ts >= p.exit_ts:
            move = k.open - p.entry if long else p.entry - k.open
            return move / risk - cost, j
        if long:
            if k.low <= p.stop:
                return (min(p.stop, k.open) - p.entry) / risk - cost, j
            if p.target is not None and k.high >= p.target:
                return (p.target - p.entry) / risk - cost, j
        else:
            if k.high >= p.stop:
                return (p.entry - max(p.stop, k.open)) / risk - cost, j
            if p.target is not None and k.low <= p.target:
                return (p.entry - p.target) / risk - cost, j
    return None


def simulate_detail(candles: list, signal_fn: Callable[[list], Optional[Proposal]],
                    start: int, end: Optional[int] = None, costs: CostModel = NO_COSTS) -> list:
    """Replay a strategy one trade at a time between `start` and `end`.

    The signal only ever sees candles up to the current bar. Trades still open
    at `end` are dropped. Returns a list of Trade.
    """
    end = len(candles) if end is None else end
    view = candles[:end]
    trades, i = [], start
    while i < end - 1:
        p = signal_fn(Prefix(view, i + 1))
        if p is None or risk_of(p) <= 0:
            i += 1
            continue
        out = walk(p, view, i + 1, costs)
        if out is None:
            break
        r, j = out
        trades.append(Trade(i, j, r, risk_of(p), p.side))
        i = j + 1
    return trades


def simulate_trades(candles: list, signal_fn, start: int, end: Optional[int] = None,
                    costs: CostModel = NO_COSTS) -> list:
    """(entry index, net R) pairs."""
    return [(t.entry, t.r) for t in simulate_detail(candles, signal_fn, start, end, costs)]


def simulate(candles: list, signal_fn, start: int, end: Optional[int] = None,
             costs: CostModel = NO_COSTS) -> list:
    return [t.r for t in simulate_detail(candles, signal_fn, start, end, costs)]


def drift_adjusted_r(trades: list, candles: list, start: int, end: Optional[int] = None) -> list:
    """Each trade's R after removing what simply being in the market would have earned.

    A long-only strategy in a rising market looks good just for being long. This
    takes the market's average move per candle over the test period, and
    subtracts it for every candle a trade was open. What is left is the part
    that came from timing, not from the market drifting.
    """
    end = len(candles) if end is None else end
    span = end - 1 - start
    if span <= 0 or not trades:
        return [t.r for t in trades]
    per_bar = (candles[end - 1].close - candles[start].close) / span
    return [t.r - (1 if t.side == Side.LONG else -1) * per_bar * max(0, t.exit - t.entry) / t.risk
            for t in trades]


def expectancy(results: list) -> float:
    return sum(results) / len(results) if results else 0.0
