"""News strategies for fast markets like gold. They need a calendar of release times.

Candles cannot show what happens inside a minute, so these model a confirmed
move: they wait for a candle to close beyond the pre-news range and then enter
at that close. They do not model a pre-placed order on both sides of the
release, which would need tick data. Costs are inflated around each release.
"""
from __future__ import annotations

from bisect import bisect_right
from typing import Optional

from ..backtest import Prefix
from ..clock import infer_timeframe
from ..types import Proposal, Side
from .strategy_analyst import Strategy


class _NewsStrategy(Strategy):
    PARAMS: tuple = ()
    _ts: list = []

    def bind(self, context):
        super().bind(context)
        self._ts = [e.ts for e in context.news if e.impact >= self.min_impact]
        return self

    def compatible(self, timeframe_s):
        if not self._ts:
            return "needs a news calendar (a CSV of release times), none was given"
        if timeframe_s and timeframe_s > 300:
            return f"needs 5-minute or finer candles to react to news (yours are {timeframe_s // 60} minutes)"
        return None

    def caveats(self) -> tuple:
        return ("models a confirmed move after the release, not an order placed before it",
                "spread and slippage around releases are a guess until you measure them on your broker",
                "candles hide what happens inside a minute, so fast whipsaws are under-counted")

    def _setup(self, symbol, history, observer):
        """Find the latest release, the pre-news range, the last pre-news candle, and calm and live volatility."""
        n = len(history)
        tf = infer_timeframe(history)
        if n < self.pre_bars + 2 or not tf or not self._ts:
            return None
        last = history[-1]
        i = bisect_right(self._ts, last.ts) - 1
        if i < 0:
            return None
        event_ts = self._ts[i]
        k = n - 1
        while k >= 0 and history[k].ts >= event_ts and n - 1 - k <= self.max_after:
            k -= 1
        if k < self.pre_bars or history[k].ts >= event_ts or n - 1 - k < 1:
            return None  # too long after the release, or no candle after it yet
        pre = history[k - self.pre_bars + 1:k + 1]
        calm = observer.observe(symbol, Prefix(history, k + 1), check_data=False)   # volatility before the release
        live = observer.observe(symbol, history, check_data=False)                  # volatility including the release
        if not calm.data_ok or calm.atr <= 0 or not live.data_ok:
            return None
        return tf, k, max(c.high for c in pre), min(c.low for c in pre), calm.atr, live.atr, last.ts + tf


class NewsBreakout(_NewsStrategy):
    """After a release, join the first candle that closes beyond the pre-news range."""

    name = "news_breakout"
    PARAMS = ("stop_atr", "reward_risk")
    ARGS = ("stop_atr", "reward_risk", "hold_bars", "pre_bars", "max_after", "min_impact")

    def __init__(self, stop_atr: float = 1.0, reward_risk: float = 1.5, hold_bars: Optional[int] = 15,
                 pre_bars: int = 10, max_after: int = 5, min_impact: int = 3):
        self.stop_atr = stop_atr
        self.reward_risk = reward_risk
        self.hold_bars = hold_bars
        self.pre_bars = pre_bars
        self.max_after = max_after
        self.min_impact = min_impact

    def signal(self, symbol: str, history, observer) -> Optional[Proposal]:
        setup = self._setup(symbol, history, observer)
        if setup is None:
            return None
        tf, k, high, low, calm_atr, live_atr, now = setup
        n, close = len(history), history[-1].close
        side = Side.LONG if close > high else Side.SHORT if close < low else None
        if side is None:
            return None
        for j in range(k + 1, n - 1):  # only the first breakout after the release counts
            cj = history[j].close
            if (side == Side.LONG and cj > high) or (side == Side.SHORT and cj < low):
                return None
        dist = live_atr * self.stop_atr  # the stop follows how wild the market is right now, not before the release
        sign = 1 if side == Side.LONG else -1
        exit_ts = now + int(self.hold_bars * tf) if self.hold_bars else None
        return Proposal(symbol, side, close, close - sign * dist, close + sign * dist * self.reward_risk,
                        "closed beyond the pre-news range after the release", exit_ts, now)


class NewsFade(_NewsStrategy):
    """After a release overshoots the pre-news range by a lot, fade back toward the middle of it."""

    name = "news_fade"
    PARAMS = ("spike_atr", "stop_atr")
    ARGS = ("spike_atr", "stop_atr", "hold_bars", "pre_bars", "max_after", "min_impact")

    def __init__(self, spike_atr: float = 1.5, stop_atr: float = 1.5, hold_bars: Optional[int] = 15,
                 pre_bars: int = 10, max_after: int = 5, min_impact: int = 3):
        self.spike_atr = spike_atr
        self.stop_atr = stop_atr
        self.hold_bars = hold_bars
        self.pre_bars = pre_bars
        self.max_after = max_after
        self.min_impact = min_impact

    def signal(self, symbol: str, history, observer) -> Optional[Proposal]:
        setup = self._setup(symbol, history, observer)
        if setup is None:
            return None
        tf, k, high, low, calm_atr, live_atr, now = setup
        n, close = len(history), history[-1].close
        over = self.spike_atr * calm_atr
        side = Side.SHORT if close > high + over else Side.LONG if close < low - over else None
        if side is None:
            return None
        for j in range(k + 1, n - 1):
            cj = history[j].close
            if (side == Side.SHORT and cj > high + over) or (side == Side.LONG and cj < low - over):
                return None
        mid = (high + low) / 2
        dist = live_atr * self.stop_atr
        stop = close + dist if side == Side.SHORT else close - dist
        exit_ts = now + int(self.hold_bars * tf) if self.hold_bars else None
        return Proposal(symbol, side, close, stop, mid,
                        "spiked far past the pre-news range, fading back toward its middle", exit_ts, now)
