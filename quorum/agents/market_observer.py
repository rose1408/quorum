from __future__ import annotations

from statistics import mean, median, pstdev

from ..types import Candle, MarketSnapshot, Regime


class MarketObserver:
    """Watches the data. Checks it is clean, then labels the regime."""

    name = "market_observer"

    def __init__(self, fast: int = 10, slow: int = 30, trend_threshold: float = 0.0005, atr_period: int = 14):
        self.fast = fast
        self.slow = slow
        self.trend_threshold = trend_threshold
        self.atr_period = atr_period

    def check_data(self, candles: list) -> list:
        problems = []
        for i, c in enumerate(candles):
            if min(c.open, c.high, c.low, c.close) <= 0:
                problems.append(f"non-positive price at index {i}")
            if c.high < c.low or c.high < max(c.open, c.close) or c.low > min(c.open, c.close):
                problems.append(f"inconsistent OHLC at index {i}")
            if i > 0 and c.ts <= candles[i - 1].ts:
                problems.append(f"timestamps not increasing at index {i}")
        if len(candles) > 2 and not problems:
            gaps = [b.ts - a.ts for a, b in zip(candles, candles[1:])]
            typical = median(gaps)
            if typical > 0 and max(gaps) > 3 * typical:
                problems.append("gap larger than 3x the typical candle interval")
        return problems

    def observe(self, symbol: str, candles: list) -> MarketSnapshot:
        problems = self.check_data(candles)
        last = candles[-1].close if candles else 0.0
        if len(candles) < self.slow + 1:
            problems.append("not enough candles")
        if problems:
            return MarketSnapshot(symbol, last, Regime.UNKNOWN, 0.0, 0.0, False, tuple(problems))

        closes = [c.close for c in candles]
        fast_ma = mean(closes[-self.fast:])
        slow_ma = mean(closes[-self.slow:])
        spread = (fast_ma - slow_ma) / slow_ma
        if spread > self.trend_threshold:
            regime = Regime.TRENDING_UP
        elif spread < -self.trend_threshold:
            regime = Regime.TRENDING_DOWN
        else:
            regime = Regime.RANGING

        recent = closes[-(self.slow + 1):]
        returns = [b / a - 1 for a, b in zip(recent, recent[1:])]
        vol = pstdev(returns)

        window = candles[-(self.atr_period + 1):]
        true_ranges = [
            max(b.high - b.low, abs(b.high - a.close), abs(b.low - a.close))
            for a, b in zip(window, window[1:])
        ]
        atr = mean(true_ranges)
        return MarketSnapshot(symbol, last, regime, vol, atr, True, ())
