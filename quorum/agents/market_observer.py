from __future__ import annotations

from statistics import mean, pstdev
from typing import Optional

from ..types import MarketSnapshot, Regime
from .data_quality import DataQualityAgent


class MarketObserver:
    """Watches the market: regime, volatility, ATR, moving averages, recent channel, stretch from the mean."""

    name = "market_observer"

    def __init__(self, fast: int = 10, slow: int = 30, trend_threshold: float = 0.0005,
                 atr_period: int = 14, channel_period: int = 20, quality: Optional[DataQualityAgent] = None):
        self.fast = fast
        self.slow = slow
        self.trend_threshold = trend_threshold
        self.atr_period = atr_period
        self.channel_period = channel_period
        self.quality = quality or DataQualityAgent()

    def observe(self, symbol: str, candles: list, check_data: bool = True) -> MarketSnapshot:
        """`check_data=False` skips the data checks. The Coordinator does that once up front."""
        problems = list(self.quality.check(candles).issues) if check_data else []
        last = candles[-1].close if candles else 0.0
        if len(candles) < self.slow + 1:
            problems.append("not enough candles")
        if problems:
            return MarketSnapshot(symbol, last, Regime.UNKNOWN, 0.0, 0.0, False, tuple(problems))

        need = max(4 * self.slow + 1, self.channel_period + 1, self.atr_period + 1)
        tail = candles[-need:]  # only what the math needs, so long histories stay fast
        closes = [c.close for c in tail]
        fast_ma = mean(closes[-self.fast:])
        slow_ma = mean(closes[-self.slow:])
        spread = (fast_ma - slow_ma) / slow_ma
        if spread > self.trend_threshold:
            regime = Regime.TRENDING_UP
        elif spread < -self.trend_threshold:
            regime = Regime.TRENDING_DOWN
        else:
            regime = Regime.RANGING

        def stdev_of(window: int) -> float:
            part = closes[-(window + 1):]
            return pstdev([b / a - 1 for a, b in zip(part, part[1:])])

        vol = stdev_of(self.slow)
        vol_state = "normal"
        if len(closes) >= 4 * self.slow + 1:
            long_vol = stdev_of(4 * self.slow)
            if long_vol > 0:
                ratio = vol / long_vol
                vol_state = "high" if ratio > 1.5 else "low" if ratio < 0.67 else "normal"

        window = tail[-(self.atr_period + 1):]
        true_ranges = [
            max(b.high - b.low, abs(b.high - a.close), abs(b.low - a.close))
            for a, b in zip(window, window[1:])
        ]
        atr = mean(true_ranges)
        prior = tail[-(self.channel_period + 1):-1] or tail[:-1]
        stretch = (last - slow_ma) / atr if atr > 0 else 0.0
        return MarketSnapshot(symbol, last, regime, vol, atr, True, (), vol_state, fast_ma, slow_ma,
                              max(c.high for c in prior), min(c.low for c in prior), stretch)
