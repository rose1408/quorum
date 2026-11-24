from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Optional


@dataclass(frozen=True)
class FlowReport:
    symbol: str
    structure: str          # trending, mean-reverting, mixed, random, or unknown
    autocorr: float         # lag-1 autocorrelation of returns
    autocorr_z: float
    variance_ratio: float   # about 1 for a random walk, above 1 trending, below 1 mean-reverting
    variance_ratio_z: float
    trend_strength: float   # (fast MA - slow MA) / ATR, signed
    momentum: float         # recent move in ATR-per-sqrt-bar units, signed
    flow_score: float       # average of the two, positive = bullish flow
    bias: str               # bullish, bearish, or neutral
    vol_state: str
    range_position: float   # 0 = at the 50-bar low, 1 = at the 50-bar high
    notes: tuple


@dataclass(frozen=True)
class MarketFlow:
    reports: dict           # symbol -> FlowReport
    correlation: dict       # (symbol, symbol) sorted -> return correlation
    ranking: tuple          # symbols, strongest flow first
    notes: tuple

    def corr(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        return self.correlation.get(tuple(sorted((a, b))), 0.0)


def _returns(candles: list, lookback: int) -> list:
    closes = [c.close for c in candles[-(lookback + 1):]]
    return [math.log(b / a) for a, b in zip(closes, closes[1:])]


def _autocorr(r: list, lag: int = 1) -> float:
    m = mean(r)
    den = sum((x - m) ** 2 for x in r)
    if den <= 0:
        return 0.0
    return sum((r[i] - m) * (r[i - lag] - m) for i in range(lag, len(r))) / den


def _variance_ratio(r: list, q: int = 4):
    """Lo-MacKinlay variance ratio and its z-score. About 1 when moves have no memory."""
    n = len(r)
    if n < 4 * q:
        return 1.0, 0.0
    mu = sum(r) / n
    var1 = sum((x - mu) ** 2 for x in r) / (n - 1)
    if var1 <= 0:
        return 1.0, 0.0
    prefix = [0.0]
    for x in r:
        prefix.append(prefix[-1] + x)
    sums = [prefix[i] - prefix[i - q] for i in range(q, n + 1)]
    m = q * (n - q + 1) * (1 - q / n)
    vr = (sum((s - q * mu) ** 2 for s in sums) / m) / var1
    phi = 2 * (2 * q - 1) * (q - 1) / (3 * q * n)
    return vr, (vr - 1) / math.sqrt(phi)


def _corr(a: list, b: list) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    ma, mb = mean(a), mean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def _clip(x: float, lim: float = 3.0) -> float:
    return max(-lim, min(lim, x))


class FlowAnalyst:
    """Reads how price is flowing, per market and across markets.

    "Flow" here means what candles can show: trend strength, momentum,
    volatility, and how the markets move together. It is not order flow, which
    needs tick or volume data.

    The structure test matters most. If a market's moves have no memory (a
    random walk), no trend or mean-reversion rule can have a real edge there,
    and anything a backtest finds is probably noise. This is a rough screen,
    not proof.
    """

    name = "flow_analyst"

    def __init__(self, lookback: int = 1000, momentum_bars: int = 20, z_threshold: float = 2.5,
                 corr_alert: float = 0.7, min_overlap: int = 30):
        self.lookback = lookback
        self.momentum_bars = momentum_bars
        self.z_threshold = z_threshold
        self.corr_alert = corr_alert
        self.min_overlap = min_overlap

    def analyze(self, symbol: str, candles: list, observer) -> FlowReport:
        snap = observer.observe(symbol, candles, check_data=False)
        r = _returns(candles, self.lookback)
        if not snap.data_ok or len(r) < 60:
            return FlowReport(symbol, "unknown", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, "neutral",
                              snap.vol_state, 0.5, ("not enough clean history to read the flow",))
        n = len(r)
        ac = _autocorr(r)
        ac_z = ac * math.sqrt(n)
        vr, vr_z = _variance_ratio(r)
        t = self.z_threshold
        trending = vr_z > t or ac_z > t
        reverting = vr_z < -t or ac_z < -t
        structure = ("mixed" if trending and reverting else "trending" if trending
                     else "mean-reverting" if reverting else "random")

        trend_strength = _clip((snap.fast_ma - snap.slow_ma) / snap.atr)
        k = self.momentum_bars
        momentum = 0.0
        if len(candles) > k:
            momentum = _clip((candles[-1].close - candles[-1 - k].close) / (snap.atr * math.sqrt(k)))
        score = (trend_strength + momentum) / 2
        bias = "bullish" if score > 0.5 else "bearish" if score < -0.5 else "neutral"

        recent = candles[-50:]
        lo, hi = min(c.low for c in recent), max(c.high for c in recent)
        pos = (candles[-1].close - lo) / (hi - lo) if hi > lo else 0.5

        notes = []
        if structure == "random":
            notes.append("moves look like a random walk, so no trend or reversion edge is expected here")
        return FlowReport(symbol, structure, ac, ac_z, vr, vr_z, trend_strength, momentum, score, bias,
                          snap.vol_state, pos, tuple(notes))

    def compare(self, reports: dict, markets: dict) -> MarketFlow:
        """Correlation, lead-lag, and strength ranking across markets."""
        ranking = tuple(sorted(reports, key=lambda s: reports[s].flow_score, reverse=True))
        notes = []
        correlation = {}
        if len(markets) >= 2:
            maps = {s: {c.ts: c.close for c in cs} for s, cs in markets.items()}
            common = sorted(set.intersection(*(set(m) for m in maps.values())))[-(self.lookback + 1):]
            if len(common) - 1 < self.min_overlap:
                notes.append("too few overlapping candles to compare the markets")
            else:
                rets = {s: [math.log(maps[s][b] / maps[s][a]) for a, b in zip(common, common[1:])] for s in maps}
                names = list(rets)
                for i, a in enumerate(names):
                    for b in names[i + 1:]:
                        correlation[tuple(sorted((a, b)))] = _corr(rets[a], rets[b])
                n = len(common) - 1
                cut = 2.5 / math.sqrt(n)
                for a in names:
                    for b in names:
                        if a != b:
                            lead = _corr(rets[a][:-1], rets[b][1:])  # a today vs b next bar
                            if abs(lead) > max(cut, 0.1):
                                notes.append(f"{a} leads {b} by one bar (correlation {lead:+.2f})")
                if correlation:
                    avg = mean(abs(v) for v in correlation.values())
                    if avg >= self.corr_alert:
                        notes.append(f"the markets move together (average correlation {avg:.2f}), "
                                     "so several positions would be one bet")
                    elif avg < 0.2:
                        notes.append(f"the markets move independently (average correlation {avg:.2f}), "
                                     "which helps diversification")
        return MarketFlow(dict(reports), correlation, ranking, tuple(notes))
