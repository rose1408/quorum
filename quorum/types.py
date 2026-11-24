from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class Candle:
    ts: int  # unix seconds
    open: float
    high: float
    low: float
    close: float


class Regime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    UNKNOWN = "unknown"


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class Vote(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    last_close: float
    regime: Regime
    volatility: float  # stdev of recent returns
    atr: float  # average true range
    data_ok: bool
    notes: tuple = ()
    vol_state: str = "normal"  # low, normal, or high vs the longer-run average
    fast_ma: float = 0.0
    slow_ma: float = 0.0
    channel_high: float = 0.0  # highest high of the previous channel_period candles
    channel_low: float = 0.0
    stretch: float = 0.0       # (close - slow MA) / ATR


@dataclass(frozen=True)
class Proposal:
    symbol: str
    side: Side
    entry: float
    stop: float
    target: float
    rationale: str


@dataclass(frozen=True)
class Verdict:
    agent: str
    vote: Vote
    reason: str
    size: Optional[float] = None
    metrics: Optional[dict] = None


@dataclass(frozen=True)
class Decision:
    proposal: Optional[Proposal]
    approved: bool
    verdicts: tuple
    size: float
    reason: str
    warnings: tuple = ()
