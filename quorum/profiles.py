"""Per-market settings: timeframe, clock, costs, and which strategies to search.

THE COST NUMBERS AND SIZE LIMITS BELOW ARE PLACEHOLDER GUESSES. Replace them with
what your broker actually charges and allows (spread, slippage, the overnight
swap for index CFDs, the smallest tradable size). Results are only as honest as
these numbers. The expected daily break (05:00 or 06:00 Philippine time, which
is the US 5pm close in summer or winter) is also a guess used to catch files
whose timestamps are in the wrong timezone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .agents.strategy_lab import DEFAULT_GRIDS
from .backtest import CostModel


@dataclass(frozen=True)
class MarketProfile:
    name: str
    description: str
    timeframe_s: int
    tz_offset_hours: float = 8.0     # the clock the rules are written in (8 = Philippine time)
    day_start_hour: float = 7.0      # when a trading day starts on that clock
    session_market: bool = True      # closes nights and weekends; False for 24/7 markets
    costs: CostModel = CostModel()
    cost_note: str = ""
    news_multiplier: float = 1.0     # spread and slippage multiplier around news releases
    news_window_s: int = 900
    grids: Optional[dict] = None     # None means the built-in generic search
    min_trades: int = 20             # holdout trades the Validator and Skeptic need
    lab_min_trades: int = 10         # trades each search slice needs
    references: tuple = ()           # (family, params) rules to score exactly as given, next to whatever the search finds
    expected_break_minutes: tuple = ()  # where the daily break should start on the rule clock (first is the main one), used to catch a wrong timezone
    min_size: Optional[float] = None    # smallest tradable size, in units of the instrument
    size_step: Optional[float] = None


_SCALP = {
    "trend": {"stop_atr": [0.8, 1.2, 1.6], "reward_risk": [1.0, 1.5], "hold_bars": [15, 30]},
    "breakout": {"stop_atr": [0.8, 1.2, 1.6], "reward_risk": [1.0, 1.5], "hold_bars": [15, 30]},
    "mean_reversion": {"entry_stretch": [2.0, 2.5], "stop_atr": [0.8, 1.2], "hold_bars": [15, 30]},
    "news_breakout": {"stop_atr": [0.8, 1.2, 1.6], "reward_risk": [1.0, 1.5, 2.0], "hold_bars": [10, 20]},
    "news_fade": {"spike_atr": [1.0, 1.5, 2.0], "stop_atr": [1.0, 1.5], "hold_bars": [10, 20]},
}

_US30 = {
    "weekly_dip": {"ma_days": [10, 20, 25, 50], "stop_pct": [0.03, 0.05, 0.08]},
    "all_day_long": {"stop_pct": [0.02, 0.03, 0.05, 0.08]},
    **DEFAULT_GRIDS,
}

PROFILES = {
    "step": MarketProfile(
        "step", "Step Index, 5-minute candles, trades around the clock", 300,
        session_market=False, costs=CostModel(spread=0.3),
        cost_note="spread is a placeholder; read the real one off your platform"),
    "xau": MarketProfile(
        "xau", "XAUUSD, 1-minute candles, scalping, with extra costs around news", 60,
        costs=CostModel(spread=0.30, slippage=0.05),
        cost_note="spread, slippage, and the news multiplier are placeholders; measure them on your broker, "
                  "gold spreads can widen many times over at big releases",
        news_multiplier=5.0, news_window_s=600, grids=_SCALP, min_trades=30, lab_min_trades=15,
        expected_break_minutes=(360, 300), min_size=1.0, size_step=1.0),
    "us30": MarketProfile(
        "us30", "US30, 5-minute candles, includes your Monday-dip and all-day-long rules", 300,
        costs=CostModel(spread=3.0, slippage=1.0, overnight_pct=0.0001),
        cost_note="spread, slippage, and the overnight financing (0.01% per day) are placeholders; "
                  "check your broker's swap for long US30 positions",
        grids=_US30, lab_min_trades=8,
        references=(("weekly_dip", {}), ("all_day_long", {})),
        expected_break_minutes=(360, 300), min_size=0.01, size_step=0.01),
    "generic": MarketProfile("generic", "no market profile, generic search", 300, session_market=False),
}
