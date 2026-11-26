from __future__ import annotations

from typing import Optional

from .news import NewsBreakout, NewsFade
from .scheduled import AllDayLong, WeeklyDip
from .strategy_analyst import Breakout, Context, MeanReversion, Strategy, TrendFollow

FAMILIES = {
    "trend": TrendFollow,
    "breakout": Breakout,
    "mean_reversion": MeanReversion,
    "weekly_dip": WeeklyDip,
    "all_day_long": AllDayLong,
    "news_breakout": NewsBreakout,
    "news_fade": NewsFade,
}


def make_strategy(family: str, params: Optional[dict] = None, context: Optional[Context] = None) -> Strategy:
    return FAMILIES[family](**(params or {})).bind(context or Context())
