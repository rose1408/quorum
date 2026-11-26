"""Rule-based strategies that run on a clock: the two US30 rule sets.

Times are on the clock in `Context.tz_offset_hours` (Philippine time by default,
a fixed UTC+8 with no daylight saving). A candle counts as "now" when it
closes, so on 5-minute candles the one that ends at 07:05 is the 7:05 candle.
Entries are filled at that candle's close, with costs charged on top.
"""
from __future__ import annotations

from typing import Optional

from ..clock import (DailyIndex, WEEKDAYS, infer_timeframe, local_midnight, minute_of_day, weekday)
from ..types import Proposal, Side
from .strategy_analyst import Strategy


class WeeklyDip(Strategy):
    """Buy the Monday dip below the daily average, hold until Wednesday morning.

    From `start_minute` on the entry weekday (7:04am Monday) until the next
    trading day starts, the market is watched against the average of the last
    `ma_days` completed trading days. Price above the average: wait. The first
    candle that closes below it: buy at market. If that never happens that
    day, there is no trade that week. Stop `stop_pct` below the entry, no
    take-profit, close at `exit_minute` on `exit_weekday` (Wednesday 4:55am).
    """

    name = "weekly_dip"
    PARAMS = ("ma_days", "stop_pct")
    ARGS = ("ma_days", "stop_pct", "entry_weekday", "exit_weekday", "start_minute", "exit_minute")

    def __init__(self, ma_days: int = 25, stop_pct: float = 0.05, entry_weekday: int = 0,
                 exit_weekday: int = 2, start_minute: int = 7 * 60 + 4, exit_minute: int = 4 * 60 + 55):
        self.ma_days = max(2, int(round(ma_days)))
        self.stop_pct = stop_pct
        self.entry_weekday = entry_weekday
        self.exit_weekday = exit_weekday
        self.start_minute = start_minute
        self.exit_minute = exit_minute
        self._index: Optional[DailyIndex] = None

    def compatible(self, timeframe_s):
        if timeframe_s and timeframe_s > 3600:
            return f"needs hourly or finer candles to time the entry (yours are {timeframe_s // 60} minutes)"
        return None

    def caveats(self) -> tuple:
        return ("the average uses daily candles that start at the market's reopening time on the rule clock; "
                "your own chart may roll its daily candle over at another time, which shifts the average a little",
                "times are fixed Philippine time; the US session moves one hour twice a year with daylight saving")

    def _daily(self, history) -> DailyIndex:
        ctx = self.context
        if self._index is None or (self._index.offset, self._index.day_start) != (ctx.tz_offset_hours, ctx.day_start_hour):
            self._index = DailyIndex(ctx.tz_offset_hours, ctx.day_start_hour)
        self._index.update(history)
        return self._index

    def _in_window(self, now: int) -> bool:
        off = self.context.tz_offset_hours
        wd, m = weekday(now, off), minute_of_day(now, off)
        day_start = int(self.context.day_start_hour * 60)
        return ((wd == self.entry_weekday and m >= self.start_minute)
                or (wd == (self.entry_weekday + 1) % 7 and m < day_start))

    def signal(self, symbol: str, history, observer) -> Optional[Proposal]:
        n = len(history)
        tf = infer_timeframe(history)
        if n < 3 or not tf:
            return None
        last = history[-1]
        now = last.ts + tf
        if not self._in_window(now):
            return None
        ma = self._daily(history).sma(self.ma_days)
        if ma is None or last.close >= ma:
            return None
        k = n - 2
        while k >= 0:  # only the first dip of the window counts, so one trade per week
            if not self._in_window(history[k].ts + tf):
                break
            if history[k].close < ma:
                return None
            k -= 1
        off = self.context.tz_offset_hours
        ahead = (self.exit_weekday - weekday(now, off)) % 7
        exit_ts = local_midnight(now, off) + ahead * 86400 + self.exit_minute * 60
        if exit_ts <= now:
            exit_ts += 7 * 86400
        entry = last.close
        return Proposal(symbol, Side.LONG, entry, entry * (1 - self.stop_pct), None,
                        f"closed below the {self.ma_days}-day average ({ma:.1f}) in the "
                        f"{WEEKDAYS[self.entry_weekday]} window: buying the dip",
                        exit_ts, now)


class AllDayLong(Strategy):
    """Buy every day at the open window, hold the whole US session, close next morning.

    Entry in the window after `entry_minute` (7:05 to 7:10am): buy at market on
    the first candle that closes inside it. No candle in the window means no
    trade that day. Stop `stop_pct` below the entry, no take-profit, close at
    `exit_minute` (5:50am) the next day.
    """

    name = "all_day_long"
    PARAMS = ("stop_pct",)
    ARGS = ("stop_pct", "entry_minute", "window_minutes", "exit_minute")

    def __init__(self, stop_pct: float = 0.05, entry_minute: int = 7 * 60 + 5, window_minutes: int = 5,
                 exit_minute: int = 5 * 60 + 50):
        self.stop_pct = stop_pct
        self.entry_minute = entry_minute
        self.window_minutes = window_minutes
        self.exit_minute = exit_minute

    def compatible(self, timeframe_s):
        if timeframe_s and timeframe_s > self.window_minutes * 60:
            return (f"needs {self.window_minutes}-minute or finer candles to see the entry window "
                    f"(yours are {timeframe_s // 60} minutes)")
        return None

    def caveats(self) -> tuple:
        return ("closing at 5:50am after a 7:05am entry holds the trade 22 hours 45 minutes, not exactly 22 hours",
                "it is long every day, so most of its result comes from the index drifting up; the drift check "
                "in the Validator subtracts that",
                "times are fixed Philippine time; the US session moves one hour twice a year with daylight saving")

    def signal(self, symbol: str, history, observer) -> Optional[Proposal]:
        n = len(history)
        tf = infer_timeframe(history)
        if n < 2 or not tf:
            return None
        off = self.context.tz_offset_hours
        now = history[-1].ts + tf
        m = minute_of_day(now, off)
        if not (self.entry_minute <= m <= self.entry_minute + self.window_minutes):
            return None
        prev = history[-2].ts + tf
        if local_midnight(prev, off) == local_midnight(now, off) and minute_of_day(prev, off) >= self.entry_minute:
            return None  # an earlier candle in the window already counted
        exit_ts = local_midnight(now, off) + self.exit_minute * 60
        if exit_ts <= now:
            exit_ts += 86400
        entry = history[-1].close
        return Proposal(symbol, Side.LONG, entry, entry * (1 - self.stop_pct), None,
                        "daily entry window: buy and hold the session", exit_ts, now)
