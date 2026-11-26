"""Clock and calendar helpers. All times are plain unix seconds plus a fixed UTC offset (no DST)."""
from __future__ import annotations

import time
from collections import Counter
from statistics import mean
from typing import Optional

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def local_ts(ts: int, offset_hours: float) -> int:
    return ts + int(offset_hours * 3600)


def weekday(ts: int, offset_hours: float) -> int:
    """Monday is 0. Unix day 0 (1970-01-01) was a Thursday."""
    return (local_ts(ts, offset_hours) // 86400 + 3) % 7


def minute_of_day(ts: int, offset_hours: float) -> int:
    return (local_ts(ts, offset_hours) % 86400) // 60


def local_midnight(ts: int, offset_hours: float) -> int:
    """The unix time of 00:00 local time on the local day containing ts."""
    return (local_ts(ts, offset_hours) // 86400) * 86400 - int(offset_hours * 3600)


def format_local(ts: int, offset_hours: float) -> str:
    t = time.gmtime(local_ts(ts, offset_hours))
    return f"{WEEKDAYS[t.tm_wday]} {t.tm_hour:02d}:{t.tm_min:02d}"


def infer_timeframe(history) -> Optional[int]:
    """The usual gap between candles in seconds, from the most recent ones."""
    n = len(history)
    if n < 2:
        return None
    tail = history[-31:]
    diffs = [b.ts - a.ts for a, b in zip(tail, tail[1:]) if b.ts > a.ts]
    return Counter(diffs).most_common(1)[0][0] if diffs else None


class DailyIndex:
    """Daily closes built from intraday candles, one trading day at a time.

    A trading day starts at `day_start_hour` local time, so US30 (which reopens
    around 7:00 Philippine time) gets one daily candle per session. Only
    completed days are kept, which means the current day never leaks into an
    average. It updates incrementally, so replaying a long history stays fast.
    """

    def __init__(self, offset_hours: float, day_start_hour: float):
        self.offset = offset_hours
        self.day_start = day_start_hour
        self.reset()

    def reset(self) -> None:
        self.closes: list = []   # closes of completed trading days
        self._day = None
        self._last_close = None
        self._n = 0
        self._first_ts = None
        self._last_ts = None

    def _day_of(self, ts: int) -> int:
        return (ts + int((self.offset - self.day_start) * 3600)) // 86400

    def update(self, history) -> None:
        n = len(history)
        if self._n and (n < self._n or history[0].ts != self._first_ts or history[self._n - 1].ts != self._last_ts):
            self.reset()
        for k in range(self._n, n):
            c = history[k]
            d = self._day_of(c.ts)
            if self._day is None:
                self._day = d
            elif d != self._day:
                self.closes.append(self._last_close)
                self._day = d
            self._last_close = c.close
        if n:
            self._first_ts = history[0].ts
            self._last_ts = history[n - 1].ts
        self._n = n

    def sma(self, days: int) -> Optional[float]:
        return mean(self.closes[-days:]) if len(self.closes) >= days else None
