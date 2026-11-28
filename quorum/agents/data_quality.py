from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from statistics import median
from typing import Optional

from ..clock import minute_of_day


@dataclass(frozen=True)
class DataReport:
    ok: bool
    issues: tuple    # problems that block trading
    warnings: tuple  # worth knowing, not blocking


class DataQualityAgent:
    """Checks the candles before anyone else sees them. Bad data ruins every other agent."""

    name = "data_quality"

    def __init__(self, gap_factor: float = 3.0, spike_k: float = 8.0, flat_run: int = 5,
                 max_feed_diff: float = 0.001, session_market: bool = False,
                 closure_s: int = 3600, max_missing: float = 0.01, news_times: tuple = (),
                 news_window_s: int = 600):
        # spikes that reverse near a news release are normal there, not bad ticks
        self.news_times = tuple(sorted(news_times))
        self.news_window_s = news_window_s
        # session markets (gold, indices) close every night and weekend, so long gaps are normal there
        self.session_market = session_market
        self.closure_s = closure_s
        self.max_missing = max_missing
        self.gap_factor = gap_factor
        self.spike_k = spike_k
        self.flat_run = flat_run
        self.max_feed_diff = max_feed_diff

    def _near_news(self, ts: int) -> bool:
        if not self.news_times:
            return False
        i = bisect_left(self.news_times, ts - self.news_window_s)
        return i < len(self.news_times) and self.news_times[i] <= ts + self.news_window_s

    @staticmethod
    def _flag(label: str, idx: list) -> str:
        return f"{len(idx)} {label} (first at index {idx[0]})"

    def check(self, candles: list, reference: Optional[list] = None) -> DataReport:
        issues, warnings = [], []
        nonfinite, nonpos, ohlc, order = [], [], [], []
        for i, c in enumerate(candles):
            if not all(math.isfinite(x) for x in (c.open, c.high, c.low, c.close)):
                nonfinite.append(i)
                continue
            if min(c.open, c.high, c.low, c.close) <= 0:
                nonpos.append(i)
            if c.high < c.low or c.high < max(c.open, c.close) or c.low > min(c.open, c.close):
                ohlc.append(i)
            if i > 0 and c.ts <= candles[i - 1].ts:
                order.append(i)
        if candles and max(candles[0].ts, candles[-1].ts) > 4102444800:
            issues.append("timestamps are far in the future, so they are probably milliseconds and not unix seconds")
        elif candles and min(candles[0].ts, candles[-1].ts) < 631152000:
            warnings.append("timestamps start before 1990, check that the time column is unix seconds or a real date")
        if nonfinite:
            issues.append(self._flag("candles with missing or infinite prices", nonfinite))
        if nonpos:
            issues.append(self._flag("candles with non-positive prices", nonpos))
        if ohlc:
            issues.append(self._flag("candles with inconsistent OHLC", ohlc))
        if order:
            issues.append(self._flag("candles with timestamps not increasing", order))

        if len(candles) > 2 and not order:
            diffs = [b.ts - a.ts for a, b in zip(candles, candles[1:])]
            typical = int(median(diffs))
            if typical > 0:
                gaps = [i for i, d in enumerate(diffs) if d > self.gap_factor * typical]
                small = [i for i, d in enumerate(diffs) if 1.5 * typical < d <= self.gap_factor * typical]
                if self.session_market:
                    closures = [i for i in gaps if diffs[i] > self.closure_s]
                    if closures:
                        warnings.append(f"{len(closures)} market closures (nights, weekends, holidays) of over "
                                        f"{self.closure_s // 3600}h, treated as normal for this market")
                    short = [i for i in range(len(diffs)) if 1.5 * typical < diffs[i] <= self.closure_s]
                else:
                    if gaps:
                        issues.append(self._flag(f"gaps over {self.gap_factor:g}x the typical interval", gaps))
                    short = small
                if short:
                    missing = sum(int(diffs[i] // typical) - 1 for i in short)
                    text = self._flag("places with missing candles", short) + f", {missing} candles missing"
                    if missing / len(candles) > self.max_missing:
                        issues.append(text)
                    else:
                        warnings.append(text)
                if any((c.ts - candles[0].ts) % typical for c in candles):
                    issues.append("timestamps are off the regular grid (timezone or feed shift?)")

        if len(candles) > 5 and not nonpos and not nonfinite:
            closes = [c.close for c in candles]
            rets = [math.log(b / a) for a, b in zip(closes, closes[1:])]
            med = median(rets)
            mad = median(abs(r - med) for r in rets)
            if mad > 0:
                sigma = 1.4826 * mad
                bad, big = [], []
                for i, r in enumerate(rets):
                    if abs(r - med) > self.spike_k * sigma:
                        nxt = rets[i + 1] if i + 1 < len(rets) else 0.0
                        # a bad tick is a V: the next move takes back nearly all of it. News spikes rarely do.
                        v_shape = nxt * r < 0 and abs(nxt + r) <= 0.25 * abs(r)
                        excused = self._near_news(candles[i + 1].ts)
                        (bad if v_shape and not excused else big).append(i + 1)
                if bad:
                    issues.append(self._flag("possible bad ticks (a spike that fully reverses)", bad))
                if big:
                    warnings.append(self._flag("very large moves", big))

        run = longest = 0
        for c in candles:
            run = run + 1 if c.high == c.low else 0
            longest = max(longest, run)
        if longest > self.flat_run:
            warnings.append(f"{longest} flat candles in a row, feed may be frozen")

        if reference:
            ref = {c.ts: c.close for c in reference}
            pairs = [(c.close, ref[c.ts]) for c in candles if c.ts in ref]
            if not pairs:
                issues.append("no overlapping timestamps with the reference feed (timezone shift?)")
            elif len(pairs) < 10:
                warnings.append("feed comparison skipped: fewer than 10 overlapping candles")
            else:
                worst = max(abs(a - b) / b for a, b in pairs)
                if worst > self.max_feed_diff:
                    issues.append(f"feeds disagree: worst close difference {worst:.2%}")

        return DataReport(not issues, tuple(issues), tuple(warnings))

    def check_schedule(self, candles: list, offset_hours: float, expected_minutes: tuple,
                       tolerance: int = 30, min_breaks: int = 10) -> tuple:
        """Warnings if the daily trading break is not where this market's break should be on the rule clock.

        If a file's times are in broker server time instead of UTC, every clock rule (and a news
        calendar given in UTC) would silently be hours off. The daily break is a fingerprint of the
        timezone. Returns a tuple of warnings, empty when things look right or there is too little to judge.
        """
        if len(candles) < 3 or not expected_minutes:
            return ()
        diffs = [b.ts - a.ts for a, b in zip(candles, candles[1:])]
        typical = int(median(diffs))
        if typical <= 0:
            return ()
        starts = [minute_of_day(a.ts + typical, offset_hours) for a, b in zip(candles, candles[1:])
                  if 3600 < b.ts - a.ts <= 8 * 3600]
        if len(starts) < min_breaks:
            return ()

        def near(m: int) -> int:
            return min(min(abs(m - e), 1440 - abs(m - e)) for e in expected_minutes)

        fits = sum(1 for m in starts if near(m) <= tolerance)
        if fits / len(starts) >= 0.8:
            return ()
        common = max(set(m // 15 * 15 for m in starts), key=lambda m: sum(1 for x in starts if x // 15 * 15 == m))
        shifts = [((common - e + 720) % 1440 - 720) for e in expected_minutes]
        primary = shifts[0]
        others = sorted(set(shifts[1:]) - {primary})
        fmt = lambda m: f"{m // 60:02d}:{m % 60:02d}"
        extra = (f" (or {', '.join(f'{o / 60:+.0f}' for o in others)} if the break you see is the other season's)"
                 if others else "")
        return (f"the daily break starts around {fmt(common)} on the rule clock, but this market's break should be near "
                + " or ".join(fmt(e) for e in expected_minutes)
                + f"; the file's times look shifted by about {primary / 60:+.0f} hours (broker server time instead of UTC?). "
                  f"If so, scan again with --data-utc-offset {primary / 60:+.0f}{extra}. "
                "Clock rules and the news calendar would be misaligned, so check the timestamps before trusting any result",)
