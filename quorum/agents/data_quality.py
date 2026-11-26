from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Optional


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
                 closure_s: int = 3600, max_missing: float = 0.01):
        # session markets (gold, indices) close every night and weekend, so long gaps are normal there
        self.session_market = session_market
        self.closure_s = closure_s
        self.max_missing = max_missing
        self.gap_factor = gap_factor
        self.spike_k = spike_k
        self.flat_run = flat_run
        self.max_feed_diff = max_feed_diff

    @staticmethod
    def _flag(label: str, idx: list) -> str:
        return f"{len(idx)} {label} (first at index {idx[0]})"

    def check(self, candles: list, reference: Optional[list] = None) -> DataReport:
        issues, warnings = [], []
        nonpos, ohlc, order = [], [], []
        for i, c in enumerate(candles):
            if min(c.open, c.high, c.low, c.close) <= 0:
                nonpos.append(i)
            if c.high < c.low or c.high < max(c.open, c.close) or c.low > min(c.open, c.close):
                ohlc.append(i)
            if i > 0 and c.ts <= candles[i - 1].ts:
                order.append(i)
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

        if len(candles) > 5 and not nonpos:
            closes = [c.close for c in candles]
            rets = [b / a - 1 for a, b in zip(closes, closes[1:])]
            med = median(rets)
            mad = median(abs(r - med) for r in rets)
            if mad > 0:
                sigma = 1.4826 * mad
                bad, big = [], []
                for i, r in enumerate(rets):
                    if abs(r - med) > self.spike_k * sigma:
                        nxt = rets[i + 1] if i + 1 < len(rets) else 0.0
                        reverses = nxt * r < 0 and abs(nxt) >= 0.5 * abs(r)
                        (bad if reverses else big).append(i + 1)
                if bad:
                    issues.append(self._flag("possible bad ticks (spike that reverses)", bad))
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
