from __future__ import annotations

import calendar
import csv
import math
import os
import random
import time
from typing import Optional

from .clock import minute_of_day, weekday
from .types import Candle, NewsEvent

_ALIASES = {
    "time": ("time", "timestamp", "datetime", "date", "ts", "epoch"),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c"),
}
_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M",
            "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
            "%Y-%m-%d", "%Y.%m.%d")


def _parse_time(text: str) -> int:
    text = text.strip()
    try:
        value = float(text)
        return int(value / 1000) if value > 1e11 else int(value)  # milliseconds or seconds
    except ValueError:
        pass
    for fmt in _FORMATS:
        try:
            return calendar.timegm(time.strptime(text, fmt))  # read as UTC
        except ValueError:
            continue
    raise ValueError(f"cannot read time {text!r}")


def load_candles(path: str) -> list:
    """Read candles from a CSV (comma, tab, or semicolon separated). Handles MT5-style exports too.

    Needs columns for time, open, high, low, close. Date and time in separate
    columns (like <DATE> and <TIME>) are joined. Rows are kept in file order, so
    the Data Quality Agent can flag anything out of order.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        header = [h.strip().strip("<>").lower() for h in next(reader)]
        index = {}
        for key, names in _ALIASES.items():
            for name in names:
                if name in header and key not in index:
                    index[key] = header.index(name)
        missing = [k for k in _ALIASES if k not in index]
        if missing:
            raise ValueError(f"{os.path.basename(path)}: missing column(s) {', '.join(missing)}")
        time_col = index["time"]
        out = []
        for row in reader:
            if not row or all(not x.strip() for x in row):
                continue
            stamp = row[time_col]
            if "date" in header and "time" in header and header.index("date") != header.index("time"):
                stamp = row[header.index("date")] + " " + row[header.index("time")]
            out.append(Candle(_parse_time(stamp), float(row[index["open"]]), float(row[index["high"]]),
                              float(row[index["low"]]), float(row[index["close"]])))
        return out


def load_markets(paths: list) -> dict:
    """Map market name (the file name without extension) to its candles."""
    return {os.path.splitext(os.path.basename(p))[0]: load_candles(p) for p in paths}


def _candles_from_closes(closes: list, rng: random.Random, start_price: float = 100.0, step: int = 60,
                         start_ts: int = 0) -> list:
    out, prev = [], start_price
    for i, c in enumerate(closes):
        h = max(prev, c) * (1 + abs(rng.gauss(0, 0.001)))
        l = min(prev, c) * (1 - abs(rng.gauss(0, 0.001)))
        out.append(Candle(start_ts + step * i, prev, h, l, c))
        prev = c
    return out


def synthetic_candles(n: int = 600, seed: int = 7, drift: float = 0.0004) -> list:
    """Fake data so the demo runs. Replace with real candles."""
    rng = random.Random(seed)
    price, out = 100.0, []
    for i in range(n):
        o = price
        c = o * (1 + rng.gauss(drift, 0.003))
        h = max(o, c) * (1 + abs(rng.gauss(0, 0.001)))
        l = min(o, c) * (1 - abs(rng.gauss(0, 0.001)))
        out.append(Candle(60 * i, o, h, l, c))
        price = c
    return out


def synthetic_market(kind: str, n: int = 1500, seed: int = 1, shocks: Optional[list] = None,
                     rho: float = 0.0, tf_s: int = 60, start_ts: int = 0) -> list:
    """Fake market with known behavior: 'trending', 'mean_reverting', or 'random'.

    `shocks` is a shared noise series; `rho` is how much of each step comes from it,
    which makes several markets move together a bit.
    """
    rng = random.Random(seed)
    sigma = 0.003
    own = [rng.gauss(0, 1) for _ in range(n)]
    eps = [(math.sqrt(rho) * shocks[i] + math.sqrt(1 - rho) * own[i]) if shocks else own[i] for i in range(n)]
    closes = []
    if kind == "trending":        # each move tends to continue
        r, x = 0.0, math.log(100.0)
        for e in eps:
            r = 0.3 * r + sigma * e
            x += r
            closes.append(math.exp(x))
    elif kind == "mean_reverting":  # pulled back toward a fixed level
        x = 0.0
        for e in eps:
            x = x * 0.9 + sigma * 1.3 * e
            closes.append(100.0 * math.exp(x))
    elif kind == "random":        # no memory at all
        x = math.log(100.0)
        for e in eps:
            x += sigma * e
            closes.append(math.exp(x))
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return _candles_from_closes(closes, rng, step=tf_s, start_ts=start_ts)


def synthetic_markets(n: int = 1500, seed: int = 7, rho: float = 0.4) -> dict:
    """Three fake pairs with different personalities that share some common movement."""
    rng = random.Random(seed)
    shocks = [rng.gauss(0, 1) for _ in range(n)]
    return {
        "TREND-A": synthetic_market("trending", n, seed + 1, shocks, rho),
        "REVERT-B": synthetic_market("mean_reverting", n, seed + 2, shocks, rho),
        "RANDOM-C": synthetic_market("random", n, seed + 3, shocks, rho),
    }


def load_news(path: str, min_impact: int = 1) -> tuple:
    """Read a news calendar CSV: a time column, an optional event name, an optional impact.

    Impact can be low/medium/high, 1/2/3, or a count of stars or bulls. Times are read as UTC.
    """
    levels = {"low": 1, "medium": 2, "med": 2, "high": 3}
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        header = [h.strip().strip("<>").lower() for h in next(reader)]
        time_col = next((header.index(n) for n in _ALIASES["time"] if n in header), None)
        if time_col is None:
            raise ValueError(f"{os.path.basename(path)}: no time column")
        name_col = next((header.index(n) for n in ("event", "name", "title", "description") if n in header), None)
        impact_col = next((header.index(n) for n in ("impact", "importance", "volatility") if n in header), None)
        out = []
        for row in reader:
            if not row or not row[time_col].strip():
                continue
            impact = 3
            if impact_col is not None and row[impact_col].strip():
                raw = row[impact_col].strip().lower()
                impact = levels.get(raw) or (int(raw) if raw.isdigit() else raw.count("*") or raw.count("\u2605") or 3)
            if impact >= min_impact:
                out.append(NewsEvent(_parse_time(row[time_col]), row[name_col].strip() if name_col is not None else "", impact))
    return tuple(sorted(out, key=lambda e: e.ts))


def _session_open(ts: int) -> bool:
    """US30-style hours in UTC: open Sunday 23:00 to Friday 22:00, with a daily break 22:00 to 23:00."""
    dow, minute = weekday(ts, 0), minute_of_day(ts, 0)
    if dow == 5 or (dow == 6 and minute < 23 * 60) or (dow == 4 and minute >= 22 * 60):
        return False
    return not (dow <= 3 and 22 * 60 <= minute < 23 * 60)


def synthetic_session_market(years: float = 1.0, tf_s: int = 300, seed: int = 1, drift_per_year: float = 0.08,
                             vol_per_year: float = 0.15, start_price: float = 40000.0, start_ts: int = 1704067200) -> list:
    """Fake index with real-looking hours: nightly breaks, weekend closures, small gaps at reopening.

    Starts Monday 2024-01-01 00:00 UTC. Drift is the steady upward pull, so long-only rules look good in it.
    """
    rng = random.Random(seed)
    bars_per_year = 252 * (23 * 3600 / tf_s)
    sigma, mu = vol_per_year / math.sqrt(bars_per_year), drift_per_year / bars_per_year
    out, price, prev_open, ts = [], start_price, True, start_ts
    end = start_ts + int(years * 365 * 86400)
    while ts < end:
        is_open = _session_open(ts)
        if is_open:
            gap = rng.gauss(0, 3 * sigma) if not prev_open else 0.0
            o = price * (1 + gap)
            c = o * math.exp(rng.gauss(mu, sigma))
            out.append(Candle(ts, o, max(o, c) * (1 + abs(rng.gauss(0, sigma / 4))),
                              min(o, c) * (1 - abs(rng.gauss(0, sigma / 4))), c))
            price = c
        prev_open = is_open
        ts += tf_s
    return out


def synthetic_news_market(days: int = 20, tf_s: int = 60, seed: int = 1, follow_through: float = 0.0,
                          start_price: float = 2000.0, start_ts: int = 1704067200) -> tuple:
    """Fake gold-like market with scheduled news bursts. Returns (candles, news events).

    Each weekday has a release at 13:30 UTC. For a few candles after it volatility jumps, and
    `follow_through` (0 to 1) makes those moves keep going in the same direction. At 0 there is
    nothing to trade; above 0 there is a real post-news pattern to find.
    """
    rng = random.Random(seed)
    sigma = 0.0003
    out, events, price, prev_open, ts = [], [], start_price, True, start_ts
    end = start_ts + days * 86400
    burst_left, last_r = 0, 0.0
    while ts < end:
        is_open = _session_open(ts)
        if is_open:
            minute, dow = minute_of_day(ts, 0), weekday(ts, 0)
            if minute == 13 * 60 + 30 and dow < 5:
                events.append(NewsEvent(ts, "US data", 3))
                burst_left, last_r = 6, 0.0
            if burst_left:
                r = follow_through * last_r + rng.gauss(0, 5 * sigma) if burst_left < 6 else rng.gauss(0, 14 * sigma)
                burst_left -= 1
            else:
                r = rng.gauss(0, sigma)
            last_r = r
            o = price * (1 + (rng.gauss(0, 2 * sigma) if not prev_open else 0.0))
            c = o * math.exp(r)
            out.append(Candle(ts, o, max(o, c) * (1 + abs(rng.gauss(0, sigma / 3))),
                              min(o, c) * (1 - abs(rng.gauss(0, sigma / 3))), c))
            price = c
        prev_open = is_open
        ts += tf_s
    return out, tuple(events)
