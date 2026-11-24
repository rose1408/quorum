from __future__ import annotations

import calendar
import csv
import math
import os
import random
import time
from typing import Optional

from .types import Candle

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


def _candles_from_closes(closes: list, rng: random.Random, start_price: float = 100.0, step: int = 60) -> list:
    out, prev = [], start_price
    for i, c in enumerate(closes):
        h = max(prev, c) * (1 + abs(rng.gauss(0, 0.001)))
        l = min(prev, c) * (1 - abs(rng.gauss(0, 0.001)))
        out.append(Candle(step * i, prev, h, l, c))
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
                     rho: float = 0.0) -> list:
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
    return _candles_from_closes(closes, rng)


def synthetic_markets(n: int = 1500, seed: int = 7, rho: float = 0.4) -> dict:
    """Three fake pairs with different personalities that share some common movement."""
    rng = random.Random(seed)
    shocks = [rng.gauss(0, 1) for _ in range(n)]
    return {
        "TREND-A": synthetic_market("trending", n, seed + 1, shocks, rho),
        "REVERT-B": synthetic_market("mean_reverting", n, seed + 2, shocks, rho),
        "RANDOM-C": synthetic_market("random", n, seed + 3, shocks, rho),
    }
