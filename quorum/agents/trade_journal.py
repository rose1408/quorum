from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Optional, Union

from ..types import Decision, Regime, Side


@dataclass
class JournalEntry:
    id: int
    ts: int
    symbol: str
    side: str
    regime: str
    size: float
    reason: str
    r_result: Optional[float] = None
    note: str = ""
    source: str = "quorum"       # "quorum" for approved ideas, "manual" for your own trades
    moved_stop: bool = False
    closed_ts: Optional[int] = None


class TradeJournal:
    """Logs trades and, once they close, runs the post-mortem.

    Besides Quorum's own approved trades, you can log manual trades. That lets
    the review spot behavior patterns: ignoring the system, moving stops,
    overtrading, and trading again right after a loss.
    """

    name = "trade_journal"

    def __init__(self):
        self.entries: list = []

    def _add(self, **kw) -> JournalEntry:
        entry = JournalEntry(id=len(self.entries) + 1, **kw)
        self.entries.append(entry)
        return entry

    def log_decision(self, decision: Decision, regime: Regime, ts: Optional[int] = None) -> Optional[int]:
        if not decision.approved or decision.proposal is None:
            return None
        e = self._add(ts=ts if ts is not None else int(time.time()), symbol=decision.proposal.symbol,
                      side=decision.proposal.side.value, regime=regime.value,
                      size=decision.size, reason=decision.reason)
        return e.id

    def log_manual(self, symbol: str, side: Union[Side, str], regime: Union[Regime, str], size: float,
                   note: str = "", ts: Optional[int] = None, moved_stop: bool = False) -> int:
        e = self._add(ts=ts if ts is not None else int(time.time()), symbol=symbol,
                      side=getattr(side, "value", side), regime=getattr(regime, "value", regime),
                      size=size, reason="manual trade", note=note, source="manual", moved_stop=moved_stop)
        return e.id

    def close(self, entry_id: int, r_result: float, note: str = "", ts: Optional[int] = None,
              moved_stop: Optional[bool] = None) -> None:
        for e in self.entries:
            if e.id == entry_id:
                e.r_result = r_result
                e.closed_ts = ts if ts is not None else int(time.time())
                if note:
                    e.note = note
                if moved_stop is not None:
                    e.moved_stop = moved_stop
                return
        raise KeyError(f"no journal entry {entry_id}")

    @staticmethod
    def _avg(entries: list) -> float:
        return sum(e.r_result for e in entries) / len(entries)

    def review(self, min_bucket: int = 5, max_per_day: int = 5, revenge_window: int = 900) -> list:
        closed = [e for e in self.entries if e.r_result is not None]
        if not closed:
            return ["no closed trades yet"]
        out = [f"{len(closed)} closed trades, average {self._avg(closed):.2f}R"]

        for label, key in (("side", lambda e: e.side), ("regime", lambda e: e.regime)):
            buckets: dict = {}
            for e in closed:
                buckets.setdefault(key(e), []).append(e)
            for name, es in sorted(buckets.items()):
                if len(es) >= min_bucket:
                    avg = self._avg(es)
                    out.append(f"{label} {name}: {len(es)} trades, {avg:.2f}R" + ("  <- losing here" if avg < 0 else ""))

        after_loss, after_win = [], []
        for prev, cur in zip(closed, closed[1:]):
            (after_loss if prev.r_result <= 0 else after_win).append(cur)
        if len(after_loss) >= min_bucket and len(after_win) >= min_bucket:
            out.append(f"after a loss: {self._avg(after_loss):.2f}R, after a win: {self._avg(after_win):.2f}R")

        followed = [e for e in closed if e.source == "quorum"]
        manual = [e for e in closed if e.source == "manual"]
        if len(followed) >= min_bucket and len(manual) >= min_bucket:
            out.append(f"followed Quorum: {len(followed)} trades, {self._avg(followed):.2f}R; "
                       f"manual: {len(manual)} trades, {self._avg(manual):.2f}R")

        moved = [e for e in closed if e.moved_stop]
        kept = [e for e in closed if not e.moved_stop]
        if len(moved) >= 3 and kept:
            out.append(f"moved the stop on {len(moved)} trades: {self._avg(moved):.2f}R "
                       f"vs {self._avg(kept):.2f}R when kept")

        per_day = Counter(e.ts // 86400 for e in self.entries)
        day, count = per_day.most_common(1)[0]
        if count > max_per_day:
            out.append(f"overtrading: {count} trades on one day, limit is {max_per_day}")

        by_close = sorted((e for e in closed if e.closed_ts is not None), key=lambda e: e.closed_ts)
        revenge = []
        for e in closed:
            if e.source != "manual":
                continue
            prior = [p for p in by_close if p is not e and p.closed_ts <= e.ts]
            if prior and prior[-1].r_result <= 0 and e.ts - prior[-1].closed_ts <= revenge_window:
                revenge.append(e)
        if len(revenge) >= 3:
            out.append(f"{len(revenge)} manual trades opened within {revenge_window // 60} minutes "
                       f"of a loss, average {self._avg(revenge):.2f}R (revenge trading?)")

        streak = worst = 0
        for e in closed:
            streak = streak + 1 if e.r_result <= 0 else 0
            worst = max(worst, streak)
        out.append(f"longest losing streak: {worst}")
        return out

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in self.entries], f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TradeJournal":
        journal = cls()
        with open(path, encoding="utf-8") as f:
            journal.entries = [JournalEntry(**row) for row in json.load(f)]
        return journal
