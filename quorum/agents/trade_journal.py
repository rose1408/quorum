from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Optional

from ..types import Decision, Regime


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


class TradeJournal:
    """Logs approved trades and, once they close, looks for patterns."""

    name = "trade_journal"

    def __init__(self):
        self.entries: list = []

    def log_decision(self, decision: Decision, regime: Regime, ts: Optional[int] = None) -> Optional[int]:
        if not decision.approved or decision.proposal is None:
            return None
        entry = JournalEntry(len(self.entries) + 1, ts if ts is not None else int(time.time()),
                             decision.proposal.symbol, decision.proposal.side.value,
                             regime.value, decision.size, decision.reason)
        self.entries.append(entry)
        return entry.id

    def close(self, entry_id: int, r_result: float, note: str = "") -> None:
        for e in self.entries:
            if e.id == entry_id:
                e.r_result, e.note = r_result, note
                return
        raise KeyError(f"no journal entry {entry_id}")

    def review(self, min_bucket: int = 5) -> list:
        closed = [e for e in self.entries if e.r_result is not None]
        if not closed:
            return ["no closed trades yet"]
        out = [f"{len(closed)} closed trades, average {sum(e.r_result for e in closed) / len(closed):.2f}R"]

        for label, key in (("side", lambda e: e.side), ("regime", lambda e: e.regime)):
            buckets: dict = {}
            for e in closed:
                buckets.setdefault(key(e), []).append(e.r_result)
            for name, rs in sorted(buckets.items()):
                if len(rs) >= min_bucket:
                    avg = sum(rs) / len(rs)
                    flag = "  <- losing here" if avg < 0 else ""
                    out.append(f"{label} {name}: {len(rs)} trades, {avg:.2f}R{flag}")

        after_loss, after_win = [], []
        for prev, cur in zip(closed, closed[1:]):
            (after_loss if prev.r_result <= 0 else after_win).append(cur.r_result)
        if len(after_loss) >= min_bucket and len(after_win) >= min_bucket:
            out.append(f"after a loss: {sum(after_loss) / len(after_loss):.2f}R, "
                       f"after a win: {sum(after_win) / len(after_win):.2f}R")

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
