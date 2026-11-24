from __future__ import annotations

import json
import time
from dataclasses import asdict


class PaperLedger:
    """Append-only log of decisions. Nothing here places a real order."""

    def __init__(self, path: str = "paper_ledger.jsonl"):
        self.path = path

    def record(self, decision) -> None:
        row = asdict(decision)
        row["logged_at"] = int(time.time())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
