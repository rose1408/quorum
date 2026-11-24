from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..types import Side


@dataclass(frozen=True)
class PaperTrade:
    symbol: str
    side: Side
    planned_entry: float
    fill_entry: float
    stop: float
    exit_price: float
    size: float

    def risk(self) -> float:
        d = self.planned_entry - self.stop if self.side == Side.LONG else self.stop - self.planned_entry
        return d

    def r_result(self) -> float:
        move = self.exit_price - self.fill_entry if self.side == Side.LONG else self.fill_entry - self.exit_price
        return move / self.risk()

    def slippage_r(self) -> float:
        """Positive means the fill was worse than planned."""
        worse = self.fill_entry - self.planned_entry if self.side == Side.LONG else self.planned_entry - self.fill_entry
        return worse / self.risk()


@dataclass(frozen=True)
class MonitorReport:
    trades: int
    avg_r: float
    avg_slippage_r: float
    drift: bool
    halt: bool
    notes: tuple


class PaperMonitor:
    """Compares paper fills to what the backtest assumed and flags drift."""

    name = "paper_monitor"

    def __init__(self, expected_r: Optional[float] = None, tolerance_r: float = 0.3,
                 min_trades: int = 10, max_slippage_r: float = 0.2):
        self.expected_r = expected_r
        self.tolerance_r = tolerance_r
        self.min_trades = min_trades
        self.max_slippage_r = max_slippage_r
        self.trades: list = []

    def record(self, trade: PaperTrade) -> None:
        if trade.risk() <= 0:
            raise ValueError("trade has no valid stop distance")
        self.trades.append(trade)

    def report(self) -> MonitorReport:
        n = len(self.trades)
        if n == 0:
            return MonitorReport(0, 0.0, 0.0, False, False, ("no paper trades yet",))
        avg_r = sum(t.r_result() for t in self.trades) / n
        avg_slip = sum(t.slippage_r() for t in self.trades) / n
        notes, drift, halt = [], False, False
        if avg_slip > self.max_slippage_r:
            notes.append(f"average slippage {avg_slip:.2f}R is above the {self.max_slippage_r:.2f}R assumed")
        if self.expected_r is not None and n >= self.min_trades:
            gap = self.expected_r - avg_r
            if gap > self.tolerance_r:
                drift = True
                notes.append(f"realized {avg_r:.2f}R vs expected {self.expected_r:.2f}R")
            if gap > 2 * self.tolerance_r:
                halt = True
        return MonitorReport(n, avg_r, avg_slip, drift, halt, tuple(notes))

    def halted(self) -> bool:
        return self.report().halt
