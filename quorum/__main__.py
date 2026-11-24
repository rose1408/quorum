from __future__ import annotations

import random

from .agents import (AccountState, MarketObserver, PaperMonitor, ResearchAnalyst, RiskManager,
                     Skeptic, StrategyAnalyst, TradeJournal, Validator)
from .coordinator import Coordinator
from .types import Candle


def synthetic_candles(n: int = 600, seed: int = 7, drift: float = 0.0004) -> list:
    """Fake data so the demo runs. Replace with real Step Index candles."""
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


def build_coordinator() -> Coordinator:
    return Coordinator(
        MarketObserver(), StrategyAnalyst(), Validator(), RiskManager(),
        skeptic=Skeptic(), monitor=PaperMonitor(), journal=TradeJournal(), researcher=ResearchAnalyst(),
    )


def main() -> None:
    coordinator = build_coordinator()
    candles = synthetic_candles()

    print("== research ==")
    report = coordinator.research("DEMO", candles)
    for r in report.results[:3]:
        print(f"  {r.params}: train {r.train_expectancy:.2f}R ({r.train_trades}), "
              f"valid {r.valid_expectancy:.2f}R ({r.valid_trades}){'  OVERFIT' if r.overfit else ''}")
    print("  best:", report.best.params if report.best else "none")
    for note in report.notes:
        print("  note:", note)

    print("== decision ==")
    decision = coordinator.decide("DEMO", candles, AccountState(equity=1000.0))
    print("approved:", decision.approved)
    print("reason:  ", decision.reason)
    if decision.proposal:
        print("proposal:", decision.proposal)
    for v in decision.verdicts:
        print(f"  {v.agent}: {v.vote.value} - {v.reason}")
    if decision.approved:
        print("size:    ", round(decision.size, 4))


if __name__ == "__main__":
    main()
