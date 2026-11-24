from __future__ import annotations

import random

from .agents import AccountState, MarketObserver, RiskManager, StrategyAnalyst, Validator
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


def main() -> None:
    coordinator = Coordinator(MarketObserver(), StrategyAnalyst(), Validator(), RiskManager())
    decision = coordinator.decide("DEMO", synthetic_candles(), AccountState(equity=1000.0))
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
