"""System tests for Quorum itself.

These check the team as a whole against known scenarios: does it contradict
itself, give different answers to the same input, let a veto slip through, or
report numbers that don't match a recalculation? Run with `python -m quorum.evals`.
When an LLM layer is added, these are the checks it must keep passing.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from .__main__ import DEMO_COSTS, build_coordinator, synthetic_candles
from .agents import (AccountState, MarketObserver, PaperMonitor, PaperTrade, RiskLimits, RiskManager,
                     StrategyAnalyst, Validator)
from .backtest import NO_COSTS, simulate
from .coordinator import Coordinator
from .types import Candle, Side, Vote


@dataclass(frozen=True)
class EvalResult:
    name: str
    passed: bool
    detail: str


def _permissive() -> Coordinator:
    """A team whose Validator waves things through, so approvals actually happen."""
    return Coordinator(MarketObserver(), StrategyAnalyst(),
                       Validator(min_trades=0, min_expectancy_r=-99, min_positive_windows=0.0, min_robust=0.0),
                       RiskManager(), costs=DEMO_COSTS)


def eval_deterministic() -> EvalResult:
    candles = synthetic_candles()
    a = build_coordinator().decide("E", candles, AccountState(equity=1000.0))
    b = build_coordinator().decide("E", candles, AccountState(equity=1000.0))
    return EvalResult("same input gives the same decision", a == b, "identical" if a == b else "decisions differ")


def eval_bad_data_blocks() -> EvalResult:
    candles = synthetic_candles(200)
    c = candles[100]
    candles[100] = Candle(c.ts, c.open, c.low - 1, c.high, c.close)  # high below low
    d = build_coordinator().decide("E", candles, AccountState(equity=1000.0))
    ok = not d.approved and d.reason.startswith("data check failed") and d.size == 0.0
    return EvalResult("bad data blocks any trade", ok, d.reason)


def eval_kill_switch_blocks() -> EvalResult:
    bad = []
    for seed in range(6):
        d = _permissive().decide("E", synthetic_candles(seed=seed), AccountState(equity=1000.0, kill_switch=True))
        if d.approved or d.size != 0.0:
            bad.append(seed)
    return EvalResult("kill switch always blocks", not bad, "held for 6 datasets" if not bad else f"leaked on seeds {bad}")


def eval_veto_dominance() -> EvalResult:
    bad, approved = [], 0
    for seed in range(6):
        for coord in (build_coordinator(), _permissive()):
            d = coord.decide("E", synthetic_candles(seed=seed), AccountState(equity=1000.0))
            all_yes = all(v.vote == Vote.APPROVE for v in d.verdicts)
            if d.approved != (bool(d.verdicts) and all_yes) or (not d.approved and d.size != 0.0):
                bad.append(seed)
            approved += d.approved
    return EvalResult("approval needs every vote, vetoed means size 0", not bad,
                      f"checked 12 decisions ({approved} approved)" if not bad else f"inconsistent on seeds {bad}")


def eval_size_within_risk() -> EvalResult:
    limits, checked, bad = RiskLimits(), 0, []
    for seed in range(8):
        acct = AccountState(equity=1000.0 + 250 * seed)
        d = _permissive().decide("E", synthetic_candles(seed=seed), acct)
        if d.approved:
            checked += 1
            risk = abs(d.proposal.entry - d.proposal.stop)
            if d.size * risk > acct.equity * limits.risk_per_trade * (1 + 1e-9):
                bad.append(seed)
    ok = checked > 0 and not bad
    return EvalResult("position size never exceeds the risk limit", ok,
                      f"checked {checked} approved trades" if checked else "no approvals to check")


def eval_costs_reduce_results() -> EvalResult:
    candles = synthetic_candles()
    obs, analyst = MarketObserver(), StrategyAnalyst()
    signal = lambda h: analyst.propose(obs.observe("E", h, check_data=False))
    free = simulate(candles, signal, 240, None, NO_COSTS)
    costly = simulate(candles, signal, 240, None, DEMO_COSTS)
    ok = len(free) == len(costly) and len(free) > 0 and sum(costly) < sum(free)
    return EvalResult("trading costs reduce results", ok, f"{sum(free):.2f}R free vs {sum(costly):.2f}R with costs")


def eval_reported_numbers_match() -> EvalResult:
    candles = synthetic_candles()
    obs, analyst = MarketObserver(), StrategyAnalyst()
    signal = lambda h: analyst.propose(obs.observe("E", h, check_data=False))
    validator = Validator(min_trades=0, min_expectancy_r=-99, min_positive_windows=0.0, min_robust=0.0)
    proposal = analyst.propose(obs.observe("E", candles, check_data=False))
    verdict = validator.review(proposal, candles, signal, costs=DEMO_COSTS)
    start = max(validator.warmup, int(len(candles) * (1 - validator.holdout)))
    expected = len(simulate(candles, signal, start, None, DEMO_COSTS))
    said = re.search(r"(\d+) trades", verdict.reason)
    ok = bool(said) and int(said.group(1)) == expected and verdict.metrics["trades"] == expected
    return EvalResult("reported numbers match a recalculation", ok, f"said {said.group(1) if said else '?'}, recomputed {expected}")


def eval_monitor_halts() -> EvalResult:
    m = PaperMonitor(expected_r=0.5, tolerance_r=0.3, min_trades=3)
    for _ in range(3):
        m.record(PaperTrade("E", Side.LONG, 100.0, 100.0, 99.0, 99.0, 1.0))
    d = Coordinator(MarketObserver(), StrategyAnalyst(), Validator(), RiskManager(), monitor=m).decide(
        "E", synthetic_candles(), AccountState(equity=1000.0))
    return EvalResult("a drifting edge halts trading", not d.approved and "halted" in d.reason, d.reason)


ALL_EVALS = [eval_deterministic, eval_bad_data_blocks, eval_kill_switch_blocks, eval_veto_dominance,
             eval_size_within_risk, eval_costs_reduce_results, eval_reported_numbers_match, eval_monitor_halts]


def run_evals() -> list:
    return [fn() for fn in ALL_EVALS]


def main() -> int:
    results = run_evals()
    for r in results:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}")
    failed = sum(1 for r in results if not r.passed)
    print(f"{len(results) - failed}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
