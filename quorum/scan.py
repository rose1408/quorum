from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Optional

from .agents import (AccountState, Context, DataQualityAgent, FlowAnalyst, FlowReport, LabReport, MarketFlow,
                     MarketObserver, PortfolioPlan, PortfolioRiskManager, StrategyLab, make_strategy)
from .backtest import drift_adjusted_r, expectancy, simulate_detail
from .clock import infer_timeframe
from .config import QuorumConfig, build_coordinator
from .profiles import MarketProfile
from .types import Decision, Vote


@dataclass(frozen=True)
class ReferenceResult:
    """A rule set scored exactly as given, over the whole history and on the holdout exam."""

    family: str
    params: dict
    trades: int
    win_rate: float
    expectancy_r: float
    drift_adjusted_r: float       # per trade, after removing what just being in the market earned
    profit_factor: Optional[float]
    max_drawdown_r: float
    total_r: float
    equity_return_pct: float      # sum of R times the risk per trade, not compounded
    buy_hold_pct: float           # simply holding over the same period
    exam: tuple
    passed: bool
    caveats: tuple
    avg_trade_pct: float = 0.0    # average result per trade as a share of the entry price, after costs


@dataclass(frozen=True)
class SymbolAnalysis:
    symbol: str
    flow: Optional[FlowReport]
    lab: Optional[LabReport]
    strategy: Optional[str]   # family the lab chose
    params: Optional[dict]
    decision: Decision
    warnings: tuple
    exam: tuple = ()          # holdout verdicts for the chosen strategy, setup or not
    profile: Optional[str] = None
    tz_offset_hours: float = 8.0
    caveats: tuple = ()
    cost_note: str = ""
    references: tuple = ()    # ReferenceResult for each rule set the profile asks to be scored as given
    search_structure: Optional[str] = None  # structure on the older data only, the part the search is allowed to see
    schedule_warnings: tuple = ()  # the daily break is not where it should be, so the file's timezone is probably wrong

    @property
    def exam_passed(self) -> bool:
        return bool(self.exam) and all(v.vote == Vote.APPROVE for v in self.exam)


@dataclass(frozen=True)
class ScanResult:
    analyses: tuple       # one per market, in the order given
    flow: MarketFlow
    plan: PortfolioPlan


def score_reference(symbol: str, candles: list, family: str, params: dict, cfg: QuorumConfig,
                    context: Context, quality: DataQualityAgent, luck: float, warmup: int = 40) -> ReferenceResult:
    strategy = make_strategy(family, params, context)
    team = build_coordinator(cfg, strategy, with_state=False, max_luck=luck, quality=quality)
    detail = simulate_detail(candles, team._signal_fn(symbol, strategy), warmup, None, cfg.costs)
    rs = [t.r for t in detail]
    wins, losses = sum(r for r in rs if r > 0), -sum(r for r in rs if r <= 0)
    equity = peak = dd = 0.0
    for r in rs:
        equity += r
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
    exam = team.exam(symbol, candles)
    return ReferenceResult(
        family, dict(strategy.params), len(rs), sum(1 for r in rs if r > 0) / len(rs) if rs else 0.0,
        expectancy(rs), expectancy(drift_adjusted_r(detail, candles, warmup)),
        wins / losses if losses > 0 else None, dd, sum(rs), sum(rs) * cfg.risk.risk_per_trade * 100,
        (candles[-1].close / candles[warmup].close - 1) * 100 if len(candles) > warmup else 0.0,
        exam, bool(exam) and all(v.vote == Vote.APPROVE for v in exam), strategy.caveats(),
        sum(t.r * t.risk / t.entry_price for t in detail if t.entry_price) / len(detail) * 100 if detail else 0.0)


def analyze_symbol(symbol: str, candles: list, account: AccountState, config: QuorumConfig,
                   reference: Optional[list] = None, profile: Optional[MarketProfile] = None,
                   news: tuple = ()) -> SymbolAnalysis:
    """The full pipeline for one market: data check, flow, strategy search, then the team's decision."""
    if profile is None:
        profile = MarketProfile("generic", "no market profile", infer_timeframe(candles) or 300,
                                session_market=False, costs=config.costs, min_trades=config.min_trades)
    costs = profile.costs
    news_times = tuple(e.ts for e in news if e.impact >= 2)
    if profile.news_multiplier > 1 and news:
        costs = costs.with_news(news_times, profile.news_multiplier, profile.news_window_s)
    risk = replace(config.risk, min_size=profile.min_size, size_step=profile.size_step)
    cfg = replace(config, risk=risk, costs=costs, min_trades=profile.min_trades, grids=profile.grids or config.grids)
    context = Context(profile.tz_offset_hours, profile.day_start_hour, tuple(sorted(news, key=lambda e: e.ts)))
    quality = DataQualityAgent(session_market=profile.session_market,
                               news_times=news_times if profile.news_multiplier > 1 else (),
                               news_window_s=profile.news_window_s)
    meta = dict(profile=profile.name, tz_offset_hours=profile.tz_offset_hours, cost_note=profile.cost_note)

    data = quality.check(candles, reference)
    warnings = tuple(data.warnings)
    if data.ok and profile.expected_break_minutes:
        # a wrong timezone would silently misalign every clock rule, so it is reported on its own, first
        meta["schedule_warnings"] = quality.check_schedule(candles, profile.tz_offset_hours,
                                                           profile.expected_break_minutes)
    if not data.ok:
        d = Decision(None, False, (), 0.0, "data check failed: " + "; ".join(data.issues), warnings)
        return SymbolAnalysis(symbol, None, None, None, None, d, warnings, **meta)

    if len(candles) < 300:
        d = Decision(None, False, (), 0.0, f"only {len(candles)} candles, far too few to analyze; "
                                           "a few thousand are needed for anything trustworthy", warnings)
        return SymbolAnalysis(symbol, None, None, None, None, d, warnings, **meta)

    observer = MarketObserver(quality=quality)
    flow = FlowAnalyst().analyze(symbol, candles, observer)   # shown in the report
    # the decisions below must not peek at the holdout, so the structure that drives them uses older data only
    search_end = int(len(candles) * (1 - cfg.holdout))
    search_flow = FlowAnalyst().analyze(symbol, candles[:search_end], observer) if search_end > 100 else flow
    meta["search_structure"] = search_flow.structure
    luck = cfg.random_max_luck if search_flow.structure == "random" else cfg.max_luck
    # the rule sets you gave are always scored exactly as given, so anything the search finds has something to beat
    references = tuple(score_reference(symbol, candles, fam, params, cfg, context, quality, luck)
                       for fam, params in profile.references
                       if make_strategy(fam, params, context).compatible(profile.timeframe_s) is None)
    meta["references"] = references
    lab = StrategyLab(grids=cfg.grids, holdout=cfg.holdout, min_trades=profile.lab_min_trades).build(
        symbol, candles, observer, cfg.costs, search_flow.structure, context, profile.timeframe_s)
    if lab.chosen is None:
        if lab.tested == 0:
            why = "not enough history for the strategy search to run"
        elif flow.structure == "unknown":
            why = "no strategy held up in research, and the market data could not be read (" + "; ".join(flow.notes) + ")"
        else:
            why = "no strategy held up in research"
        d = Decision(None, False, (), 0.0, why, warnings)
        return SymbolAnalysis(symbol, flow, lab, None, None, d, warnings, **meta)

    strategy = make_strategy(lab.chosen.family, lab.chosen.result.params, context)
    # a market that looks like a random walk has almost no chance of a real edge, so demand much stronger proof
    team = build_coordinator(cfg, strategy, with_state=False, max_luck=luck, quality=quality)
    exam = team.exam(symbol, candles)
    decision = team.decide(symbol, candles, account, reference)
    return SymbolAnalysis(symbol, flow, lab, lab.chosen.family, dict(lab.chosen.result.params), decision,
                          warnings, exam, caveats=strategy.caveats(), **meta)


def _run_one(args) -> SymbolAnalysis:
    return analyze_symbol(*args)


def scan_markets(markets: dict, account: AccountState, config: Optional[QuorumConfig] = None,
                 workers: int = 1, references: Optional[dict] = None, profiles: Optional[dict] = None,
                 news: tuple = (), progress=None) -> ScanResult:
    """Analyze several markets at once, then look across them.

    `profiles` maps a market name to its MarketProfile (timeframe, clock, costs, which strategies to
    search). `news` is a calendar of release times. With `workers` above 1 each market runs in its own
    process. Results do not depend on the worker count. `progress` is an optional function that
    receives short status messages, useful because big files take minutes.
    """
    config = config or QuorumConfig()
    references, profiles = references or {}, profiles or {}
    say = progress or (lambda message: None)
    started = time.time()
    jobs = [(s, c, account, config, references.get(s), profiles.get(s), tuple(news)) for s, c in markets.items()]
    for s, c in markets.items():
        say(f"{s}: {len(c):,} candles")
    analyses = None
    if workers > 1 and len(jobs) > 1:
        try:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
                futures = {pool.submit(_run_one, j): k for k, j in enumerate(jobs)}
                done = [None] * len(jobs)
                for fut in as_completed(futures):
                    k = futures[fut]
                    done[k] = fut.result()
                    say(f"{jobs[k][0]}: finished after {time.time() - started:.0f}s")
                analyses = done
        except Exception:
            analyses = None  # cannot run in separate processes here (or a worker failed): redo one at a time, which shows any real error
    if analyses is None:
        analyses = []
        for j in jobs:
            say(f"{j[0]}: working...")
            analyses.append(_run_one(j))
            say(f"{j[0]}: finished after {time.time() - started:.0f}s")

    reports = {a.symbol: a.flow for a in analyses if a.flow is not None}
    flow = FlowAnalyst().compare(reports, {s: markets[s] for s in reports})
    plan = PortfolioRiskManager(config.portfolio).allocate(analyses, flow, account)
    return ScanResult(tuple(analyses), flow, plan)
