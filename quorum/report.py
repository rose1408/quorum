from __future__ import annotations

from .clock import format_local
from .scan import ScanResult


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def render_report(scan: ScanResult, equity: float) -> str:
    """One readable summary of a scan. This is the single voice the team speaks with."""
    out = ["# Quorum market scan", ""]
    names = [a.symbol for a in scan.analyses]
    out += ["Advisory only. Nothing here places an order.", ""]

    out += ["## Overview", "", "| Market | Structure | Flow | Strategy | Holdout exam | Setup now |", "|---|---|---|---|---|---|"]
    for a in scan.analyses:
        structure = a.flow.structure if a.flow else "n/a"
        bias = f"{a.flow.bias} ({a.flow.flow_score:+.2f})" if a.flow else "n/a"
        strat = a.strategy or "none"
        exam = "n/a" if not a.exam else "PASSED" if a.exam_passed else "failed"
        verdict = "TRADE IDEA" if a.decision.approved else "no"
        out.append(f"| {a.symbol} | {structure} | {bias} | {strat} | {exam} | {verdict} |")
    out.append("")

    out += ["## Across markets", ""]
    if scan.flow.ranking:
        out.append("Strongest to weakest flow: " + ", ".join(scan.flow.ranking))
    if scan.flow.correlation:
        out += ["", "| Pair | Correlation |", "|---|---|"]
        for (a, b), v in sorted(scan.flow.correlation.items()):
            out.append(f"| {a} / {b} | {v:+.2f} |")
    for note in scan.flow.notes:
        out.append(f"- {note}")
    out.append("")

    for a in scan.analyses:
        out += [f"## {a.symbol}", ""]
        for w in a.schedule_warnings:
            out.append(f"- **Check this first:** {w}")
        for w in a.warnings:
            out.append(f"- Data note: {w}")
        if a.profile and a.profile != "generic":
            out.append(f"- Profile: {a.profile}")
            if a.cost_note:
                out.append(f"- Costs: {a.cost_note}")
        if a.flow:
            f = a.flow
            out.append(f"- Structure: **{f.structure}** (lag-1 autocorrelation {f.autocorr:+.3f}, z {f.autocorr_z:+.1f}; "
                       f"variance ratio {f.variance_ratio:.2f}, z {f.variance_ratio_z:+.1f})")
            out.append(f"- Flow: {f.bias}, trend strength {f.trend_strength:+.2f}, momentum {f.momentum:+.2f}, "
                       f"volatility {f.vol_state}, sitting at {f.range_position:.0%} of its 50-bar range")
            for note in f.notes:
                out.append(f"- {note}")
            if a.search_structure and a.search_structure != f.structure:
                out.append(f"- On the older data the search was allowed to see, the structure reads {a.search_structure}")
        if a.references:
            out += ["", "Your rules as given, tested on this history:", "",
                    "| Rule | Trades | Win rate | Avg R | Avg trade | Avg R without market drift | Profit factor | "
                    "Worst drawdown | Return at risk per trade | Just holding | Holdout exam |",
                    "|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in a.references:
                out.append(f"| {r.family} | {r.trades} | {r.win_rate:.0%} | {r.expectancy_r:+.3f} | {r.avg_trade_pct:+.3f}% | {r.drift_adjusted_r:+.3f} | "
                           f"{_pct(r.profit_factor)} | {r.max_drawdown_r:.1f}R | {r.equity_return_pct:+.1f}% | "
                           f"{r.buy_hold_pct:+.1f}% | {'PASSED' if r.passed else 'failed'} |")
            for r in a.references:
                if not r.passed and r.exam:
                    out.append(f"  - {r.family}: " + "; ".join(f"{v.agent} {v.vote.value}, {v.reason}" for v in r.exam))
            out.append("")
            found = a.exam_passed and a.strategy not in {r.family for r in a.references}
            passed = [r.family for r in a.references if r.passed]
            if passed and not found:
                out.append(f"- Verdict: your rule(s) {', '.join(passed)} passed the holdout exam and nothing the search found did")
            elif passed and found:
                out.append(f"- Verdict: your rule(s) {', '.join(passed)} and the searched {a.strategy} strategy both passed; compare them above")
            elif found:
                out.append(f"- Verdict: the searched {a.strategy} strategy passed the holdout exam and your rules did not")
            else:
                out.append("- Verdict: neither your rules nor anything the search found passed the holdout exam; stand down on this market for now")
        if a.lab:
            out.append(f"- Strategy search: {a.lab.tested} combinations tested")
            if a.strategy:
                out.append(f"- Chosen: {a.strategy} {a.params}")
            for c in a.caveats:
                out.append(f"  - note: {c}")
            top = [c for c in a.lab.candidates if c.eligible][:3] or list(a.lab.candidates[:3])
            for c in top:
                r = c.result
                flag = "" if c.eligible else " (did not qualify)"
                out.append(f"  - {c.family} {r.params}: train {r.train_expectancy:+.2f}R ({r.train_trades} trades), "
                           f"validation {r.valid_expectancy:+.2f}R ({r.valid_trades}){flag}")
            for note in a.lab.notes:
                out.append(f"- {note}")
        if a.exam:
            out.append(f"- Holdout exam on the chosen strategy (data the search never saw): "
                       f"**{'passed' if a.exam_passed else 'failed'}**")
            for v in a.exam:
                out.append(f"  - {v.agent}: {v.vote.value}, {v.reason}")
                if v.metrics and v.agent == "validator":
                    m = v.metrics
                    out.append(f"    - profit factor {_pct(m.get('profit_factor'))}, max drawdown {m.get('max_drawdown_r', 0):.1f}R, "
                               f"after removing the market's own drift {m.get('drift_adjusted_expectancy_r', 0):+.2f}R per trade")
        d = a.decision
        out.append(f"- Live decision: **{'approved' if d.approved else 'not approved'}**, {d.reason}")
        if d.proposal:
            p = d.proposal
            tgt = f"target {p.target:.4f}" if p.target is not None else "no take-profit"
            ext = f", closes {format_local(p.exit_ts, a.tz_offset_hours)} (rule clock)" if p.exit_ts else ""
            out.append(f"- Idea: {p.side.value} at {p.entry:.4f}, stop {p.stop:.4f}, {tgt}{ext} ({p.rationale})")
        for v in d.verdicts:
            out.append(f"  - {v.agent}: {v.vote.value}, {v.reason}")
        out.append("")

    out += ["## Portfolio plan", ""]
    plan = scan.plan
    if plan.allocations:
        out += ["| Market | Side | Size | Risk of equity |", "|---|---|---|---|"]
        for al in plan.allocations:
            out.append(f"| {al.symbol} | {al.side} | {al.size:.4f} | {al.risk_fraction:.2%} |")
        out.append("")
        out.append(f"Total risk {plan.total_risk:.2%} of {equity:,.2f} equity.")
    for sym, why in plan.skipped:
        out.append(f"- Skipped {sym}: {why}")
    for note in plan.notes:
        out.append(f"- {note}")
    out += ["", "## Read this before acting", "",
            "- Many combinations were searched per market. Some will look good by luck. The holdout test is the real check.",
            "- Results assume the cost model you gave. Costs left at zero make everything look better than it is.",
            "- A past edge can stop working. Paper trade first and watch for drift."]
    return "\n".join(out) + "\n"
