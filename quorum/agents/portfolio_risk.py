from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioLimits:
    max_total_risk: float = 0.02   # fraction of equity at risk across all positions
    max_positions: int = 2
    max_correlation: float = 0.7   # two trades that bet the same way on markets this linked count as one


@dataclass(frozen=True)
class Allocation:
    symbol: str
    side: str
    size: float
    risk_fraction: float  # share of equity at risk on this trade
    note: str


@dataclass(frozen=True)
class PortfolioPlan:
    allocations: tuple
    skipped: tuple        # (symbol, reason)
    total_risk: float
    notes: tuple


class PortfolioRiskManager:
    """Looks across all markets at once. Per-trade limits live in the Risk Manager; this adds the team-wide ones.

    From the ideas that already won approval, it picks the best by validated
    expectancy, one at a time, and skips any that would break the position
    limit, the total-risk cap, or double up the same bet on linked markets.
    """

    name = "portfolio_risk"

    def __init__(self, limits: PortfolioLimits = PortfolioLimits()):
        self.limits = limits

    def allocate(self, analyses, flow, account) -> PortfolioPlan:
        L = self.limits
        approved = [a for a in analyses if a.decision.approved and a.decision.proposal is not None]
        if not approved:
            return PortfolioPlan((), (), 0.0, ("no market produced an approved idea",))

        def expectancy(a) -> float:
            metrics = a.decision.verdicts[0].metrics or {}
            return metrics.get("expectancy_r", 0.0)

        approved.sort(key=expectancy, reverse=True)
        slots = max(0, L.max_positions - account.open_positions)
        chosen, skipped, total = [], [], 0.0
        for a in approved:
            d, p = a.decision, a.decision.proposal
            risk_frac = d.size * abs(p.entry - p.stop) / account.equity
            if len(chosen) >= slots:
                skipped.append((a.symbol, f"position limit of {L.max_positions} reached"))
                continue
            clash = None
            for other in chosen:
                same = 1 if p.side == other.decision.proposal.side else -1
                link = flow.corr(a.symbol, other.symbol) if flow is not None else 0.0
                if same * link >= L.max_correlation:
                    clash = (other.symbol, link)
                    break
            if clash:
                skipped.append((a.symbol, f"same bet as {clash[0]} (correlation {clash[1]:+.2f}), kept the stronger one"))
                continue
            if total + risk_frac > L.max_total_risk + 1e-12:
                skipped.append((a.symbol, f"would take total risk to {total + risk_frac:.1%}, cap is {L.max_total_risk:.1%}"))
                continue
            chosen.append(a)
            total += risk_frac

        allocations = tuple(
            Allocation(a.symbol, a.decision.proposal.side.value, a.decision.size,
                       a.decision.size * abs(a.decision.proposal.entry - a.decision.proposal.stop) / account.equity,
                       a.decision.reason)
            for a in chosen)
        return PortfolioPlan(allocations, tuple(skipped), total, ())
