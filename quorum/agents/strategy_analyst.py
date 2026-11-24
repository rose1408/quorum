from __future__ import annotations

from typing import Optional

from ..types import MarketSnapshot, Proposal, Regime, Side


class Strategy:
    """Base for the strategy families. Each turns a market snapshot into a trade idea, or nothing.

    The rules below are generic textbook setups so the research pipeline has
    something real to test. Your own strategy gets added here as another family.
    """

    name = "strategy"
    PARAMS: tuple = ()  # the numeric settings the lab may tune
    skip_high_vol = False

    @property
    def params(self) -> dict:
        out = {k: getattr(self, k) for k in self.PARAMS}
        out["skip_high_vol"] = self.skip_high_vol
        return out

    def variants(self, scale: float = 0.2) -> list:
        """Nearby settings (each number nudged up and down) for the Validator's robustness check."""
        out = []
        for key in self.PARAMS:
            for factor in (1 - scale, 1 + scale):
                kwargs = self.params
                kwargs[key] = round(kwargs[key] * factor, 6)
                out.append(type(self)(**kwargs))
        return out

    def _usable(self, s: MarketSnapshot) -> bool:
        return s.data_ok and s.atr > 0 and not (self.skip_high_vol and s.vol_state == "high")

    def propose(self, snapshot: MarketSnapshot) -> Optional[Proposal]:
        raise NotImplementedError


class TrendFollow(Strategy):
    """Trade with the trend (fast average vs slow average), ATR stop, fixed reward/risk."""

    name = "trend"
    PARAMS = ("stop_atr", "reward_risk")

    def __init__(self, stop_atr: float = 2.0, reward_risk: float = 2.0, skip_high_vol: bool = False):
        self.stop_atr = stop_atr
        self.reward_risk = reward_risk
        self.skip_high_vol = skip_high_vol

    def propose(self, s: MarketSnapshot) -> Optional[Proposal]:
        if not self._usable(s):
            return None
        entry, dist = s.last_close, s.atr * self.stop_atr
        if s.regime == Regime.TRENDING_UP:
            return Proposal(s.symbol, Side.LONG, entry, entry - dist, entry + dist * self.reward_risk,
                            "uptrend: fast MA above slow MA")
        if s.regime == Regime.TRENDING_DOWN:
            return Proposal(s.symbol, Side.SHORT, entry, entry + dist, entry - dist * self.reward_risk,
                            "downtrend: fast MA below slow MA")
        return None


class Breakout(Strategy):
    """Trade a close beyond the recent high or low, ATR stop, fixed reward/risk."""

    name = "breakout"
    PARAMS = ("stop_atr", "reward_risk")

    def __init__(self, stop_atr: float = 1.5, reward_risk: float = 2.0, skip_high_vol: bool = False):
        self.stop_atr = stop_atr
        self.reward_risk = reward_risk
        self.skip_high_vol = skip_high_vol

    def propose(self, s: MarketSnapshot) -> Optional[Proposal]:
        if not self._usable(s) or s.channel_high <= 0:
            return None
        entry, dist = s.last_close, s.atr * self.stop_atr
        if entry > s.channel_high:
            return Proposal(s.symbol, Side.LONG, entry, entry - dist, entry + dist * self.reward_risk,
                            "closed above the recent high")
        if entry < s.channel_low:
            return Proposal(s.symbol, Side.SHORT, entry, entry + dist, entry - dist * self.reward_risk,
                            "closed below the recent low")
        return None


class MeanReversion(Strategy):
    """Fade a stretch far from the slow average, aiming back at the average."""

    name = "mean_reversion"
    PARAMS = ("entry_stretch", "stop_atr")

    def __init__(self, entry_stretch: float = 2.5, stop_atr: float = 1.5, skip_high_vol: bool = False):
        self.entry_stretch = entry_stretch
        self.stop_atr = stop_atr
        self.skip_high_vol = skip_high_vol

    def propose(self, s: MarketSnapshot) -> Optional[Proposal]:
        if not self._usable(s):
            return None
        entry, dist = s.last_close, s.atr * self.stop_atr
        if s.stretch <= -self.entry_stretch and s.slow_ma > entry:
            return Proposal(s.symbol, Side.LONG, entry, entry - dist, s.slow_ma,
                            f"{-s.stretch:.1f} ATR below the average, fading back to it")
        if s.stretch >= self.entry_stretch and s.slow_ma < entry:
            return Proposal(s.symbol, Side.SHORT, entry, entry + dist, s.slow_ma,
                            f"{s.stretch:.1f} ATR above the average, fading back to it")
        return None


StrategyAnalyst = TrendFollow  # the original name, kept so older code keeps working

FAMILIES = {"trend": TrendFollow, "breakout": Breakout, "mean_reversion": MeanReversion}
