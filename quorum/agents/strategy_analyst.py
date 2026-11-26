from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from ..clock import infer_timeframe
from ..types import MarketSnapshot, Proposal, Regime, Side


@dataclass(frozen=True)
class Context:
    """What a strategy may need to know about the market it trades besides its candles."""

    tz_offset_hours: float = 8.0   # the clock the rules are written in (8 = Philippine time)
    day_start_hour: float = 7.0    # when a trading day starts on that clock
    news: tuple = ()               # NewsEvent items, sorted by time


class Strategy:
    """Base for every strategy family. Turns market history into a trade idea, or nothing.

    Simple families override `propose` (they see a market snapshot). Families
    tied to a clock or a news calendar override `signal` (they see the candles
    and their timestamps). The rules for generic families are plain textbook
    setups so the research pipeline has something real to test.
    """

    name = "strategy"
    PARAMS: tuple = ()  # the numeric settings the lab may nudge and tune
    ARGS: tuple = ()    # every constructor argument, so an exact copy can be rebuilt
    skip_high_vol = False
    hold_bars = None    # optional time stop: close after this many candles
    context = Context()

    def bind(self, context: Context) -> "Strategy":
        self.context = context
        return self

    @property
    def params(self) -> dict:
        return {k: getattr(self, k) for k in self.ARGS}

    def variants(self, scale: float = 0.2) -> list:
        """Nearby settings (each tunable number nudged up and down) for the Validator's robustness check."""
        out = []
        for key in self.PARAMS:
            for factor in (1 - scale, 1 + scale):
                kwargs = self.params
                v = kwargs[key]
                kwargs[key] = max(1, int(round(v * factor))) if isinstance(v, int) else round(v * factor, 6)
                out.append(type(self)(**kwargs).bind(self.context))
        return out

    def compatible(self, timeframe_s: Optional[int]) -> Optional[str]:
        """A reason this strategy cannot be tested on this data, or None if it can."""
        return None

    def caveats(self) -> tuple:
        """Plain-language limits of how the rules were modeled."""
        return ()

    def _usable(self, s: MarketSnapshot) -> bool:
        return s.data_ok and s.atr > 0 and not (self.skip_high_vol and s.vol_state == "high")

    def propose(self, snapshot: MarketSnapshot) -> Optional[Proposal]:
        raise NotImplementedError

    def signal(self, symbol: str, history, observer) -> Optional[Proposal]:
        p = self.propose(observer.observe(symbol, history, check_data=False))
        if p is None:
            return None
        tf = infer_timeframe(history)
        now = history[-1].ts + tf if tf else None
        exit_ts = p.exit_ts
        if self.hold_bars and tf:
            exit_ts = now + int(round(self.hold_bars * tf))
        return replace(p, ts=now, exit_ts=exit_ts)


class TrendFollow(Strategy):
    """Trade with the trend (fast average vs slow average), ATR stop, fixed reward/risk."""

    name = "trend"
    PARAMS = ("stop_atr", "reward_risk")
    ARGS = ("stop_atr", "reward_risk", "skip_high_vol", "hold_bars")

    def __init__(self, stop_atr: float = 2.0, reward_risk: float = 2.0, skip_high_vol: bool = False,
                 hold_bars: Optional[int] = None):
        self.stop_atr = stop_atr
        self.reward_risk = reward_risk
        self.skip_high_vol = skip_high_vol
        self.hold_bars = hold_bars

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
    ARGS = ("stop_atr", "reward_risk", "skip_high_vol", "hold_bars")

    def __init__(self, stop_atr: float = 1.5, reward_risk: float = 2.0, skip_high_vol: bool = False,
                 hold_bars: Optional[int] = None):
        self.stop_atr = stop_atr
        self.reward_risk = reward_risk
        self.skip_high_vol = skip_high_vol
        self.hold_bars = hold_bars

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
    ARGS = ("entry_stretch", "stop_atr", "skip_high_vol", "hold_bars")

    def __init__(self, entry_stretch: float = 2.5, stop_atr: float = 1.5, skip_high_vol: bool = False,
                 hold_bars: Optional[int] = None):
        self.entry_stretch = entry_stretch
        self.stop_atr = stop_atr
        self.skip_high_vol = skip_high_vol
        self.hold_bars = hold_bars

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
