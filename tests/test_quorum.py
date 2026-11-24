import unittest

from quorum.__main__ import synthetic_candles
from quorum.agents import AccountState, MarketObserver, RiskManager, StrategyAnalyst, Validator
from quorum.coordinator import Coordinator
from quorum.types import Candle, Proposal, Side, Vote


def make_coordinator(**validator_kwargs):
    return Coordinator(MarketObserver(), StrategyAnalyst(), Validator(**validator_kwargs), RiskManager())


class ObserverTests(unittest.TestCase):
    def test_flags_bad_data(self):
        candles = synthetic_candles(100)
        candles[50] = Candle(candles[50].ts, 100, 99, 101, 100)  # high < low
        snap = MarketObserver().observe("X", candles)
        self.assertFalse(snap.data_ok)

    def test_flags_too_few_candles(self):
        snap = MarketObserver().observe("X", synthetic_candles(10))
        self.assertFalse(snap.data_ok)


class RiskTests(unittest.TestCase):
    def setUp(self):
        self.p = Proposal("X", Side.LONG, 100.0, 99.0, 102.0, "t")

    def test_sizes_by_risk(self):
        v = RiskManager().review(self.p, AccountState(equity=1000.0))
        self.assertEqual(v.vote, Vote.APPROVE)
        self.assertAlmostEqual(v.size, 10.0)  # 1% of 1000 / 1.0 stop distance

    def test_kill_switch_vetoes(self):
        v = RiskManager().review(self.p, AccountState(equity=1000.0, kill_switch=True))
        self.assertEqual(v.vote, Vote.REJECT)

    def test_daily_loss_limit(self):
        v = RiskManager().review(self.p, AccountState(equity=1000.0, daily_pnl=-40.0))
        self.assertEqual(v.vote, Vote.REJECT)


class CoordinatorTests(unittest.TestCase):
    def test_bad_data_means_no_trade(self):
        d = make_coordinator().decide("X", synthetic_candles(10), AccountState(equity=1000.0))
        self.assertFalse(d.approved)
        self.assertIsNone(d.proposal)

    def test_risk_veto_blocks_trade(self):
        c = make_coordinator(min_trades=0, min_expectancy_r=-99)
        d = c.decide("X", synthetic_candles(), AccountState(equity=1000.0, kill_switch=True))
        self.assertFalse(d.approved)
        self.assertEqual(d.size, 0.0)

    def test_validator_veto_blocks_trade(self):
        c = make_coordinator(min_trades=10_000)
        d = c.decide("X", synthetic_candles(), AccountState(equity=1000.0))
        self.assertFalse(d.approved)


if __name__ == "__main__":
    unittest.main()
