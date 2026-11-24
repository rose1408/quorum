import unittest

from quorum.__main__ import synthetic_candles
from quorum.agents import (AccountState, MarketObserver, PaperMonitor, PaperTrade, ResearchAnalyst, RiskManager,
                           Skeptic, StrategyAnalyst, TradeJournal, Validator)
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


class SkepticTests(unittest.TestCase):
    def test_rejects_thin_evidence(self):
        self.assertEqual(Skeptic().assess([2.0, -1.0, 2.0]).vote, Vote.REJECT)

    def test_rejects_luck(self):
        results = [2.0, -1.0] * 5 + [-1.0, -1.0, -1.0, 2.0] * 3  # barely positive, noisy
        v = Skeptic(max_luck=0.05).assess(results)
        self.assertEqual(v.vote, Vote.REJECT)

    def test_approves_solid_edge(self):
        results = [2.0] * 15 + [-1.0] * 5 + [2.0] * 10 + [-1.0] * 5
        self.assertEqual(Skeptic().assess(results).vote, Vote.APPROVE)

    def test_rejects_edge_that_flips(self):
        results = [2.0] * 12 + [-1.0] * 3 + [-1.0] * 12 + [2.0] * 3
        self.assertEqual(Skeptic().assess(results).vote, Vote.REJECT)


class ResearchTests(unittest.TestCase):
    def test_runs_every_combination_and_warns(self):
        coord = Coordinator(MarketObserver(), StrategyAnalyst(), Validator(), RiskManager(),
                            researcher=ResearchAnalyst(grid={"stop_atr": [1.5, 2.0], "reward_risk": [2.0]}))
        report = coord.research("X", synthetic_candles())
        self.assertEqual(len(report.results), 2)
        self.assertTrue(any("tested 2 combinations" in n for n in report.notes))

    def test_short_history(self):
        coord = Coordinator(MarketObserver(), StrategyAnalyst(), Validator(), RiskManager(),
                            researcher=ResearchAnalyst())
        report = coord.research("X", synthetic_candles(50))
        self.assertIsNone(report.best)


def paper(fill, exit_price, side=Side.LONG):
    return PaperTrade("X", side, 100.0, fill, 99.0 if side == Side.LONG else 101.0, exit_price, 1.0)


class MonitorTests(unittest.TestCase):
    def test_r_and_slippage(self):
        t = paper(100.1, 102.0)
        self.assertAlmostEqual(t.r_result(), 1.9)
        self.assertAlmostEqual(t.slippage_r(), 0.1)

    def test_drift_and_halt(self):
        m = PaperMonitor(expected_r=0.5, tolerance_r=0.3, min_trades=5)
        for _ in range(5):
            m.record(paper(100.0, 99.0))  # -1R each
        report = m.report()
        self.assertTrue(report.drift)
        self.assertTrue(m.halted())

    def test_no_halt_before_enough_trades(self):
        m = PaperMonitor(expected_r=0.5, min_trades=10)
        m.record(paper(100.0, 99.0))
        self.assertFalse(m.halted())

    def test_coordinator_stops_when_halted(self):
        m = PaperMonitor(expected_r=0.5, tolerance_r=0.3, min_trades=2)
        m.record(paper(100.0, 99.0))
        m.record(paper(100.0, 99.0))
        coord = Coordinator(MarketObserver(), StrategyAnalyst(), Validator(), RiskManager(), monitor=m)
        d = coord.decide("X", synthetic_candles(), AccountState(equity=1000.0))
        self.assertFalse(d.approved)
        self.assertIn("halted", d.reason)


class JournalTests(unittest.TestCase):
    def test_logs_only_approved_and_reviews(self):
        coord = Coordinator(MarketObserver(), StrategyAnalyst(), Validator(min_trades=0, min_expectancy_r=-99),
                            RiskManager(), journal=TradeJournal())
        d = coord.decide("X", synthetic_candles(), AccountState(equity=1000.0))
        self.assertTrue(d.approved)
        self.assertEqual(len(coord.journal.entries), 1)
        self.assertEqual(coord.review_journal(), ["no closed trades yet"])

    def test_review_flags_losing_bucket(self):
        j = TradeJournal()
        from quorum.types import Decision, Regime
        p = Proposal("X", Side.SHORT, 100.0, 101.0, 98.0, "t")
        for r in [-1, -1, 2, -1, -1, -1]:
            eid = j.log_decision(Decision(p, True, (), 1.0, "ok"), Regime.RANGING)
            j.close(eid, float(r))
        text = "\n".join(j.review(min_bucket=5))
        self.assertIn("losing here", text)
        self.assertIn("longest losing streak: 3", text)


class SkepticCoordinatorTests(unittest.TestCase):
    def test_skeptic_veto_blocks_trade(self):
        coord = Coordinator(MarketObserver(), StrategyAnalyst(), Validator(min_trades=0, min_expectancy_r=-99),
                            RiskManager(), skeptic=Skeptic(min_trades=10_000))
        d = coord.decide("X", synthetic_candles(), AccountState(equity=1000.0))
        self.assertFalse(d.approved)
        self.assertIn("skeptic", d.reason)


if __name__ == "__main__":
    unittest.main()
