import unittest

from quorum.__main__ import synthetic_candles
from quorum.agents import (AccountState, DataQualityAgent, MarketObserver, PaperMonitor, PaperTrade,
                           ResearchAnalyst, RiskLimits, RiskManager, Skeptic, StrategyAnalyst, TradeJournal,
                           Validator)
from quorum.backtest import NO_COSTS, CostModel, cost_r, simulate, walk
from quorum.coordinator import Coordinator
from quorum.evals import run_evals
from quorum.types import Candle, Decision, Proposal, Regime, Side, Vote

EASY = dict(min_trades=0, min_expectancy_r=-99, min_positive_windows=0.0, min_robust=0.0)


def make_coordinator(**validator_kwargs):
    return Coordinator(MarketObserver(), StrategyAnalyst(), Validator(**validator_kwargs), RiskManager())


class DataQualityTests(unittest.TestCase):
    def setUp(self):
        self.q = DataQualityAgent()

    def test_clean_data_passes(self):
        self.assertTrue(self.q.check(synthetic_candles(200)).ok)

    def test_flags_inconsistent_ohlc(self):
        candles = synthetic_candles(100)
        candles[50] = Candle(candles[50].ts, 100, 99, 101, 100)  # high < low
        self.assertFalse(self.q.check(candles).ok)

    def test_flags_gap(self):
        candles = synthetic_candles(200)
        del candles[50:61]
        report = self.q.check(candles)
        self.assertFalse(report.ok)
        self.assertTrue(any("gaps" in i for i in report.issues))

    def test_flags_misaligned_timestamp(self):
        candles = synthetic_candles(100)
        c = candles[40]
        candles[40] = Candle(c.ts + 7, c.open, c.high, c.low, c.close)
        self.assertTrue(any("grid" in i for i in self.q.check(candles).issues))

    def test_flags_bad_tick_that_reverses(self):
        candles = synthetic_candles(200)
        c = candles[100]
        candles[100] = Candle(c.ts, c.open, c.close * 1.25, c.low, c.close * 1.25)
        self.assertTrue(any("bad ticks" in i for i in self.q.check(candles).issues))

    def test_flags_feed_mismatch(self):
        candles = synthetic_candles(100)
        other = [Candle(c.ts, c.open, c.high, c.low, c.close * 1.01) for c in candles]
        self.assertTrue(any("feeds disagree" in i for i in self.q.check(candles, other).issues))
        self.assertTrue(self.q.check(candles, candles).ok)

    def test_flags_no_overlap(self):
        candles = synthetic_candles(100)
        shifted = [Candle(c.ts + 3600, c.open, c.high, c.low, c.close) for c in candles]
        self.assertFalse(self.q.check(candles, shifted).ok)

    def test_warns_on_frozen_feed(self):
        candles = synthetic_candles(100)
        for i in range(30, 40):
            p = candles[i].close
            candles[i] = Candle(candles[i].ts, p, p, p, p)
        report = self.q.check(candles)
        self.assertTrue(any("flat" in w for w in report.warnings))


class ObserverTests(unittest.TestCase):
    def test_flags_bad_data(self):
        candles = synthetic_candles(100)
        candles[50] = Candle(candles[50].ts, 100, 99, 101, 100)
        self.assertFalse(MarketObserver().observe("X", candles).data_ok)

    def test_flags_too_few_candles(self):
        self.assertFalse(MarketObserver().observe("X", synthetic_candles(10)).data_ok)

    def test_labels_trend_and_volatility(self):
        snap = MarketObserver().observe("X", synthetic_candles(300, drift=0.003))
        self.assertEqual(snap.regime, Regime.TRENDING_UP)
        self.assertIn(snap.vol_state, ("low", "normal", "high"))


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.p = Proposal("X", Side.LONG, 100.0, 99.0, 102.0, "t")
        self.candles = [Candle(0, 100, 100.5, 99.5, 100), Candle(60, 100, 102.5, 99.8, 102)]

    def test_costs_come_off_wins_and_losses(self):
        costs = CostModel(spread=0.1, slippage=0.05, fee=0.0)  # 0.2 round trip, 0.2R on a 1.0 stop
        self.assertAlmostEqual(cost_r(self.p, costs), 0.2)
        win, _ = walk(self.p, self.candles, 1, costs)
        self.assertAlmostEqual(win, 2.0 - 0.2)
        lose, _ = walk(self.p, [Candle(0, 100, 100, 100, 100), Candle(60, 100, 100, 98, 98)], 1, costs)
        self.assertAlmostEqual(lose, -1.2)

    def test_stop_wins_a_tie_inside_a_candle(self):
        both = [Candle(0, 100, 100, 100, 100), Candle(60, 100, 103, 98, 100)]
        r, _ = walk(self.p, both, 1)
        self.assertEqual(r, -1.0)

    def test_signal_never_sees_the_future(self):
        candles = synthetic_candles(120)
        seen = []
        simulate(candles, lambda h: seen.append(len(h)) or None, 40, 100)
        self.assertEqual(max(seen), 99)
        self.assertEqual(seen, sorted(seen))


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.p = Proposal("X", Side.LONG, 100.0, 99.0, 102.0, "t")
        self.candles = synthetic_candles()

    def test_rejects_wrong_side_stop(self):
        bad = Proposal("X", Side.LONG, 100.0, 101.0, 102.0, "t")
        self.assertEqual(Validator().review(bad, self.candles, lambda h: None).vote, Vote.REJECT)

    def test_rejects_too_few_trades(self):
        v = Validator(min_trades=5).review(self.p, self.candles, lambda h: None)
        self.assertEqual(v.vote, Vote.REJECT)

    def test_rejects_fragile_edge(self):
        always_long = lambda h: Proposal("X", Side.LONG, h[-1].close, h[-1].close - 1, h[-1].close + 2, "t")
        never = lambda h: None
        v = Validator(**{**EASY, "min_robust": 0.5}).review(self.p, self.candles, always_long,
                                                            neighbors=[never, never, never, never])
        self.assertEqual(v.vote, Vote.REJECT)
        self.assertIn("fragile", v.reason)

    def test_rejects_inconsistent_windows(self):
        def first_half_winner(h):
            # long with a close target early in the holdout, impossible target later
            n = len(h)
            tgt = h[-1].close + (0.01 if n < 420 else 1000)
            return Proposal("X", Side.LONG, h[-1].close, h[-1].close - 0.02, tgt, "t")
        v = Validator(**{**EASY, "min_positive_windows": 0.9}).review(self.p, self.candles, first_half_winner)
        self.assertIn("windows", v.reason)

    def test_costs_can_flip_the_verdict(self):
        sig = lambda h: Proposal("X", Side.LONG, h[-1].close, h[-1].close - 1, h[-1].close + 1.05, "t")
        cheap = Validator(**{**EASY, "min_expectancy_r": 0.0}).review(self.p, self.candles, sig, costs=NO_COSTS)
        pricey = Validator(**{**EASY, "min_expectancy_r": 0.0}).review(self.p, self.candles, sig,
                                                                       costs=CostModel(spread=0.5))
        self.assertLess(pricey.metrics["expectancy_r"], cheap.metrics["expectancy_r"])


class StrategyTests(unittest.TestCase):
    def test_variants_nudge_each_setting(self):
        vs = StrategyAnalyst(stop_atr=2.0, reward_risk=2.0).variants(0.2)
        self.assertEqual(len(vs), 4)
        self.assertEqual({(v.stop_atr, v.reward_risk) for v in vs},
                         {(1.6, 2.0), (2.4, 2.0), (2.0, 1.6), (2.0, 2.4)})

    def test_can_skip_high_volatility(self):
        snap = MarketObserver().observe("X", synthetic_candles(300, drift=0.003))
        from dataclasses import replace
        high = replace(snap, vol_state="high")
        self.assertIsNotNone(StrategyAnalyst().propose(high))
        self.assertIsNone(StrategyAnalyst(skip_high_vol=True).propose(high))


class RiskTests(unittest.TestCase):
    def setUp(self):
        self.p = Proposal("X", Side.LONG, 100.0, 99.0, 102.0, "t")

    def test_sizes_by_risk(self):
        v = RiskManager().review(self.p, AccountState(equity=1000.0))
        self.assertEqual(v.vote, Vote.APPROVE)
        self.assertAlmostEqual(v.size, 10.0)  # 1% of 1000 / 1.0 stop distance

    def test_kill_switch_vetoes(self):
        self.assertEqual(RiskManager().review(self.p, AccountState(1000.0, kill_switch=True)).vote, Vote.REJECT)

    def test_daily_loss_limit(self):
        self.assertEqual(RiskManager().review(self.p, AccountState(1000.0, daily_pnl=-40.0)).vote, Vote.REJECT)

    def test_drawdown_limit(self):
        v = RiskManager().review(self.p, AccountState(850.0, peak_equity=1000.0))
        self.assertEqual(v.vote, Vote.REJECT)
        self.assertIn("drawdown", v.reason)

    def test_losing_streak_cooldown(self):
        v = RiskManager().review(self.p, AccountState(1000.0, consecutive_losses=4))
        self.assertEqual(v.vote, Vote.REJECT)

    def test_rejects_when_costs_eat_the_risk(self):
        v = RiskManager().review(self.p, AccountState(1000.0), costs=CostModel(spread=0.5))
        self.assertEqual(v.vote, Vote.REJECT)
        self.assertIn("costs", v.reason)

    def test_exposure_cap_shrinks_size(self):
        rm = RiskManager(RiskLimits(max_notional_multiple=0.5))
        v = rm.review(self.p, AccountState(1000.0))
        self.assertAlmostEqual(v.size, 1000.0 * 0.5 / 100.0)
        self.assertIn("capped", v.reason)


class SkepticTests(unittest.TestCase):
    def test_rejects_thin_evidence(self):
        self.assertEqual(Skeptic().assess([2.0, -1.0, 2.0]).vote, Vote.REJECT)

    def test_rejects_luck(self):
        results = [2.0, -1.0] * 5 + [-1.0, -1.0, -1.0, 2.0] * 3
        self.assertEqual(Skeptic(max_luck=0.05).assess(results).vote, Vote.REJECT)

    def test_approves_solid_edge(self):
        results = [2.0] * 15 + [-1.0] * 5 + [2.0] * 10 + [-1.0] * 5
        self.assertEqual(Skeptic().assess(results).vote, Vote.APPROVE)

    def test_rejects_edge_that_flips(self):
        results = [2.0] * 12 + [-1.0] * 3 + [-1.0] * 12 + [2.0] * 3
        self.assertEqual(Skeptic().assess(results).vote, Vote.REJECT)

    def test_rejects_edge_from_one_lucky_trade(self):
        results = [-0.1, 0.1] * 10 + [-0.5] * 5 + [9.0]
        v = Skeptic(max_luck=1.0).assess(results)
        self.assertEqual(v.vote, Vote.REJECT)
        self.assertIn("single trade", v.reason)


class ResearchTests(unittest.TestCase):
    def test_runs_every_combination_and_warns(self):
        coord = Coordinator(MarketObserver(), StrategyAnalyst(), Validator(), RiskManager(),
                            researcher=ResearchAnalyst(grid={"stop_atr": [1.5, 2.0], "reward_risk": [2.0]}))
        report = coord.research("X", synthetic_candles())
        self.assertEqual(len(report.results), 2)
        self.assertTrue(any("tested 2 combinations" in n for n in report.notes))
        self.assertTrue(any("of 2 combinations were profitable" in n for n in report.notes))

    def test_short_history(self):
        coord = Coordinator(MarketObserver(), StrategyAnalyst(), Validator(), RiskManager(),
                            researcher=ResearchAnalyst())
        self.assertIsNone(coord.research("X", synthetic_candles(50)).best)


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
            m.record(paper(100.0, 99.0))
        self.assertTrue(m.report().drift)
        self.assertTrue(m.halted())

    def test_no_halt_before_enough_trades(self):
        m = PaperMonitor(expected_r=0.5, min_trades=10)
        m.record(paper(100.0, 99.0))
        self.assertFalse(m.halted())

    def test_only_recent_trades_count(self):
        m = PaperMonitor(expected_r=0.5, tolerance_r=0.3, min_trades=5, window=5)
        for _ in range(20):
            m.record(paper(100.0, 102.0))  # long run of good trades
        for _ in range(5):
            m.record(paper(100.0, 99.0))   # edge stops working
        self.assertTrue(m.halted())

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
        coord = Coordinator(MarketObserver(), StrategyAnalyst(), Validator(**EASY), RiskManager(),
                            journal=TradeJournal())
        d = coord.decide("X", synthetic_candles(), AccountState(equity=1000.0))
        self.assertTrue(d.approved)
        self.assertEqual(len(coord.journal.entries), 1)
        self.assertEqual(coord.review_journal(), ["no closed trades yet"])

    def test_review_flags_losing_bucket(self):
        j = TradeJournal()
        p = Proposal("X", Side.SHORT, 100.0, 101.0, 98.0, "t")
        for r in [-1, -1, 2, -1, -1, -1]:
            eid = j.log_decision(Decision(p, True, (), 1.0, "ok"), Regime.RANGING)
            j.close(eid, float(r))
        text = "\n".join(j.review(min_bucket=5))
        self.assertIn("losing here", text)
        self.assertIn("longest losing streak: 3", text)

    def test_spots_revenge_trading(self):
        j = TradeJournal()
        t = 1_000_000
        for k in range(4):
            lost = j.log_manual("X", Side.LONG, Regime.RANGING, 1.0, ts=t)
            j.close(lost, -1.0, ts=t + 60)
            again = j.log_manual("X", Side.LONG, Regime.RANGING, 1.0, ts=t + 120)  # 1 minute after the loss
            j.close(again, -1.0, ts=t + 180)
            t += 100_000
        self.assertTrue(any("revenge" in line for line in j.review()))

    def test_spots_overtrading_and_moved_stops(self):
        j = TradeJournal()
        for k in range(7):
            eid = j.log_manual("X", Side.LONG, Regime.RANGING, 1.0, ts=86400 * 10 + k * 3600, moved_stop=k < 4)
            j.close(eid, -1.0 if k < 4 else 1.0, ts=86400 * 10 + k * 3600 + 600)
        text = "\n".join(j.review())
        self.assertIn("overtrading", text)
        self.assertIn("moved the stop on 4 trades", text)

    def test_compares_following_quorum_with_manual(self):
        j = TradeJournal()
        p = Proposal("X", Side.LONG, 100.0, 99.0, 102.0, "t")
        for k in range(5):
            a = j.log_decision(Decision(p, True, (), 1.0, "ok"), Regime.RANGING, ts=k * 100_000)
            j.close(a, 1.0, ts=k * 100_000 + 10)
            b = j.log_manual("X", Side.LONG, Regime.RANGING, 1.0, ts=k * 100_000 + 50_000)
            j.close(b, -1.0, ts=k * 100_000 + 50_010)
        self.assertTrue(any(line.startswith("followed Quorum") for line in j.review()))

    def test_save_and_load_round_trip(self):
        import os, tempfile
        j = TradeJournal()
        eid = j.log_manual("X", Side.LONG, Regime.RANGING, 1.0, ts=5, moved_stop=True)
        j.close(eid, 1.5, ts=9)
        path = os.path.join(tempfile.mkdtemp(), "j.json")
        j.save(path)
        self.assertEqual(TradeJournal.load(path).entries, j.entries)


class CoordinatorTests(unittest.TestCase):
    def test_bad_data_means_no_trade(self):
        d = make_coordinator().decide("X", synthetic_candles(10), AccountState(equity=1000.0))
        self.assertFalse(d.approved)
        self.assertIsNone(d.proposal)

    def test_risk_veto_blocks_trade(self):
        d = make_coordinator(**EASY).decide("X", synthetic_candles(), AccountState(1000.0, kill_switch=True))
        self.assertFalse(d.approved)
        self.assertEqual(d.size, 0.0)

    def test_validator_veto_blocks_trade(self):
        d = make_coordinator(min_trades=10_000).decide("X", synthetic_candles(), AccountState(equity=1000.0))
        self.assertFalse(d.approved)

    def test_skeptic_veto_blocks_trade(self):
        coord = Coordinator(MarketObserver(), StrategyAnalyst(), Validator(**EASY), RiskManager(),
                            skeptic=Skeptic(min_trades=10_000))
        d = coord.decide("X", synthetic_candles(), AccountState(equity=1000.0))
        self.assertFalse(d.approved)
        self.assertIn("skeptic", d.reason)

    def test_data_problems_block_and_feed_mismatch_is_caught(self):
        candles = synthetic_candles()
        other = [Candle(c.ts, c.open, c.high, c.low, c.close * 1.02) for c in candles]
        d = make_coordinator(**EASY).decide("X", candles, AccountState(equity=1000.0), reference=other)
        self.assertFalse(d.approved)
        self.assertIn("feeds disagree", d.reason)

    def test_first_approval_sets_the_monitor_baseline(self):
        m = PaperMonitor()
        coord = Coordinator(MarketObserver(), StrategyAnalyst(), Validator(**EASY), RiskManager(), monitor=m)
        d = coord.decide("X", synthetic_candles(), AccountState(equity=1000.0))
        self.assertTrue(d.approved)
        self.assertIsNotNone(m.expected_r)

    def test_costs_reach_the_risk_manager(self):
        coord = Coordinator(MarketObserver(), StrategyAnalyst(), Validator(**EASY), RiskManager(),
                            costs=CostModel(spread=50.0))
        d = coord.decide("X", synthetic_candles(), AccountState(equity=1000.0))
        self.assertFalse(d.approved)
        self.assertIn("costs", d.reason)


class SystemEvalTests(unittest.TestCase):
    def test_all_system_evals_pass(self):
        failed = [r for r in run_evals() if not r.passed]
        self.assertEqual(failed, [], [f"{r.name}: {r.detail}" for r in failed])


if __name__ == "__main__":
    unittest.main()
