import unittest

import os
import tempfile

from quorum.data import load_candles, load_markets, synthetic_candles, synthetic_market, synthetic_markets
from quorum.agents import (AccountState, Breakout, DataQualityAgent, FlowAnalyst, MarketFlow, MarketObserver,
                           MeanReversion, PaperMonitor, PaperTrade, PortfolioLimits, PortfolioRiskManager,
                           ResearchAnalyst, RiskLimits, RiskManager, Skeptic, StrategyAnalyst, StrategyLab,
                           TradeJournal, TrendFollow, Validator)
from quorum.config import QuorumConfig, build_coordinator
from quorum.evals import fake_analysis
from quorum.report import render_report
from quorum.scan import scan_markets
from quorum.__main__ import main as cli_main
from quorum.backtest import NO_COSTS, CostModel, Trade, cost_r, simulate, walk
from quorum.coordinator import Coordinator
from quorum.evals import run_evals
from quorum.types import Candle, Decision, MarketSnapshot, NewsEvent, Proposal, Regime, Side, Vote
from quorum.agents import AllDayLong, Context, NewsBreakout, NewsFade, WeeklyDip, make_strategy
from quorum.backtest import Prefix, drift_adjusted_r, simulate_detail
from quorum.clock import DailyIndex, format_local, infer_timeframe, local_midnight, minute_of_day, weekday
from quorum.data import _session_open, load_news, synthetic_news_market, synthetic_session_market
from quorum.profiles import PROFILES
from quorum.scan import score_reference

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


def snap(**kw):
    base = dict(symbol="X", last_close=100.0, regime=Regime.RANGING, volatility=0.003, atr=1.0, data_ok=True,
                notes=(), vol_state="normal", fast_ma=100.0, slow_ma=100.0, channel_high=101.0,
                channel_low=99.0, stretch=0.0)
    base.update(kw)
    return MarketSnapshot(**base)


class StrategyFamilyTests(unittest.TestCase):
    def test_mean_reversion_fades_a_stretch_back_to_the_average(self):
        p = MeanReversion(entry_stretch=2.0, stop_atr=1.5).propose(snap(last_close=97.0, slow_ma=100.0, stretch=-3.0))
        self.assertEqual(p.side, Side.LONG)
        self.assertEqual(p.target, 100.0)
        s = MeanReversion(entry_stretch=2.0).propose(snap(last_close=103.0, slow_ma=100.0, stretch=3.0))
        self.assertEqual(s.side, Side.SHORT)
        self.assertIsNone(MeanReversion(entry_stretch=2.0).propose(snap(stretch=1.0)))

    def test_breakout_needs_a_close_beyond_the_channel(self):
        self.assertEqual(Breakout().propose(snap(last_close=102.0)).side, Side.LONG)
        self.assertEqual(Breakout().propose(snap(last_close=98.0)).side, Side.SHORT)
        self.assertIsNone(Breakout().propose(snap(last_close=100.0)))

    def test_every_family_nudges_its_own_settings(self):
        self.assertEqual(len(MeanReversion().variants()), 4)
        self.assertEqual(len(Breakout().variants()), 4)
        self.assertTrue(all(isinstance(v, TrendFollow) for v in TrendFollow().variants()))

    def test_old_name_still_works(self):
        self.assertIs(StrategyAnalyst, TrendFollow)


class DataLoadingTests(unittest.TestCase):
    def write(self, text):
        path = os.path.join(tempfile.mkdtemp(), "EURUSD.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_comma_csv_with_iso_times(self):
        path = self.write("time,open,high,low,close\n2024-01-02 10:00:00,1.0,1.2,0.9,1.1\n2024-01-02 10:01:00,1.1,1.3,1.0,1.2\n")
        c = load_candles(path)
        self.assertEqual(len(c), 2)
        self.assertEqual(c[1].ts - c[0].ts, 60)
        self.assertEqual(c[1].close, 1.2)

    def test_mt5_style_tab_export_with_split_date_and_time(self):
        path = self.write("<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\n2024.01.02\t10:00\t1.0\t1.2\t0.9\t1.1\n2024.01.02\t10:05\t1.1\t1.3\t1.0\t1.2\n")
        c = load_candles(path)
        self.assertEqual(c[1].ts - c[0].ts, 300)

    def test_epoch_milliseconds(self):
        path = self.write("timestamp,open,high,low,close\n1704189600000,1,2,0.5,1.5\n1704189660000,1.5,2,1,1.2\n")
        c = load_candles(path)
        self.assertEqual(c[1].ts - c[0].ts, 60)

    def test_missing_column_is_explained(self):
        with self.assertRaises(ValueError) as ctx:
            load_candles(self.write("time,open,high,low\n1,1,1,1\n"))
        self.assertIn("close", str(ctx.exception))

    def test_market_name_comes_from_the_file_name(self):
        path = self.write("time,open,high,low,close\n1,1,1,1,1\n")
        self.assertEqual(list(load_markets([path])), ["EURUSD"])


class FlowTests(unittest.TestCase):
    def label(self, kind, seed):
        return FlowAnalyst().analyze("X", synthetic_market(kind, 3000, seed), MarketObserver())

    def test_trending_market_is_labeled_trending(self):
        self.assertEqual(self.label("trending", 1).structure, "trending")

    def test_mean_reverting_market_is_labeled_mean_reverting(self):
        self.assertEqual(self.label("mean_reverting", 2).structure, "mean-reverting")

    def test_random_market_is_labeled_random_and_warned(self):
        r = self.label("random", 3)
        self.assertEqual(r.structure, "random")
        self.assertTrue(r.notes)

    def test_short_history_is_unknown(self):
        self.assertEqual(FlowAnalyst().analyze("X", synthetic_candles(40), MarketObserver()).structure, "unknown")

    def test_ranks_by_flow_and_measures_correlation(self):
        markets = synthetic_markets(n=600, rho=0.5)
        fa = FlowAnalyst()
        reports = {s: fa.analyze(s, c, MarketObserver()) for s, c in markets.items()}
        flow = fa.compare(reports, markets)
        scores = [reports[s].flow_score for s in flow.ranking]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len(flow.correlation), 3)
        self.assertTrue(all(0.2 < v < 0.8 for v in flow.correlation.values()))

    def test_markets_that_move_together_are_called_one_bet(self):
        a = synthetic_market("random", 600, 4)
        b = [Candle(c.ts, c.open * 1.01, c.high * 1.01, c.low * 1.01, c.close * 1.01) for c in a]
        fa = FlowAnalyst()
        flow = fa.compare({}, {"A": a, "B": b})
        self.assertGreater(flow.corr("A", "B"), 0.99)
        self.assertTrue(any("one bet" in n for n in flow.notes))


class StrategyLabTests(unittest.TestCase):
    def test_searches_every_combination_and_is_honest_about_it(self):
        candles = synthetic_market("random", 1200, 9)
        lab = StrategyLab().build("X", candles, MarketObserver(), structure="random")
        self.assertEqual(lab.tested, 21)
        self.assertTrue(any("looks better than it really is" in n for n in lab.notes))
        self.assertTrue(any("random walk" in n for n in lab.notes))
        scores = [c.score for c in lab.candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_never_picks_an_ineligible_candidate(self):
        lab = StrategyLab().build("X", synthetic_market("mean_reverting", 1500, 5), MarketObserver())
        if lab.chosen is not None:
            self.assertTrue(lab.chosen.eligible)
            self.assertGreater(lab.chosen.score, 0)

    def test_short_history_finds_nothing(self):
        lab = StrategyLab().build("X", synthetic_candles(60), MarketObserver())
        self.assertIsNone(lab.chosen)
        self.assertEqual(lab.tested, 0)


class PortfolioTests(unittest.TestCase):
    def plan(self, analyses, corr=None, limits=PortfolioLimits(), account=None):
        return PortfolioRiskManager(limits).allocate(analyses, MarketFlow({}, corr or {}, (), ()),
                                                     account or AccountState(equity=1000.0))

    def test_nothing_approved_means_nothing_to_do(self):
        self.assertEqual(self.plan([]).allocations, ())

    def test_linked_markets_count_as_one_bet(self):
        a = fake_analysis("A", Side.LONG, 10, 0.5)
        b = fake_analysis("B", Side.LONG, 10, 0.9)
        plan = self.plan([a, b], {("A", "B"): 0.9})
        self.assertEqual([x.symbol for x in plan.allocations], ["B"])  # the stronger one is kept
        self.assertIn("same bet", plan.skipped[0][1])

    def test_opposite_sides_on_negatively_linked_markets_are_also_one_bet(self):
        plan = self.plan([fake_analysis("A", Side.LONG, 10, 0.5), fake_analysis("B", Side.SHORT, 10, 0.4)],
                         {("A", "B"): -0.9})
        self.assertEqual(len(plan.allocations), 1)

    def test_independent_markets_can_both_trade(self):
        plan = self.plan([fake_analysis("A", Side.LONG, 10, 0.5), fake_analysis("B", Side.LONG, 10, 0.4)],
                         {("A", "B"): 0.1})
        self.assertEqual(len(plan.allocations), 2)
        self.assertAlmostEqual(plan.total_risk, 0.02)

    def test_total_risk_cap(self):
        analyses = [fake_analysis(s, Side.LONG, 10, e) for s, e in (("A", 0.9), ("B", 0.8), ("C", 0.7))]
        plan = self.plan(analyses, limits=PortfolioLimits(max_total_risk=0.02, max_positions=3))
        self.assertEqual(len(plan.allocations), 2)
        self.assertIn("cap", plan.skipped[0][1])

    def test_position_limit_and_open_positions(self):
        analyses = [fake_analysis("A", Side.LONG, 10, 0.9), fake_analysis("B", Side.LONG, 10, 0.8)]
        self.assertEqual(len(self.plan(analyses, limits=PortfolioLimits(max_positions=1)).allocations), 1)
        busy = AccountState(equity=1000.0, open_positions=1)
        self.assertEqual(len(self.plan(analyses, limits=PortfolioLimits(max_positions=2), account=busy).allocations), 1)


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.markets = synthetic_markets(n=700)
        self.result = scan_markets(self.markets, AccountState(equity=1000.0), QuorumConfig(), workers=1)

    def test_one_analysis_per_market_in_order(self):
        self.assertEqual([a.symbol for a in self.result.analyses], list(self.markets))

    def test_bad_data_in_one_market_does_not_stop_the_others(self):
        broken = dict(self.markets)
        c = broken["TREND-A"]
        broken["TREND-A"] = c[:100] + [Candle(c[100].ts, 100, 99, 101, 100)] + c[101:]
        r = scan_markets(broken, AccountState(equity=1000.0), QuorumConfig(), workers=1)
        a = {x.symbol: x for x in r.analyses}
        self.assertTrue(a["TREND-A"].decision.reason.startswith("data check failed"))
        self.assertIsNotNone(a["REVERT-B"].flow)

    def test_report_names_every_market_and_the_warnings(self):
        text = render_report(self.result, 1000.0)
        for name in self.markets:
            self.assertIn(name, text)
        self.assertIn("Holdout exam", text)
        self.assertIn("Read this before acting", text)

    def test_exam_runs_even_without_a_live_setup(self):
        team = build_coordinator(QuorumConfig(), TrendFollow(), with_state=False)
        verdicts = team.exam("X", synthetic_candles())
        self.assertEqual(len(verdicts), 2)  # validator and skeptic

    def test_command_line_scan_writes_a_report(self):
        folder = tempfile.mkdtemp()
        paths = []
        for name, candles in list(self.markets.items()):
            path = os.path.join(folder, name + ".csv")
            with open(path, "w") as f:
                f.write("time,open,high,low,close\n")
                for c in candles:
                    f.write(f"{c.ts},{c.open},{c.high},{c.low},{c.close}\n")
            paths.append(path)
        out = os.path.join(folder, "report.md")
        self.assertEqual(cli_main(["scan", *paths, "--workers", "1", "--out", out]), 0)
        with open(out) as f:
            self.assertIn("Quorum market scan", f.read())


PH = 8
SUN_2300_UTC = 1704063600   # Sunday 31 Dec 2023 23:00 UTC, which is Monday 1 Jan 07:00 Philippine time


def session_candles(price_fn, days, start=SUN_2300_UTC, tf=300, skip=lambda ts: False):
    """US30-style candles (nightly break, weekend closed) with a price you control, 5-minute by default."""
    out, prev, ts = [], 40000.0, start
    while ts < start + days * 86400:
        if _session_open(ts) and not skip(ts):
            c = price_fn(ts)
            out.append(Candle(ts, prev, max(prev, c), min(prev, c), c))
            prev = c
        ts += tf
    return out


def ph(ts):
    return format_local(ts, PH)


class ClockTests(unittest.TestCase):
    def test_philippine_clock(self):
        self.assertEqual(ph(SUN_2300_UTC), "Mon 07:00")
        self.assertEqual(weekday(SUN_2300_UTC, PH), 0)
        self.assertEqual(minute_of_day(SUN_2300_UTC, PH), 7 * 60)
        self.assertEqual(ph(SUN_2300_UTC + 2 * 86400 - 125 * 60), "Wed 04:55")

    def test_local_midnight(self):
        self.assertEqual(ph(local_midnight(SUN_2300_UTC + 3600, PH)), "Mon 00:00")

    def test_timeframe_survives_a_gap(self):
        c = synthetic_candles(80)
        del c[30:40]
        self.assertEqual(infer_timeframe(c[:60]), 60)
        self.assertIsNone(infer_timeframe(c[:1]))

    def test_daily_index_counts_trading_days_and_is_incremental(self):
        c = session_candles(lambda ts: 40000.0 + (ts - SUN_2300_UTC) / 86400, 14)
        full = DailyIndex(PH, 7)
        full.update(c)
        stepwise = DailyIndex(PH, 7)
        for n in range(1, len(c) + 1, 37):
            stepwise.update(Prefix(c, n))
        stepwise.update(c)
        self.assertEqual(full.closes, stepwise.closes)
        self.assertEqual(len(full.closes), 9)   # nine completed trading days before the tenth, current one
        self.assertIsNone(full.sma(25))
        self.assertIsNotNone(full.sma(5))

    def test_daily_index_resets_when_history_changes(self):
        c = session_candles(lambda ts: 40000.0, 10)
        idx = DailyIndex(PH, 7)
        idx.update(c)
        idx.update(c[:50])            # shorter history, as when a new replay starts
        self.assertEqual(idx.closes, [])


class EngineTests(unittest.TestCase):
    def test_prefix_behaves_like_a_list_without_copying(self):
        c = synthetic_candles(50)
        p = Prefix(c, 20)
        self.assertEqual(len(p), 20)
        self.assertIs(p[-1], c[19])
        self.assertEqual(p[3:6], c[3:6])
        self.assertEqual(p[-3:], c[17:20])
        self.assertEqual(len(list(p)), 20)
        with self.assertRaises(IndexError):
            p[20]

    def test_time_exit_fills_at_the_open_of_the_first_candle_at_or_after_the_time(self):
        p = Proposal("X", Side.LONG, 100.0, 95.0, None, "t", exit_ts=180)
        c = [Candle(0, 100, 100, 100, 100), Candle(60, 100, 120, 100, 119), Candle(120, 119, 119, 118, 118),
             Candle(180, 117, 118, 116, 117), Candle(240, 117, 117, 100, 100)]
        r, j = walk(p, c, 1)
        self.assertEqual(j, 3)
        self.assertAlmostEqual(r, (117 - 100) / 5)   # no take-profit, so the 120 high changes nothing

    def test_no_target_means_the_trade_only_ends_on_stop_or_time(self):
        p = Proposal("X", Side.LONG, 100.0, 99.0, None, "t")
        c = [Candle(0, 100, 100, 100, 100)] + [Candle(60 * i, 100 + i, 101 + i, 100 + i, 101 + i) for i in range(1, 6)]
        self.assertIsNone(walk(p, c, 1))

    def test_gap_through_the_stop_is_booked_at_the_gap_price(self):
        long_p = Proposal("X", Side.LONG, 100.0, 99.0, 102.0, "t")
        self.assertAlmostEqual(walk(long_p, [Candle(0, 100, 100, 100, 100), Candle(60, 95, 96, 94, 95)], 1)[0], -5.0)
        short_p = Proposal("X", Side.SHORT, 100.0, 101.0, 98.0, "t")
        self.assertAlmostEqual(walk(short_p, [Candle(0, 100, 100, 100, 100), Candle(60, 104, 105, 103, 104)], 1)[0], -4.0)

    def test_news_time_costs_are_inflated_only_near_a_release(self):
        base = CostModel(spread=0.3, slippage=0.05)
        c = base.with_news([10_000], 5.0, 600)
        self.assertAlmostEqual(c.round_trip(None), 0.4)
        self.assertAlmostEqual(c.round_trip(9_500), 2.0)
        self.assertAlmostEqual(c.round_trip(10_600), 2.0)
        self.assertAlmostEqual(c.round_trip(11_000), 0.4)

    def test_overnight_financing_is_charged_for_time_held(self):
        costs = CostModel(overnight_pct=0.0001)
        p = Proposal("X", Side.LONG, 40000.0, 38000.0, None, "t", exit_ts=86400)
        c = [Candle(0, 40000, 40000, 40000, 40000), Candle(60, 40000, 40000, 40000, 40000), Candle(86400, 40000, 40000, 40000, 40000)]
        free, _ = walk(p, c, 1)
        paid, _ = walk(p, c, 1, costs)
        self.assertAlmostEqual(free - paid, 40000 * 0.0001 / 2000, places=4)  # 4 price points on a 2000 point stop

    def test_drift_is_subtracted_per_candle_held(self):
        c = [Candle(i, 100 + i, 100 + i, 100 + i, 100 + i) for i in range(20)]
        t = [Trade(2, 7, 5.0, 1.0, Side.LONG)]    # earned 5 over 5 candles in a market rising 1 a candle
        self.assertAlmostEqual(drift_adjusted_r(t, c, 0)[0], 0.0, places=6)
        s = [Trade(2, 7, -5.0, 1.0, Side.SHORT)]
        self.assertAlmostEqual(drift_adjusted_r(s, c, 0)[0], 0.0, places=6)

    def test_replay_takes_one_trade_at_a_time(self):
        c = synthetic_candles(200)
        always = lambda h: Proposal("X", Side.LONG, h[-1].close, h[-1].close - 5, None, "t", h[-1].ts + 61 * 5, h[-1].ts + 60)
        trades = simulate_detail(c, always, 10)
        self.assertTrue(all(b.entry > a.exit for a, b in zip(trades, trades[1:])))


class RiskNoTargetTests(unittest.TestCase):
    def test_a_trade_with_no_take_profit_is_sized_and_approved(self):
        p = Proposal("US30", Side.LONG, 40000.0, 38000.0, None, "t")
        v = RiskManager().review(p, AccountState(equity=1000.0))
        self.assertEqual(v.vote, Vote.APPROVE)
        self.assertAlmostEqual(v.size, 1000.0 * 0.01 / 2000.0)

    def test_still_rejects_a_stop_on_the_wrong_side(self):
        p = Proposal("US30", Side.LONG, 40000.0, 41000.0, None, "t")
        self.assertEqual(RiskManager().review(p, AccountState(equity=1000.0)).vote, Vote.REJECT)

    def test_news_time_costs_reach_the_risk_check(self):
        costs = CostModel(spread=0.3, slippage=0.05).with_news([1000], 5.0, 600)
        quiet = Proposal("XAU", Side.LONG, 2000.0, 1997.0, 2006.0, "t", ts=50_000)
        busy = Proposal("XAU", Side.LONG, 2000.0, 1997.0, 2006.0, "t", ts=1000)
        self.assertEqual(RiskManager().review(quiet, AccountState(1000.0), costs=costs).vote, Vote.APPROVE)
        v = RiskManager().review(busy, AccountState(1000.0), costs=costs)
        self.assertEqual(v.vote, Vote.REJECT)
        self.assertIn("costs", v.reason)


class SessionDataTests(unittest.TestCase):
    def test_nightly_and_weekend_closures_are_normal_for_a_session_market(self):
        c = synthetic_session_market(years=0.1, seed=2)
        r = DataQualityAgent(session_market=True).check(c)
        self.assertTrue(r.ok, r.issues)
        self.assertTrue(any("closures" in w for w in r.warnings))
        self.assertFalse(DataQualityAgent().check(c).ok)

    def test_many_missing_candles_inside_sessions_still_block(self):
        c = synthetic_session_market(years=0.1, seed=2)
        thinned = [x for i, x in enumerate(c) if i % 12 != 5]
        r = DataQualityAgent(session_market=True).check(thinned)
        self.assertFalse(r.ok)
        self.assertTrue(any("missing" in i for i in r.issues))

    def test_flow_ignores_returns_across_closures(self):
        c = synthetic_session_market(years=0.3, seed=5, vol_per_year=0.02)
        r = FlowAnalyst().analyze("X", c, MarketObserver())
        self.assertNotEqual(r.structure, "unknown")


class WeeklyDipRuleTests(unittest.TestCase):
    """Your Monday-dip rule, checked on hand-built candles with known prices at known Philippine times."""

    MONDAY = SUN_2300_UTC + 35 * 86400    # Monday 07:00 PH of week 6, after 25 flat trading days

    def trades(self, price_fn, days=39):
        c = session_candles(price_fn, days)
        s = WeeklyDip().bind(Context())
        return c, simulate_detail(c, lambda h: s.signal("US30", h, MarketObserver()), 100, None, NO_COSTS)

    def hours(self, ts):
        return (ts - self.MONDAY) / 3600

    def test_above_the_average_all_day_means_no_trade_that_week(self):
        _, tr = self.trades(lambda ts: 40000.0 if ts < self.MONDAY else 40500.0)
        self.assertEqual(tr, [])

    def test_buys_the_first_dip_below_the_average_and_exits_wednesday_morning(self):
        c, tr = self.trades(lambda ts: 40000.0 if ts < self.MONDAY else (39500.0 if 5 <= self.hours(ts) < 7 else 40500.0))
        self.assertEqual(len(tr), 1)
        t = tr[0]
        self.assertEqual(ph(c[t.entry].ts + 300), "Mon 12:05")
        self.assertEqual(ph(c[t.exit].ts), "Wed 04:55")
        self.assertAlmostEqual(c[t.entry].close * 0.05, t.risk)            # stop is 5% below the entry
        self.assertAlmostEqual(t.r, (40500 - 39500) / (39500 * 0.05))      # no take-profit, closed at the time exit

    def test_already_below_the_average_at_the_start_buys_on_the_first_candle_after_7_04(self):
        c, tr = self.trades(lambda ts: 40000.0 if ts < self.MONDAY else 39500.0)
        self.assertEqual(ph(c[tr[0].entry].ts + 300), "Mon 07:05")

    def test_tuesday_early_morning_still_belongs_to_mondays_trading_day(self):
        c, tr = self.trades(lambda ts: 40000.0 if ts < self.MONDAY else (39500.0 if 20 <= self.hours(ts) < 22 else 40500.0))
        self.assertEqual(len(tr), 1)
        self.assertEqual(ph(c[tr[0].entry].ts + 300), "Tue 03:05")
        self.assertEqual(ph(c[tr[0].exit].ts), "Wed 04:55")

    def test_a_dip_after_mondays_trading_day_is_too_late(self):
        _, tr = self.trades(lambda ts: 40000.0 if ts < self.MONDAY else (39500.0 if 27 <= self.hours(ts) < 29 else 40500.0))
        self.assertEqual(tr, [])

    def test_a_stopped_out_trade_is_not_re_entered_the_same_week(self):
        def px(ts):
            h = self.hours(ts)
            if ts < self.MONDAY:
                return 40000.0
            return 39500.0 if 5 <= h < 7 else (40500.0 if h < 5 else 36900.0)    # then a 6% crash that stays low
        _, tr = self.trades(px)
        self.assertEqual(len(tr), 1)
        self.assertAlmostEqual(tr[0].r, -1.0)

    def test_needs_25_completed_days_before_it_trades(self):
        c = session_candles(lambda ts: 39000.0, 20)
        s = WeeklyDip().bind(Context())
        self.assertEqual(simulate_detail(c, lambda h: s.signal("US30", h, MarketObserver()), 100, None), [])

    def test_a_shorter_average_changes_what_counts_as_a_dip(self):
        def px(ts):
            if ts < self.MONDAY - 5 * 86400:      # before Wednesday of the week before
                return 40000.0
            return 42000.0 if ts < self.MONDAY else 41000.0     # Wed, Thu, Fri were high; Monday is lower than that
        c = session_candles(px, 39)
        longer = WeeklyDip(ma_days=25).bind(Context())
        shorter = WeeklyDip(ma_days=3).bind(Context())
        n_long = len(simulate_detail(c, lambda h: longer.signal("US30", h, MarketObserver()), 100))
        n_short = len(simulate_detail(c, lambda h: shorter.signal("US30", h, MarketObserver()), 100))
        self.assertEqual(n_long, 0)     # 41000 is above the 25-day average of about 40240
        self.assertEqual(n_short, 1)    # but below the 3-day average of 42000

    def test_hourly_candles_work_but_four_hour_candles_do_not(self):
        self.assertIsNone(WeeklyDip().compatible(3600))
        self.assertIn("hourly", WeeklyDip().compatible(14400))


class AllDayLongRuleTests(unittest.TestCase):
    def trades(self, price_fn, days=28, skip=lambda ts: False, **kw):
        c = session_candles(price_fn, days, skip=skip)
        s = AllDayLong(**kw).bind(Context())
        return c, simulate_detail(c, lambda h: s.signal("US30", h, MarketObserver()), 10, None, NO_COSTS)

    def test_enters_at_7_05_and_closes_at_5_50_the_next_morning(self):
        c, tr = self.trades(lambda ts: 40000.0)
        self.assertGreaterEqual(len(tr), 19)
        for t in tr:
            self.assertEqual(ph(c[t.entry].ts + 300), ph(c[t.entry].ts + 300)[:4] + "07:05")
            self.assertTrue(ph(c[t.exit].ts).endswith("05:50"))
            self.assertAlmostEqual((c[t.exit].ts - (c[t.entry].ts + 300)) / 3600, 22.75)

    def test_one_trade_per_day_never_two(self):
        c, tr = self.trades(lambda ts: 40000.0)
        days = [local_midnight(c[t.entry].ts + 300, PH) for t in tr]
        self.assertEqual(len(days), len(set(days)))

    def test_no_candle_in_the_window_means_no_trade_that_day(self):
        full, a = self.trades(lambda ts: 40000.0)
        day = local_midnight(full[a[5].entry].ts + 300, PH)
        in_window = lambda ts: day + 7 * 3600 <= ts + 300 - 0 <= day + 7 * 3600 + 15 * 60
        _, b = self.trades(lambda ts: 40000.0, skip=in_window)
        self.assertEqual(len(b), len(a) - 1)

    def test_a_crash_hits_the_five_percent_stop_and_does_not_re_enter_that_day(self):
        monday = SUN_2300_UTC + 7 * 86400
        crash = lambda ts: 36000.0 if monday + 9 * 3600 <= ts < monday + 12 * 3600 else 40000.0
        c, tr = self.trades(crash, days=14)
        stopped = [t for t in tr if t.r < -0.9]
        self.assertEqual(len(stopped), 1)
        self.assertAlmostEqual(stopped[0].r, -1.0)
        days = [local_midnight(c[t.entry].ts + 300, PH) for t in tr]
        self.assertEqual(len(days), len(set(days)))

    def test_needs_candles_as_fine_as_the_window(self):
        self.assertIsNone(AllDayLong().compatible(300))
        self.assertIsNone(AllDayLong().compatible(60))
        self.assertIn("5-minute", AllDayLong().compatible(900))

    def test_caveats_mention_the_22_hour_mismatch_and_the_drift(self):
        text = " ".join(AllDayLong().caveats())
        self.assertIn("22 hours 45 minutes", text)
        self.assertIn("drift", text)


class NewsStrategyTests(unittest.TestCase):
    def flat_with_release(self, post_closes, pre=60):
        """Flat gold at 2000 with a release at the candle after `pre` candles, then the given closes."""
        c = [Candle(60 * i, 2000.0, 2000.2, 1999.8, 2000.0) for i in range(pre)]
        prev = 2000.0
        for k, close in enumerate(post_closes):
            c.append(Candle(60 * (pre + k), prev, max(prev, close) + 0.1, min(prev, close) - 0.1, close))
            prev = close
        return c, NewsEvent(60 * pre, "NFP", 3)

    def test_without_a_calendar_it_says_so(self):
        reason = NewsBreakout().bind(Context()).compatible(60)
        self.assertIn("news calendar", reason)
        self.assertIn("5-minute", NewsBreakout().bind(Context(news=(NewsEvent(1, "x"),))).compatible(900))

    def test_joins_only_the_first_close_beyond_the_pre_news_range(self):
        c, ev = self.flat_with_release([2003.0, 2006.0, 2007.0])
        s = NewsBreakout(stop_atr=1.0, reward_risk=1.5, hold_bars=10).bind(Context(news=(ev,)))
        obs = MarketObserver()
        self.assertIsNone(s.signal("XAU", c[:60], obs))                       # before the release
        first = s.signal("XAU", c[:61], obs)
        self.assertIsNotNone(first)
        self.assertEqual(first.side, Side.LONG)
        self.assertEqual(first.exit_ts, 60 * 61 + 10 * 60)
        self.assertIsNone(s.signal("XAU", c[:62], obs))                       # already broke out last candle
        self.assertIsNone(s.signal("XAU", c[:63], obs))

    def test_goes_short_on_a_break_below_the_range(self):
        c, ev = self.flat_with_release([1997.0, 1995.0])
        sig = NewsBreakout().bind(Context(news=(ev,))).signal("XAU", c[:61], MarketObserver())
        self.assertEqual(sig.side, Side.SHORT)
        self.assertGreater(sig.stop, sig.entry)

    def test_ignores_candles_long_after_the_release(self):
        c, ev = self.flat_with_release([2000.0] * 8 + [2010.0])
        self.assertIsNone(NewsBreakout().bind(Context(news=(ev,))).signal("XAU", c, MarketObserver()))

    def test_low_impact_releases_are_ignored(self):
        c, _ = self.flat_with_release([2003.0])
        low = NewsEvent(60 * 60, "minor", 1)
        self.assertIsNone(NewsBreakout().bind(Context(news=(low,))).signal("XAU", c, MarketObserver()))

    def test_fade_waits_for_a_big_overshoot_then_aims_for_the_middle(self):
        c, ev = self.flat_with_release([2000.4, 2012.0])
        s = NewsFade(spike_atr=1.5, stop_atr=1.5).bind(Context(news=(ev,)))
        self.assertIsNone(s.signal("XAU", c[:61], MarketObserver()))          # small move, not a spike
        sig = s.signal("XAU", c[:62], MarketObserver())
        self.assertEqual(sig.side, Side.SHORT)
        self.assertAlmostEqual(sig.target, 2000.0, delta=0.2)                 # middle of the pre-news range

    def test_a_real_post_news_pattern_is_found_and_a_missing_one_is_not(self):
        """With follow-through the strategy earns something before costs; without it, it does not."""
        results = {}
        for follow in (0.0, 0.6):
            candles, events = synthetic_news_market(days=60, follow_through=follow, seed=5)
            s = NewsBreakout(stop_atr=2.0, reward_risk=1.5, hold_bars=15).bind(Context(news=events))
            tr = simulate_detail(candles, lambda h: s.signal("XAU", h, MarketObserver()), 60, None, NO_COSTS)
            results[follow] = sum(t.r for t in tr) / len(tr)
        self.assertGreater(results[0.6], 0.5)
        self.assertLess(results[0.0], 0.4)

    def test_news_costs_can_turn_the_pattern_into_a_loss(self):
        candles, events = synthetic_news_market(days=60, follow_through=0.6, seed=5)
        s = NewsBreakout(stop_atr=1.0, reward_risk=1.5, hold_bars=15).bind(Context(news=events))
        run = lambda costs: simulate_detail(candles, lambda h: s.signal("XAU", h, MarketObserver()), 60, None, costs)
        calm = sum(t.r for t in run(CostModel(spread=0.30, slippage=0.05)))
        busy = sum(t.r for t in run(CostModel(spread=0.30, slippage=0.05).with_news([e.ts for e in events], 5.0, 600)))
        self.assertGreater(calm, 0)
        self.assertLess(busy, calm)


class NewsLoadingTests(unittest.TestCase):
    def write(self, text):
        path = os.path.join(tempfile.mkdtemp(), "news.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_reads_times_names_and_impact_words(self):
        path = self.write("time,event,impact\n2024-01-05 13:30:00,Nonfarm Payrolls,High\n2024-01-04 15:00:00,ISM,Medium\n2024-01-03 09:00:00,minor,Low\n")
        ev = load_news(path)
        self.assertEqual([e.name for e in ev], ["minor", "ISM", "Nonfarm Payrolls"])   # sorted by time
        self.assertEqual([e.impact for e in ev], [1, 2, 3])
        self.assertEqual(len(load_news(path, min_impact=3)), 1)

    def test_numeric_impact_and_missing_impact(self):
        ev = load_news(self.write("timestamp,impact\n1704461400,3\n1704547800,\n"))
        self.assertEqual([e.impact for e in ev], [3, 3])

    def test_needs_a_time_column(self):
        with self.assertRaises(ValueError):
            load_news(self.write("event,impact\nx,high\n"))


class ProfileAndLabTests(unittest.TestCase):
    def test_every_profile_family_exists(self):
        from quorum.agents import FAMILIES
        for prof in PROFILES.values():
            for fam in (prof.grids or {}):
                self.assertIn(fam, FAMILIES)
            for fam, _ in prof.references:
                self.assertIn(fam, FAMILIES)

    def test_the_xau_profile_inflates_news_costs_and_the_us30_profile_charges_financing(self):
        self.assertGreater(PROFILES["xau"].news_multiplier, 1)
        self.assertGreater(PROFILES["us30"].costs.overnight_pct, 0)
        self.assertEqual({f for f, _ in PROFILES["us30"].references}, {"weekly_dip", "all_day_long"})

    def test_lab_skips_families_that_cannot_run_on_this_data(self):
        c = synthetic_market("random", 900, 3, tf_s=900)
        lab = StrategyLab(grids={"all_day_long": {"stop_pct": [0.05]}, "news_breakout": {"stop_atr": [1.0]}}).build(
            "X", c, MarketObserver(), timeframe_s=900)
        self.assertEqual(lab.tested, 0)
        text = " ".join(lab.notes)
        self.assertIn("all_day_long was skipped", text)
        self.assertIn("news_breakout was skipped", text)

    def test_lab_says_when_a_family_made_too_few_trades(self):
        candles, events = synthetic_news_market(days=12, follow_through=0.5, seed=2)
        lab = StrategyLab(grids={"news_breakout": {"stop_atr": [1.0, 2.0], "reward_risk": [1.5], "hold_bars": [10]}},
                          min_trades=15).build("X", candles, MarketObserver(), context=Context(news=events), timeframe_s=60)
        self.assertTrue(any("news_breakout: too few trades" in n for n in lab.notes))

    def test_a_clock_strategy_gets_its_context_from_make_strategy(self):
        s = make_strategy("weekly_dip", {"ma_days": 10}, Context(tz_offset_hours=0))
        self.assertEqual(s.context.tz_offset_hours, 0)
        self.assertEqual(s.ma_days, 10)

    def test_variants_keep_integer_settings_whole_and_the_context(self):
        s = make_strategy("weekly_dip", {"ma_days": 25}, Context(tz_offset_hours=3))
        v = s.variants()
        self.assertEqual({x.ma_days for x in v if x.stop_pct == s.stop_pct}, {20, 30})
        self.assertTrue(all(x.context.tz_offset_hours == 3 for x in v))


class ScorecardTests(unittest.TestCase):
    def setUp(self):
        from dataclasses import replace
        prof = PROFILES["us30"]
        self.cfg = replace(QuorumConfig(), costs=prof.costs, min_trades=prof.min_trades)
        self.ctx = Context(prof.tz_offset_hours, prof.day_start_hour)
        self.q = DataQualityAgent(session_market=True)

    def test_scores_the_rule_as_given_with_return_and_drift_numbers(self):
        c = synthetic_session_market(years=0.8, seed=6, drift_per_year=0.1)
        r = score_reference("US30", c, "all_day_long", {}, self.cfg, self.ctx, self.q, 0.05)
        self.assertGreater(r.trades, 150)
        self.assertEqual(r.params["stop_pct"], 0.05)
        self.assertAlmostEqual(r.equity_return_pct, r.total_r * 1.0, places=6)   # 1% risk per trade
        self.assertGreater(r.buy_hold_pct, -50)
        self.assertTrue(r.exam)

    def test_a_scan_scores_both_us30_rules_next_to_the_search(self):
        c = synthetic_session_market(years=0.6, seed=7)
        res = scan_markets({"US30": c}, AccountState(equity=1000.0), QuorumConfig(), workers=1,
                           profiles={"US30": PROFILES["us30"]})
        a = res.analyses[0]
        self.assertEqual({r.family for r in a.references}, {"weekly_dip", "all_day_long"})
        text = render_report(res, 1000.0)
        self.assertIn("Your rules as given", text)
        self.assertIn("Verdict:", text)
        self.assertIn("Just holding", text)

    def test_session_data_is_not_blocked_in_a_scan(self):
        c = synthetic_session_market(years=0.3, seed=8)
        res = scan_markets({"US30": c}, AccountState(equity=1000.0), QuorumConfig(), workers=1,
                           profiles={"US30": PROFILES["us30"]})
        self.assertFalse(res.analyses[0].decision.reason.startswith("data check failed"))


class CommandLineProfileTests(unittest.TestCase):
    def csv(self, folder, name, candles):
        path = os.path.join(folder, name + ".csv")
        with open(path, "w") as f:
            f.write("time,open,high,low,close\n")
            for c in candles:
                f.write(f"{c.ts},{c.open},{c.high},{c.low},{c.close}\n")
        return path

    def test_an_unknown_profile_is_refused(self):
        folder = tempfile.mkdtemp()
        p = self.csv(folder, "A", synthetic_candles(100))
        with self.assertRaises(SystemExit):
            cli_main(["scan", p + "=bogus"])

    def test_profiles_and_news_are_accepted(self):
        folder = tempfile.mkdtemp()
        candles, events = synthetic_news_market(days=3, follow_through=0.3, seed=1)
        p = self.csv(folder, "XAUUSD", candles)
        news = os.path.join(folder, "news.csv")
        with open(news, "w") as f:
            f.write("time,event,impact\n" + "\n".join(f"{e.ts},{e.name},high" for e in events) + "\n")
        out = os.path.join(folder, "r.md")
        self.assertEqual(cli_main(["scan", p + "=xau", "--news", news, "--workers", "1", "--out", out]), 0)
        with open(out) as f:
            text = f.read()
        self.assertIn("XAUUSD", text)
        self.assertIn("Profile: xau", text)


class MonitorPersistenceTests(unittest.TestCase):
    def test_save_and_load_keeps_settings_and_trades(self):
        m = PaperMonitor(expected_r=0.4, tolerance_r=0.2, min_trades=3, window=7)
        m.record(paper(100.2, 101.5))
        m.record(paper(100.0, 99.0, Side.LONG))
        path = os.path.join(tempfile.mkdtemp(), "monitor.json")
        m.save(path)
        back = PaperMonitor.load(path)
        self.assertEqual(back.trades, m.trades)
        self.assertEqual((back.expected_r, back.tolerance_r, back.min_trades, back.window), (0.4, 0.2, 3, 7))
        self.assertEqual(back.report(), m.report())


class SystemEvalTests(unittest.TestCase):
    def test_all_system_evals_pass(self):
        failed = [r for r in run_evals() if not r.passed]
        self.assertEqual(failed, [], [f"{r.name}: {r.detail}" for r in failed])


if __name__ == "__main__":
    unittest.main()
