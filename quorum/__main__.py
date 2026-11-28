from __future__ import annotations

import argparse
import sys

from .agents import AccountState, RiskLimits
from .backtest import CostModel
from .config import DEMO_COSTS, QuorumConfig
from .data import (load_markets, load_news, synthetic_market, synthetic_news_market,
                   synthetic_session_market)
from .profiles import PROFILES
from .report import render_report
from .scan import scan_markets


def _demo_target_markets():
    """Synthetic stand-ins shaped like the three target markets. Real candles replace these."""
    xau, news = synthetic_news_market(days=45, follow_through=0.5)
    markets = {
        "STEP-M5": synthetic_market("random", 9000, 11, tf_s=300, start_ts=1704063600),
        "XAU-M1": xau,
        "US30-M5": synthetic_session_market(years=2.0),
    }
    return markets, {"STEP-M5": PROFILES["step"], "XAU-M1": PROFILES["xau"], "US30-M5": PROFILES["us30"]}, news


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="quorum", description="Quorum: advisory multi-market research. No orders are placed.")
    sub = parser.add_subparsers(dest="cmd")
    demo = sub.add_parser("demo", help="scan synthetic stand-ins for Step Index, XAUUSD and US30")
    scan = sub.add_parser("scan", help="scan your own candle files (CSV with time, open, high, low, close)")
    scan.add_argument("files", nargs="+",
                      help="one CSV per market, optionally FILE=PROFILE where PROFILE is " + ", ".join(PROFILES))
    scan.add_argument("--news", help="news calendar CSV (time, event, impact), used by the xau profile")
    scan.add_argument("--data-utc-offset", type=float, default=0.0,
                      help="timezone of the times in your candle files in hours from UTC (2 means UTC+2, common for broker servers)")
    scan.add_argument("--news-utc-offset", type=float, default=0.0,
                      help="timezone of the times in the news calendar in hours from UTC")
    for p in (scan, demo):
        p.add_argument("--equity", type=float, default=1000.0)
        p.add_argument("--workers", type=int, default=3, help="markets analyzed at the same time")
        p.add_argument("--out", help="write the report to this file as Markdown")
    scan.add_argument("--spread", type=float, default=0.0, help="spread in price units; only for markets with no profile")
    scan.add_argument("--slippage", type=float, default=0.0, help="slippage per side; only for markets with no profile")
    scan.add_argument("--fee", type=float, default=0.0, help="fee per round trip; only for markets with no profile")
    scan.add_argument("--risk", type=float, default=0.01, help="share of equity risked per trade")
    args = parser.parse_args(argv)

    profiles, news = {}, ()
    if args.cmd == "scan":
        paths = []
        chosen = {}
        for entry in args.files:
            path, _, prof = entry.rpartition("=") if "=" in entry else (entry, "", "")
            if prof and prof not in PROFILES:
                parser.error(f"unknown profile {prof!r}; choose from {', '.join(PROFILES)}")
            paths.append(path)
            chosen[path] = prof
        markets = load_markets(paths, args.data_utc_offset)
        for name, c in markets.items():
            if len(c) < 500:
                print(f"warning: {name} has only {len(c)} candles, too few for a trustworthy result", file=sys.stderr)
        for path, name in zip(paths, markets):
            if chosen[path]:
                profiles[name] = PROFILES[chosen[path]]
        if args.news:
            news = load_news(args.news, utc_offset_hours=args.news_utc_offset)
        config = QuorumConfig(costs=CostModel(args.spread, args.slippage, args.fee),
                              risk=RiskLimits(risk_per_trade=args.risk))
    else:
        markets, profiles, news = _demo_target_markets()
        config = QuorumConfig(costs=DEMO_COSTS)

    result = scan_markets(markets, AccountState(equity=args.equity), config, workers=args.workers,
                          profiles=profiles, news=news,
                          progress=lambda message: print(message, file=sys.stderr, flush=True))
    text = render_report(result, args.equity)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"report written to {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
