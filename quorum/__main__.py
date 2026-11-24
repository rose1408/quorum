from __future__ import annotations

import argparse
import sys

from .agents import AccountState, PortfolioLimits, RiskLimits
from .backtest import CostModel
from .config import DEMO_COSTS, QuorumConfig
from .data import load_markets, synthetic_markets
from .report import render_report
from .scan import scan_markets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="quorum", description="Quorum: advisory multi-market research. No orders are placed.")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("demo", help="scan three synthetic markets with known behavior")
    scan = sub.add_parser("scan", help="scan your own candle files (CSV with time, open, high, low, close)")
    scan.add_argument("files", nargs="+", help="one CSV per market; the file name becomes the market name")
    for p in (scan, sub.choices["demo"]):
        p.add_argument("--equity", type=float, default=1000.0)
        p.add_argument("--workers", type=int, default=3, help="markets analyzed at the same time")
        p.add_argument("--out", help="write the report to this file as Markdown")
    scan.add_argument("--spread", type=float, default=0.0, help="spread in price units, per round trip")
    scan.add_argument("--slippage", type=float, default=0.0, help="slippage in price units, per side")
    scan.add_argument("--fee", type=float, default=0.0, help="fee in price units, per round trip")
    scan.add_argument("--risk", type=float, default=0.01, help="share of equity risked per trade")
    args = parser.parse_args(argv)

    if args.cmd == "scan":
        markets = load_markets(args.files)
        config = QuorumConfig(costs=CostModel(args.spread, args.slippage, args.fee),
                              risk=RiskLimits(risk_per_trade=args.risk))
    else:
        markets = synthetic_markets(n=3000)
        config = QuorumConfig(costs=DEMO_COSTS)

    result = scan_markets(markets, AccountState(equity=args.equity), config, workers=args.workers)
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
