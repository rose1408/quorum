# Quorum

A multi-agent trading research tool. It studies several markets at once, searches for strategies that work, tries hard to break them, and only then says whether there is a trade idea.

**Status: advisory and paper trading only.** Nothing in this repo places real orders. Every agent is plain Python with fixed rules. There is no LLM layer yet.

## Quick start

```
python -m quorum demo                              # three synthetic markets with known behavior
python -m quorum scan A.csv B.csv C.csv \
    --spread 0.02 --slippage 0.01 --equity 1000 \
    --out report.md                                # your own markets, analyzed at the same time
python -m quorum.evals                             # system tests
python -m unittest discover -s tests              # unit tests
```

CSV files need time, open, high, low, close columns (comma, tab, or semicolon separated; MT5-style exports with separate date and time columns work). The file name becomes the market name. No dependencies beyond Python 3.10+.

## What a scan does

For each market, in parallel:

1. **Data Quality** checks the candles (and compares them with a second feed if you give one).
2. **Flow Analyst** reads trend strength, momentum, and volatility, and runs a *structure test* (below).
3. **Strategy Lab** searches three strategy families (trend, breakout, mean reversion) and their settings, using only the older 60% of the history. It keeps a candidate only if it is profitable after costs on both its training and validation slices.
4. **Holdout exam**: the Validator and Skeptic test the chosen strategy on the recent 40% that the search never saw.
5. If a setup exists right now, the **Coordinator** collects the Validator, Risk Manager, and Skeptic votes. All must approve.

Then across markets: **Flow Analyst** measures correlation and lead-lag, and the **Portfolio Risk Manager** picks which approved ideas to take given a total-risk cap, a position limit, and a rule that two ideas betting the same way on linked markets count as one. The **report** is the single voice the team speaks with.

## The agents

| Agent | Job |
|---|---|
| Data Quality | broken candles, gaps, off-grid timestamps, bad ticks that reverse, frozen feeds, mismatch against a second feed |
| Market Observer | regime, volatility state, ATR, averages, recent channel, stretch from the mean |
| Flow Analyst | trend, momentum, volatility per market; structure test (trending, mean-reverting, random); correlation, lead-lag, ranking across markets |
| Strategy Lab | searches strategy families and settings, picks by the weaker of training and validation results, never touches the holdout |
| Research & Experiment Analyst | parameter experiments for one strategy family on older data only |
| Validator | holdout replay after costs, walk-forward windows, nearby-settings robustness |
| Skeptic | edge stable across halves, not one lucky trade, bootstrap luck test (stricter when the market looks random) |
| Risk Manager | position size, daily loss, drawdown, losing-streak cooldown, cost cap, kill switch (hard-coded rules) |
| Portfolio Risk Manager | total risk, position count, and correlated-bet limits across markets |
| Paper Monitor | compares recent paper fills to the backtest, flags drift, can halt trading |
| Trade Journal | logs Quorum and manual trades; spots ignoring Quorum, moved stops, overtrading, revenge trading |
| Coordinator | the only component that talks to the agents |

## The structure test, and why it matters

If a market's moves have no memory (a random walk), no trend or mean-reversion rule can have a real edge there, and anything a backtest finds is probably noise. The Flow Analyst screens for this with a lag-1 autocorrelation test and a variance-ratio test. It is a rough screen, not proof, and weak real edges can slip under it.

I believe Deriv generates its synthetic indices (such as Step Index) with a random number generator. If so, Quorum should usually report nothing tradable there, and that is the correct answer. Check Deriv's own documentation.

## What was measured

With costs on, 3 synthetic market types, many random seeds (synthetic data only, so treat these as a sanity check, not a promise about real markets):

| Market type (1,500 candles) | Passed the holdout exam |
|---|---|
| Pure random, 40 markets | 0 of 40 with the current settings (2 of 40 with a looser 20% luck limit) |
| Mean-reverting, 20 markets | 10 of 20 |
| Trending (breakout), 20 markets | 2 of 20 |
| Trending (breakout), 16 markets at 3,000 candles | 6 of 16 |

The tool is deliberately strict, so it misses real edges when history is short. Feed it several thousand candles per market.

## Limits

- Candles only. "Flow" means price flow, not order flow, which needs tick or volume data.
- Costs default to zero, which is optimistic. Set them to what your broker charges.
- The strategy families are generic textbook rules. Your own strategy gets added as another family.
- The search tests many combinations per market, so the winner always looks better than it is. The holdout exam is the real test, and it can still pass by luck.
- A past edge can stop working. Paper trade first and use the Paper Monitor.

## Next steps

- Add the Step Index strategy as a strategy family
- Feed real candles and a real cost model
- Feed real paper fills into the Paper Monitor and persist its state between scans
- Add an LLM layer on top for explanations, always calling the code for numbers; the system evals must keep passing
