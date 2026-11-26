# Quorum

A multi-agent trading research tool. It studies several markets at once, scores your own rules exactly as you wrote them, searches for something better, tries hard to break whatever it finds, and only then says whether there is a trade idea.

**Status: advisory and paper trading only.** Nothing in this repo places real orders. Every agent is plain Python with fixed rules. There is no LLM layer yet.

## Quick start

```
python -m quorum demo                                  # synthetic stand-ins for Step Index, XAUUSD, US30
python -m quorum scan STEP.csv=step XAUUSD.csv=xau US30.csv=us30 \
    --news news.csv --equity 1000 --out report.md      # your own candles, analyzed at the same time
python -m quorum scan A.csv B.csv C.csv --spread 0.02  # any markets, generic search
python -m quorum.evals                                 # system tests
python -m unittest discover -s tests                   # unit tests
```

Candle files need time, open, high, low, close columns (comma, tab, or semicolon separated; MT5-style exports with separate date and time columns work). The file name becomes the market name. `=PROFILE` picks a market profile: `step`, `xau`, `us30`, or `generic`. A news calendar is a CSV with a time column and optional event and impact columns (low/medium/high or 1 to 3). Times are read as UTC. No dependencies beyond Python 3.10+.

## Market profiles

A profile sets a market's timeframe, clock, costs, and which strategies to search. **The cost numbers in `quorum/profiles.py` are placeholder guesses. Replace them with what your broker charges**, including the overnight swap for index CFDs.

| Profile | Market | Notes |
|---|---|---|
| `step` | Step Index, 5-minute | trades around the clock; generic strategy families |
| `xau` | XAUUSD, 1-minute, scalping | tight-stop scalp settings with time stops; two news strategies; spread and slippage multiplied around news releases |
| `us30` | US30, 5-minute | your Monday-dip and all-day-long rules, plus the generic families; overnight financing charged |

### Your US30 rules, as implemented

Times are fixed Philippine time (UTC+8, no daylight saving). A candle counts as "now" when it closes.

- **Weekly dip (`weekly_dip`)**: from 7:04am Monday until the next trading day starts at 7:00am Tuesday, watch price against the average of the last 25 completed trading days. The first candle that closes below it is a market buy. If price never closes below it, no trade that week. Stop 5% below entry, no take-profit, close Wednesday 4:55am. One trade per week; a stopped-out trade is not re-entered.
- **All day long (`all_day_long`)**: the first candle closing in the 7:05 to 7:10am window is a market buy. No candle in the window means no trade that day. Stop 5% below entry, no take-profit, close at 5:50am the next day. Needs 5-minute or finer candles.

Things worth knowing: 7:05am to 5:50am is 22 hours 45 minutes, not 22 hours. The "daily" average is built from days that start at 7:00am on that clock, so it can differ a little from your chart's daily candles. The US session moves one hour twice a year with daylight saving, but these times are fixed.

Both rules are always scored exactly as given, next to whatever the search finds, with their return compared against simply holding the index.

## What a scan does

For each market, in parallel:

1. **Data Quality** checks the candles. Session markets (gold, indices) get nightly and weekend closures treated as normal, while missing candles inside sessions still count.
2. **Flow Analyst** reads trend, momentum, and volatility and runs the *structure test* (below).
3. **Your rules as given** (if the profile has any) are replayed over the whole history and examined on the holdout.
4. **Strategy Lab** searches the profile's strategy families and settings on the older 60% of the history, with costs. A candidate must be profitable on both its training and validation slices.
5. **Holdout exam**: the Validator and Skeptic test the chosen strategy on the recent 40% that the search never saw.
6. If a setup exists right now, the **Coordinator** collects the Validator, Risk Manager, and Skeptic votes. All must approve.

Then across markets: correlation, lead-lag, and a **Portfolio Risk Manager** that applies a total-risk cap, a position limit, and a rule that two ideas betting the same way on linked markets count as one.

## The agents

| Agent | Job |
|---|---|
| Data Quality | broken candles, missing candles, off-grid timestamps, bad ticks that reverse, frozen feeds, mismatch against a second feed; session-aware |
| Market Observer | regime, volatility state, ATR, averages, recent channel, stretch from the mean (cached, so each candle is analyzed once) |
| Flow Analyst | trend, momentum, volatility; structure test (trending, mean-reverting, random); correlation, lead-lag, ranking across markets |
| Strategy families | trend, breakout, mean reversion (with optional time stop), weekly dip, all day long, news breakout, news fade |
| Strategy Lab | searches families and settings, picks by the weaker of training and validation results, never touches the holdout, says when a family made too few trades to judge |
| Research & Experiment Analyst | parameter experiments for one strategy family on older data only |
| Validator | holdout replay after costs, walk-forward windows, nearby-settings robustness, and a drift check |
| Skeptic | edge stable across halves, not one lucky trade, bootstrap luck test judged after removing market drift |
| Risk Manager | position size, daily loss, drawdown, losing-streak cooldown, cost cap, kill switch; handles trades with no take-profit |
| Portfolio Risk Manager | total risk, position count, and correlated-bet limits across markets |
| Paper Monitor | compares recent paper fills to the backtest, flags drift, can halt trading; saves and loads its state |
| Trade Journal | logs Quorum and manual trades; spots ignoring Quorum, moved stops, overtrading, revenge trading |
| Coordinator | the only component that talks to the agents |

## How trades are modeled

- Entries fill at the close of the signal candle. Stops are checked before targets inside a candle (the pessimistic choice).
- A candle that opens beyond the stop (a gap) fills at its open, so weekend gaps cost more than 1R.
- A time exit fills at the open of the first candle at or after the exit time.
- Costs: spread, slippage, fees, overnight financing for time held, and a spread and slippage multiplier around news releases.
- **Drift check.** A long-only rule in a rising market looks good just for being long. The Validator and Skeptic subtract the market's own average move for every candle a trade was open and judge what is left.

## The structure test, and why it matters

If a market's moves have no memory (a random walk), no trend or mean-reversion rule can have a real edge there, and anything a backtest finds is probably noise. The Flow Analyst screens for this with a lag-1 autocorrelation test and a variance-ratio test. It is a rough screen, not proof, and weak real edges can slip under it. When it says "random", the Skeptic demands much stronger proof.

I believe Deriv generates its synthetic indices (such as Step Index) with a random number generator. If so, Quorum should usually report nothing tradable there, and that is the correct answer. Check Deriv's own documentation.

## What was measured

Synthetic data only, costs on, so treat these as sanity checks and not promises about real markets.

| Market type | Passed the holdout exam |
|---|---|
| Pure random, 1,500 candles, 40 markets | 0 of 40 (a looser 20% luck limit let 2 of 40 through) |
| Mean-reverting, 1,500 candles, 20 markets | 10 of 20 |
| Trending (breakout), 3,000 candles, 16 markets | 4 of 16 |

The tool is deliberately strict, so it misses real edges when history is short. How much history is enough depends on how often a strategy trades:

- Weekly dip makes about one trade every one or two weeks, so it needs roughly two or more years of candles to reach 20 holdout trades.
- All day long makes about one trade a day, so about one year.
- News strategies make about one trade per release, so they need many months of 1-minute data.

## Limits

- Candles only. "Flow" means price flow, not order flow, which needs tick or volume data.
- Candles hide what happens inside a candle, so fast whipsaws, especially at news, are under-counted. The news strategies model a confirmed move after a release, not an order placed before it.
- The cost numbers are placeholders until you replace them. Gold spreads can widen many times over at big releases.
- The generic strategy families are textbook rules. A rule you want tested should be added as another family.
- The search tests many combinations per market, so the winner always looks better than it is. The holdout exam is the real test, and it can still pass by luck.
- A past edge can stop working. Paper trade first and use the Paper Monitor.

## Next steps

- Add your Step Index strategy as a strategy family
- Run on real candles with real costs, and a real news calendar for gold
- Feed real paper fills into the Paper Monitor
- Add an LLM layer on top for explanations, always calling the code for numbers; the system evals must keep passing
