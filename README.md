# Quorum

A multi-agent trading research tool. It studies several markets at once, scores your own rules exactly as you wrote them, searches for something better, tries hard to break whatever it finds, and only then says whether there is a trade idea.

**Status: advisory and paper trading only.** Nothing in this repo places real orders. Every agent is plain Python with fixed rules. There is no LLM layer yet.

## Quick start

```
python -m quorum demo                                  # synthetic stand-ins for Step Index, XAUUSD, US30
python -m quorum scan STEP.csv=step XAUUSD.csv=xau US30.csv=us30 \
    --news news.csv --equity 1000 --out report.md      # your own candles, analyzed at the same time
python -m quorum scan A.csv B.csv C.csv --spread 0.02  # any markets, generic search
python -m quorum scan US30.csv=us30 --data-utc-offset 2   # file times are in UTC+2 (broker server time)
python -m quorum.evals                                 # system tests
python -m unittest discover -s tests                   # unit tests
```

Candle files need time, open, high, low, close columns (comma, tab, or semicolon separated; MT5-style exports with separate date and time columns work). The file name becomes the market name. `=PROFILE` picks a market profile: `step`, `xau`, `us30`, or `generic`. A news calendar is a CSV with a time column and optional event and impact columns (low/medium/high or 1 to 3). Times are read as UTC unless you pass `--data-utc-offset` (candle files) or `--news-utc-offset` (calendar). Tested on Python 3.12 only; the code avoids 3.12-only syntax and declares 3.10+. No dependencies.

## Preparing your data

Read this before the first real scan. Most wrong results come from the data, not the strategies.

- **Timezone.** MT5 and most brokers export in *server time*, often UTC+2 or UTC+3, not UTC. Your Philippine-time rules and the news calendar need true UTC. If the file is in UTC+2, pass `--data-utc-offset 2`. For markets with a daily break (US30, XAUUSD) Quorum checks where the break falls and prints a **Check this first** line at the top of that market's section if it looks shifted, with the offset to try. Server clocks that follow daylight saving move by an hour twice a year, and a single fixed offset cannot fix that.
- **How much history.** Several thousand candles per market is the floor. Weekly dip needs about two years of 5-minute candles, all day long about one year, and the news strategies many months of 1-minute candles. With less, Quorum says "too few trades to judge" instead of guessing. A scan of 119,000 one-minute candles took about 46 seconds and 288,000 five-minute candles about 95 seconds on the test machine.
- **Clean files.** One header row, no merged cells. Blank prices are refused with the row number. Dates like 02/01/2024 are refused as ambiguous; use 2024-01-02 or 02.01.2024. Prices with a decimal comma work in semicolon or tab separated files.
- **News calendar.** Release times, ideally only the big ones, with an impact column. Quorum cannot fetch a calendar for you.
- **Sizes.** Position sizes are in units of the instrument. The profiles carry placeholder smallest-size limits; if the right size for a stop is below your broker's minimum, Quorum says the account is too small for that stop instead of rounding up and risking more than planned.

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

## Adding your own strategy

A strategy is a class in `quorum/agents/` with a `signal(symbol, history, observer)` method. The one rule that matters: **compute everything only from `history`**, the candles it is handed up to the current one. Do not precompute indicators over a whole file, keep your own copy of the data, or read anything from the future. Quorum's harness only shows a strategy the candles up to now, and a system eval replaces the future with made-up prices and checks that no strategy changes its mind, but nothing can police a strategy that quietly reads data it was not handed. This matters most when porting a TradingView or Pine strategy, where indicators are often computed over the whole chart.

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

## Trust checks

`python -m quorum.evals` runs 15 checks on the whole team, including: same input gives the same decision, bad data blocks trades, a veto always wins, size never exceeds the risk limit, a gap through the stop is booked at the gap price, being long in a rising market is not mistaken for an edge, closures do not block session markets, no strategy changes its answer when the future is replaced, and the strategy search never sees the holdout. A deliberate holdout leak was confirmed to make that last check fail.

## Limits

- Candles only. "Flow" means price flow, not order flow, which needs tick or volume data.
- Candles hide what happens inside a candle, so fast whipsaws, especially at news, are under-counted. The news strategies model a confirmed move after a release, not an order placed before it.
- The cost numbers are placeholders until you replace them. Gold spreads can widen many times over at big releases.
- The generic strategy families are textbook rules. A rule you want tested should be added as another family.
- The search tests many combinations per market, so the winner always looks better than it is. The holdout exam is the real test, and it can still pass by luck.
- A past edge can stop working. Paper trade first and use the Paper Monitor.
- The timezone check assumes the daily break starts at 05:00 or 06:00 Philippine time. That is a guess about your broker's schedule, so it only warns and never blocks.
- Contract size, minimum lot, and swap are placeholders. Costs also assume fills at the signal candle's close; a real market order can fill worse.

## Next steps

- Add your Step Index strategy as a strategy family
- Run on real candles with real costs, and a real news calendar for gold
- Feed real paper fills into the Paper Monitor
- Add an LLM layer on top for explanations, always calling the code for numbers; the system evals must keep passing
