# Quorum

A multi-agent trading assistant. A trade idea only goes forward when the agents that can say no all agree.

**Status: advisory and paper trading only.** Nothing in this repo places real orders. The Strategy Analyst still holds a placeholder rule, not a real strategy.

## How it works

The **Coordinator** (the AI Research Coordinator) is the only component that talks to the agents. For each decision it:

1. checks that the **Paper Monitor** has not halted trading
2. asks the **Data Quality Agent** to check the candles (and compare them with a second feed if given)
3. asks the **Market Observer** to label the regime, volatility, and ATR
4. asks the **Strategy Analyst** for a trade idea (or none)
5. asks the **Validator**, the **Risk Manager**, and the **Skeptic** to vote
6. logs approved trades in the **Trade Journal**

The idea is approved only if every voting agent approves. Any one of them can veto. One `CostModel` (spread, slippage, fees) is shared by all agents so they judge with the same assumptions.

| Agent | Job |
|---|---|
| Data Quality | broken candles, gaps, off-grid timestamps, bad ticks that reverse, frozen feeds, mismatch against a second feed |
| Market Observer | regime, volatility state, ATR |
| Strategy Analyst | turns a snapshot into an entry, stop, and target; can produce nearby settings for robustness checks |
| Research & Experiment Analyst | parameter experiments on older data only, flags overfitting, says how many combinations were profitable, recommends but never changes live settings |
| Validator | holdout replay after costs, walk-forward windows, nearby-settings robustness |
| Skeptic | edge stable across halves, not one lucky trade, bootstrap luck test, max drawdown |
| Risk Manager | position size, daily loss, drawdown, losing-streak cooldown, cost-to-risk cap, optional exposure cap, kill switch (hard-coded rules) |
| Paper Monitor | compares recent paper fills to the backtest, flags drift, can halt trading |
| Trade Journal | logs Quorum and manual trades; finds patterns by side and regime, plus ignoring Quorum, moved stops, overtrading, revenge trading |

The Research Analyst only uses the older 60% of the history. The recent 40% is the holdout the Validator and Skeptic judge on, so nobody tunes on it. The backtest harness only ever shows a strategy candles up to the current bar.

## System tests

`python -m quorum.evals` runs checks on the team as a whole: same input gives the same decision, bad data blocks trades, the kill switch always blocks, a veto always wins, size never exceeds the risk limit, costs lower results, and reported numbers match a recalculation. These must keep passing when an LLM layer is added.

## Try it

```
python -m quorum                      # demo on synthetic candles
python -m quorum.evals                # system tests
python -m unittest discover -s tests  # unit tests
```

No dependencies beyond Python 3.10+. Today every agent is plain Python with fixed rules. There is no LLM layer yet.

## Next steps

- Port the real Step Index strategy into `StrategyAnalyst` (the current rule is a placeholder)
- Feed real Step Index candles and a real cost model
- Feed real paper fills into the Paper Monitor
- Add an LLM layer on top for explanations, always calling the code for numbers
