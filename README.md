# Quorum

A multi-agent trading assistant. A trade idea only goes forward when the agents that can say no all agree.

**Status: advisory and paper trading only.** Nothing in this repo places real orders.

## How it works

The **Coordinator** (the AI Research Coordinator) is the only component that talks to the agents. For each decision it:

1. checks that the **Paper Monitor** has not halted trading
2. asks the **Market Observer** to check the data and label the regime
3. asks the **Strategy Analyst** for a trade idea (or none)
4. asks the **Validator**, the **Risk Manager**, and the **Skeptic** to vote
5. logs approved trades in the **Trade Journal**

The idea is approved only if every voting agent approves. Any one of them can veto.

| Agent | Job |
|---|---|
| Market Observer | data integrity checks, regime, volatility, ATR |
| Strategy Analyst | turns a snapshot into an entry, stop, and target |
| Research & Experiment Analyst | parameter experiments on older data only, flags overfitting, recommends but never changes live settings |
| Validator | holdout replay in R multiples, pessimistic about stops |
| Skeptic | asks if the edge is stable and could be luck (half-split check, bootstrap) |
| Risk Manager | position size, daily loss limit, kill switch, reward/risk floor |
| Paper Monitor | compares paper fills to backtest assumptions, flags drift, can halt trading |
| Trade Journal | logs approved trades, finds patterns by side and regime, tracks losing streaks |

The Market Observer also does the data-integrity and regime-detection jobs for now. They can be split out later.

The Research Analyst only uses the older 60% of the history. The recent 40% is the holdout the Validator and Skeptic judge on, so nobody tunes on it.

## Try it

```
python -m quorum                      # demo on synthetic candles
python -m unittest discover -s tests  # run tests
```

No dependencies beyond Python 3.10+.

## Next steps

- Port the real Step Index strategy into `StrategyAnalyst` (the current rule is a placeholder)
- Feed real Step Index candles instead of synthetic ones
- Feed real paper fills into the Paper Monitor and set its expected R from the Validator's result
