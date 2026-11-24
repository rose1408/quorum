# Quorum

A multi-agent trading assistant. A trade idea only goes forward when the agents that can say no all agree.

**Status: advisory and paper trading only.** Nothing in this repo places real orders.

## How it works

The **Coordinator** is the only component that talks to the agents. For each decision it:

1. asks the **Market Observer** to check the data and label the regime
2. asks the **Strategy Analyst** for a trade idea (or none)
3. asks the **Validator** to replay the idea over holdout data and try to break it
4. asks the **Risk Manager** for position size and a limit check

The idea is approved only if both the Validator and the Risk Manager approve. Either can veto.

| Agent | Job |
|---|---|
| Market Observer | data integrity checks, regime, volatility, ATR |
| Strategy Analyst | turns a snapshot into an entry, stop, and target |
| Validator | holdout replay in R multiples, pessimistic about stops |
| Risk Manager | position size, daily loss limit, kill switch, reward/risk floor |

## Try it

```
python -m quorum                      # demo on synthetic candles
python -m unittest discover -s tests  # run tests
```

No dependencies beyond Python 3.10+.

## Next steps

- Port the real Step Index strategy into `StrategyAnalyst` (the current rule is a placeholder)
- Feed real Step Index candles instead of synthetic ones
- Add the Research and Experiment Analyst, then the Skeptic and Trade Journal agents
- Keep the Validator's holdout data out of any strategy tuning
