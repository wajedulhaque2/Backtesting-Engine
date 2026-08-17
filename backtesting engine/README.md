# Backtesting Engine — Implementation Directory

For the project overview, verified results and portfolio walkthrough, see the [root README](../README.md).

This directory contains the reusable implementation and research artefacts behind the project.

## Key files

- `main.py` — complete executable research workflow.
- `main.ipynb` — notebook version of the end-to-end analysis.
- `backtesting_engine_complete_theory_guide.ipynb` — detailed backtesting theory and methodology.
- `backtester.py` — portfolio simulation, execution assumptions and metrics.
- `strategies.py` — Buy & Hold, Moving Average, Momentum and Mean Reversion rules.
- `research.py` — strategy-suite execution and parameter optimisation.
- `market_data.py` — market-data download/cache handling.
- `tests/` — automated checks for core engine behaviour.
- `outputs/` — saved parameter searches, summaries, trade logs and charts used in the root README.

## Run

```bash
pip install -r requirements.txt
python main.py
```

The default saved research configuration uses SPY, a 2010–2026 sample and a 2019 train/test split. Parameters are selected on the training period and evaluated out of sample.
