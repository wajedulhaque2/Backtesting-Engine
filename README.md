# Systematic Trading Backtesting Engine

A reusable Python framework for researching rule-based trading strategies with **transaction costs, delayed execution, trade-level analytics, parameter selection and out-of-sample testing**.

**Python · pandas · NumPy · yfinance · matplotlib · pytest · systematic trading research**

![Out-of-sample portfolio comparison](backtesting%20engine/outputs/testing_portfolio_comparison.png)

## Project objective

The project is designed to answer a more useful question than “did this strategy make money historically?”:

> **Does the strategy still look reasonable after realistic execution assumptions, parameter selection and a genuine out-of-sample test?**

The engine separates market data, strategy rules, backtesting logic and research/optimisation code into reusable modules. It compares four approaches under the same assumptions:

- Buy and Hold;
- Moving-Average Crossover;
- Momentum;
- Mean Reversion.

## Research setup

The saved configuration in `main.py` uses:

| Setting | Value |
|---|---|
| Instrument | SPY |
| Full sample | 01 Jan 2010 – 01 Jan 2026 |
| Train/test split | 01 Jan 2019 |
| Initial capital | $10,000 |
| Transaction cost | 0.10% per position change |
| Execution delay | 1 trading period |
| Trading periods/year | 252 |

Parameters are selected on the **training period using Sharpe ratio** and are then evaluated on the later testing period. Full-period results are retained only as a descriptive overview.

## Selected strategy parameters

The saved optimisation output selected:

```text
Moving Average: 50-day / 150-day
Momentum:       126-day lookback
Mean Reversion: 20-day lookback
                entry z-score = 1.5
                exit z-score  = 0.5
```

These parameters come from the repository's saved `selected_parameters.json`, rather than being chosen from the final test results.

## Out-of-sample results

The testing output stored in the repository reports:

| Strategy | Total return | Annualised return | Sharpe | Max drawdown | Completed trades |
|---|---:|---:|---:|---:|---:|
| Buy and Hold | **202.77%** | **17.16%** | **0.901** | -33.72% | 0 |
| Moving Average | 89.24% | 9.55% | 0.661 | -33.72% | 4 |
| Momentum | 67.99% | 7.70% | 0.647 | **-23.95%** | 27 |
| Mean Reversion | 11.41% | 1.56% | 0.177 | -30.59% | 33 |

The important portfolio lesson is not that one trading rule “wins”. In this saved SPY test, **Buy and Hold produced the strongest return and Sharpe ratio**, while the momentum strategy experienced the smallest maximum drawdown of the four. The comparison shows why strategy evaluation should include risk, turnover and trading costs rather than return alone.

![Out-of-sample drawdown comparison](backtesting%20engine/outputs/testing_drawdown_comparison.png)

## Backtesting safeguards

### Delayed execution

Signals are shifted before they affect positions. This reduces look-ahead bias by preventing the strategy from trading on information from the same bar that generated the signal.

### Transaction costs

The engine charges costs when portfolio exposure changes, so frequent trading is explicitly penalised.

### Train/test separation

Strategy parameters are selected using the pre-2019 training period. The post-split section is treated as the main evidence rather than optimising on the full historical sample.

### Trade-level analysis

The engine records individual trades and reports metrics beyond portfolio returns, including:

- completed trades;
- trade win rate;
- average trade return;
- profit factor;
- market exposure;
- daily win rate.

## Performance metrics

Each strategy is evaluated with a consistent set of portfolio statistics:

- final portfolio value;
- total and annualised return;
- annualised volatility;
- Sharpe ratio;
- Sortino ratio;
- maximum drawdown;
- daily win rate;
- market exposure;
- turnover/orders;
- trade-level performance.

## Project structure

```text
Backtesting-Engine/
├── README.md
└── backtesting engine/
    ├── backtester.py
    ├── market_data.py
    ├── strategies.py
    ├── research.py
    ├── main.py
    ├── main.ipynb
    ├── backtesting_engine_complete_theory_guide.ipynb
    ├── requirements.txt
    ├── tests/
    │   ├── conftest.py
    │   └── test_backtester.py
    └── outputs/
        ├── selected_parameters.json
        ├── testing_summary.csv
        ├── full_summary.csv
        ├── parameter-search outputs
        ├── daily strategy results
        ├── trade logs
        └── saved charts
```

## Start here

For a portfolio review, I recommend this order:

1. **`README.md`** — project purpose, methodology and results.
2. **`backtesting engine/main.ipynb`** — end-to-end research workflow.
3. **`backtesting engine/backtesting_engine_complete_theory_guide.ipynb`** — detailed explanation of the backtesting concepts and common errors.
4. **`backtesting engine/backtester.py`** — reusable execution and portfolio engine.
5. **`backtesting engine/tests/`** — automated checks for core backtesting behaviour.

## How to run

```bash
cd "backtesting engine"
pip install -r requirements.txt
python main.py
```

`main.py` downloads/loads market data, performs the training-period parameter searches, runs the selected strategies on full/training/testing samples, exports CSV results and saves comparison charts to `outputs/`.

To experiment with another instrument or date range, edit the clearly marked `SETTINGS` section at the top of `main.py`.

## Limitations

- This is a research backtester, not a live execution or portfolio-management system.
- Transaction costs are modelled as a fixed proportional rate; market impact, bid/ask spreads, taxes and liquidity constraints are not modelled separately.
- The saved results are for SPY and one train/test split; performance on other instruments or periods can differ substantially.
- Parameter search can still overfit the training period, so the out-of-sample section is more important than the in-sample results.
- Historical performance does not imply future profitability.

## Skills demonstrated

**Object-oriented/reusable Python design · systematic strategy research · market-data handling · bias-aware backtesting · transaction-cost modelling · signal execution · parameter optimisation · out-of-sample evaluation · risk metrics · trade analytics · automated testing**

> Educational and research project only — not investment advice.
