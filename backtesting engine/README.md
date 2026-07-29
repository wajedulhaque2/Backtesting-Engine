# Final Backtesting Engine

A reusable Python project for comparing trading strategies against buy-and-hold historical performance.

## Included

- One-period signal delay to reduce look-ahead bias
- Long, cash, and short positions
- Transaction costs based on turnover
- Buy-and-hold benchmark
- Moving-average trend strategy
- Trailing-return momentum strategy
- Rolling z-score mean-reversion strategy
- Training/testing split
- Training-only parameter search
- Out-of-sample evaluation
- Individual trade extraction
- CSV exports and charts
- Automated tests

## Metrics

- Final portfolio value
- Total and annualized return
- Annualized volatility
- Sharpe and Sortino ratios
- Maximum drawdown
- Daily win rate
- Market exposure
- Number of orders
- Completed-trade win rate
- Average, best, and worst trade return
- Profit factor

## Project files

```text
backtester.py       Core engine and metrics
strategies.py       Strategy signal generators
market_data.py      yfinance download and CSV cache
research.py         Parameter search and comparison helpers
main.py             Complete research workflow
main.ipynb          Notebook launcher
requirements.txt    Python dependencies
tests/              Automated tests
outputs/            Generated CSV files and charts
```

## Installation

Open the project folder in VS Code, open a terminal, and run:

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the finished project

```bash
python main.py
```

Or open `main.ipynb` and run its cells.

The default project downloads SPY adjusted daily prices, uses 2010–2018 as training data, and evaluates the selected settings on 2019–2025 data. Edit the **SETTINGS** section near the top of `main.py` to change the ticker, dates, costs, or candidate parameters.

## Run tests

```bash
pytest -q
```

The tests use synthetic prices, so they do not require an internet connection.

## How to interpret the output

The parameter search uses only the training period. The out-of-sample testing table is the most important result because those dates were not used to choose the settings. The full-period table is descriptive and should not be treated as independent evidence.

A high historical return or Sharpe ratio does not prove that a strategy will work in the future. Real trading may also involve taxes, spreads, slippage, market impact, borrowing constraints, and data-quality problems that this educational engine does not fully model.

## Data note

The downloader requests `auto_adjust=True` from yfinance and uses the returned `Close` series. Downloaded data is cached in `.cache/` so reruns are faster. Delete the cached file or set `REFRESH_MARKET_DATA = True` to request fresh data.
