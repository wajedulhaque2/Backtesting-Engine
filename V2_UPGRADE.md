# Backtesting Engine V2

This upgrade keeps the original single-asset research workflow intact and adds two reusable research components: a multi-asset target-weight portfolio engine and rolling walk-forward parameter validation.

## Multi-asset portfolio engine

`backtesting engine/portfolio_backtester.py` accepts a price `DataFrame` and a target-weight `DataFrame`. Weights are fractions of equity, so strategies can express fractional exposure, shorts, market-neutral portfolios and limited leverage.

```python
import pandas as pd
from portfolio_backtester import PortfolioBacktestConfig, run_portfolio_backtest

prices = pd.DataFrame({"SPY": spy_prices, "TLT": tlt_prices}).dropna()
target_weights = pd.DataFrame(
    {"SPY": 0.60, "TLT": 0.40},
    index=prices.index,
)

config = PortfolioBacktestConfig(
    commission_rate=0.0001,
    spread_rate=0.0005,
    slippage_rate=0.0002,
    annual_cash_rate=0.04,
    annual_short_borrow_rate=0.03,
    execution_delay=1,
    max_gross_exposure=1.50,
    max_absolute_net_exposure=1.00,
)

results, metrics, executed_weights = run_portfolio_backtest(
    prices, target_weights, config
)
```

The engine reports gross/net exposure, cash weight, short notional, turnover, transaction costs, borrow costs, drawdown, Sharpe/Sortino and portfolio returns. Exposure limits are enforced before returns are calculated.

`spread_rate` should be interpreted as the execution spread cost charged per unit of turnover, not necessarily the full quoted bid/ask spread.

## Walk-forward validation

`backtesting engine/walk_forward.py` repeatedly selects parameters on a training window and evaluates the selected configuration only on the following unseen window. Indicator history from the training sample is retained when test signals are generated, so the test period does not suffer an artificial indicator cold start.

```python
from walk_forward import WalkForwardConfig, walk_forward_parameter_search
from strategies import momentum_strategy

walk_forward = walk_forward_parameter_search(
    prices=spy_prices,
    parameter_combinations=[
        {"lookback_window": 63},
        {"lookback_window": 126},
        {"lookback_window": 252},
    ],
    strategy_function=momentum_strategy,
    backtest_config=backtest_config,
    walk_forward_config=WalkForwardConfig(
        train_periods=756,
        test_periods=252,
        step_periods=252,
    ),
)
```

## Verification

The V2 tests cover fractional long/short portfolios, one-period execution delay, transaction-cost accounting, cash yield, borrow costs, gross-exposure rejection, market-neutral exposure and chronology across multiple walk-forward windows. GitHub Actions runs the complete test suite on every push and pull request.

## Scope

This remains a daily-bar research engine. It does not claim to reproduce intraday order-book execution, market impact or broker margin rules. The explicit target-weight architecture is intended as the stable base for future portfolio strategies, including multi-pair statistical arbitrage and volatility-scaled allocations.
