"""Reusable daily-bar backtesting engine.

The engine accepts a price series and a target-position series:

    1  = fully long
    0  = cash
   -1  = fully short

Signals are delayed by one trading period by default, which prevents a strategy
from using today's closing information to earn today's close-to-close return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration shared by every strategy backtest."""

    initial_capital: float = 10_000.0
    transaction_cost_rate: float = 0.001
    annual_risk_free_rate: float = 0.0
    periods_per_year: int = 252
    execution_delay: int = 1

    def validate(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be greater than zero")
        if not 0 <= self.transaction_cost_rate < 1:
            raise ValueError("transaction_cost_rate must be between 0 and 1")
        if self.annual_risk_free_rate <= -1:
            raise ValueError("annual_risk_free_rate must be greater than -100%")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be greater than zero")
        if self.execution_delay < 0:
            raise ValueError("execution_delay cannot be negative")


def _prepare_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")

    clean = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()

    if clean.empty:
        raise ValueError("prices contains no usable values")
    if len(clean) < 2:
        raise ValueError("at least two prices are required")
    if (clean <= 0).any():
        raise ValueError("all prices must be greater than zero")

    return clean


def _prepare_target_positions(
    target_positions: pd.Series,
    price_index: pd.Index,
) -> pd.Series:
    if not isinstance(target_positions, pd.Series):
        raise TypeError("target_positions must be a pandas Series")

    positions = (
        pd.to_numeric(target_positions, errors="coerce")
        .reindex(price_index)
        .fillna(0.0)
        .astype(float)
    )

    allowed = np.isclose(positions, -1.0) | np.isclose(positions, 0.0) | np.isclose(
        positions, 1.0
    )
    if not bool(np.all(allowed)):
        raise ValueError("target positions must contain only -1, 0, or 1")

    return positions.round().astype(float)


def calculate_drawdown(portfolio: pd.Series) -> pd.Series:
    """Return the percentage decline from each previous portfolio peak."""

    running_peak = portfolio.cummax()
    return portfolio.div(running_peak).sub(1.0)


def extract_trades(results: pd.DataFrame) -> pd.DataFrame:
    """Convert an executed-position series into individual long/short trades.

    Entry and exit transaction costs are already included because trade returns
    are compounded from the engine's net daily strategy returns.
    """

    required = {"position", "strategy_return", "price"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results is missing required columns: {sorted(missing)}")

    trade_columns = [
        "entry_date",
        "exit_date",
        "side",
        "entry_price",
        "exit_price",
        "holding_days",
        "trade_return",
        "status",
    ]

    trades: list[dict[str, Any]] = []
    current_side = 0
    entry_date: Any | None = None
    entry_price: float | None = None
    accumulated_returns: list[float] = []
    holding_days = 0
    previous_date: Any | None = None
    previous_price: float | None = None

    def close_trade(exit_date: Any, exit_price: float, status: str) -> None:
        nonlocal current_side, entry_date, entry_price, accumulated_returns, holding_days

        if current_side == 0 or entry_date is None or entry_price is None:
            return

        trade_return = float(np.prod(1.0 + np.asarray(accumulated_returns)) - 1.0)
        trades.append(
            {
                "entry_date": entry_date,
                "exit_date": exit_date,
                "side": "Long" if current_side == 1 else "Short",
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "holding_days": int(holding_days),
                "trade_return": trade_return,
                "status": status,
            }
        )

        current_side = 0
        entry_date = None
        entry_price = None
        accumulated_returns = []
        holding_days = 0

    for date, row in results.iterrows():
        position = int(row["position"])
        daily_return = float(row["strategy_return"])
        price = float(row["price"])

        if current_side == 0:
            if position != 0:
                current_side = position
                entry_date = date
                entry_price = price
                accumulated_returns = [daily_return]
                holding_days = 1

        elif position == current_side:
            accumulated_returns.append(daily_return)
            holding_days += 1

        elif position == 0:
            # On an exit day the gross strategy return is zero, but the exit
            # transaction cost is charged. Include that cost in the old trade.
            accumulated_returns.append(daily_return)
            close_trade(date, price, "Closed")

        else:
            # Direct reversal. Close the old trade at the previous observation,
            # then begin the new trade today. The day's reversal cost is assigned
            # to the new trade because today's return uses today's new position.
            if previous_date is not None and previous_price is not None:
                close_trade(previous_date, previous_price, "Closed")

            current_side = position
            entry_date = date
            entry_price = price
            accumulated_returns = [daily_return]
            holding_days = 1

        previous_date = date
        previous_price = price

    if current_side != 0 and previous_date is not None and previous_price is not None:
        close_trade(previous_date, previous_price, "Open")

    return pd.DataFrame(trades, columns=trade_columns)


def calculate_metrics(
    results: pd.DataFrame,
    trades: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, float | int]:
    """Calculate return, risk, activity, and trade statistics."""

    strategy_returns = results["strategy_return"]
    portfolio = results["portfolio"]
    positions = results["position"]

    final_value = float(portfolio.iloc[-1])
    total_return = final_value / config.initial_capital - 1.0

    if isinstance(results.index, pd.DatetimeIndex) and len(results.index) > 1:
        elapsed_years = max(
            (results.index[-1] - results.index[0]).total_seconds()
            / (365.25 * 24 * 60 * 60),
            1.0 / config.periods_per_year,
        )
    else:
        elapsed_years = len(results) / config.periods_per_year

    annualized_return = (final_value / config.initial_capital) ** (
        1.0 / elapsed_years
    ) - 1.0

    daily_volatility = float(strategy_returns.std(ddof=1))
    annualized_volatility = daily_volatility * np.sqrt(config.periods_per_year)

    daily_risk_free_rate = (1.0 + config.annual_risk_free_rate) ** (
        1.0 / config.periods_per_year
    ) - 1.0
    excess_returns = strategy_returns - daily_risk_free_rate

    if daily_volatility == 0 or np.isnan(daily_volatility):
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (
            float(excess_returns.mean())
            / daily_volatility
            * np.sqrt(config.periods_per_year)
        )

    negative_returns = strategy_returns[strategy_returns < 0]
    downside_deviation = float(negative_returns.std(ddof=1))
    if negative_returns.empty or downside_deviation == 0 or np.isnan(downside_deviation):
        sortino_ratio = np.nan
    else:
        sortino_ratio = (
            float(excess_returns.mean())
            / downside_deviation
            * np.sqrt(config.periods_per_year)
        )

    active_returns = strategy_returns[positions != 0]
    daily_win_rate = (
        float((active_returns > 0).mean()) if not active_returns.empty else np.nan
    )

    closed_trades = trades[trades["status"] == "Closed"] if not trades.empty else trades
    winning_trades = (
        closed_trades[closed_trades["trade_return"] > 0]
        if not closed_trades.empty
        else closed_trades
    )
    losing_trades = (
        closed_trades[closed_trades["trade_return"] < 0]
        if not closed_trades.empty
        else closed_trades
    )

    gross_profit = float(winning_trades["trade_return"].sum()) if not winning_trades.empty else 0.0
    gross_loss = float(-losing_trades["trade_return"].sum()) if not losing_trades.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    return {
        "final_value": final_value,
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "max_drawdown": float(results["drawdown"].min()),
        "daily_win_rate": float(daily_win_rate),
        "market_exposure": float(positions.abs().mean()),
        "number_of_orders": int(round(float(results["turnover"].sum()))),
        "number_of_completed_trades": int(len(closed_trades)),
        "trade_win_rate": (
            float((closed_trades["trade_return"] > 0).mean())
            if not closed_trades.empty
            else np.nan
        ),
        "average_trade_return": (
            float(closed_trades["trade_return"].mean())
            if not closed_trades.empty
            else np.nan
        ),
        "best_trade_return": (
            float(closed_trades["trade_return"].max())
            if not closed_trades.empty
            else np.nan
        ),
        "worst_trade_return": (
            float(closed_trades["trade_return"].min())
            if not closed_trades.empty
            else np.nan
        ),
        "profit_factor": float(profit_factor),
    }


def run_backtest(
    prices: pd.Series,
    target_positions: pd.Series,
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int], pd.DataFrame]:
    """Run a daily close-to-close backtest.

    Parameters
    ----------
    prices:
        Positive historical prices indexed by date.
    target_positions:
        Strategy decisions indexed like prices. Values must be -1, 0, or 1.
    config:
        Capital, transaction-cost, risk-free-rate, and timing assumptions.

    Returns
    -------
    results, metrics, trades
    """

    config = config or BacktestConfig()
    config.validate()

    prices = _prepare_prices(prices)
    target_positions = _prepare_target_positions(target_positions, prices.index)

    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
    positions = target_positions.shift(config.execution_delay).fillna(0.0)

    previous_positions = positions.shift(1).fillna(0.0)
    turnover = positions.sub(previous_positions).abs()

    gross_strategy_returns = positions.mul(asset_returns)
    transaction_costs = turnover.mul(config.transaction_cost_rate)
    strategy_returns = gross_strategy_returns.sub(transaction_costs)

    if (strategy_returns <= -1.0).any():
        raise ValueError(
            "A daily net return was -100% or worse. Check costs, prices, and positions."
        )

    portfolio = config.initial_capital * strategy_returns.add(1.0).cumprod()
    drawdown = calculate_drawdown(portfolio)

    results = pd.DataFrame(
        {
            "price": prices,
            "target_position": target_positions,
            "position": positions,
            "asset_return": asset_returns,
            "gross_strategy_return": gross_strategy_returns,
            "turnover": turnover,
            "transaction_cost": transaction_costs,
            "strategy_return": strategy_returns,
            "portfolio": portfolio,
            "drawdown": drawdown,
        }
    )

    trades = extract_trades(results)
    metrics = calculate_metrics(results, trades, config)

    return results, metrics, trades
