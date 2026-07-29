"""Parameter search, strategy-suite execution, and summary helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd

from backtester import BacktestConfig, run_backtest
from strategies import (
    buy_and_hold_strategy,
    mean_reversion_strategy,
    momentum_strategy,
    moving_average_strategy,
)


def metrics_table(strategy_runs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Create one comparison table from multiple strategy-run dictionaries."""

    rows: dict[str, dict[str, Any]] = {}
    for name, run in strategy_runs.items():
        metrics = run["metrics"]
        rows[name] = {
            "Final Value": metrics["final_value"],
            "Total Return": metrics["total_return"],
            "Annualized Return": metrics["annualized_return"],
            "Annualized Volatility": metrics["annualized_volatility"],
            "Sharpe Ratio": metrics["sharpe_ratio"],
            "Sortino Ratio": metrics["sortino_ratio"],
            "Maximum Drawdown": metrics["max_drawdown"],
            "Daily Win Rate": metrics["daily_win_rate"],
            "Market Exposure": metrics["market_exposure"],
            "Orders": metrics["number_of_orders"],
            "Completed Trades": metrics["number_of_completed_trades"],
            "Trade Win Rate": metrics["trade_win_rate"],
            "Average Trade Return": metrics["average_trade_return"],
            "Profit Factor": metrics["profit_factor"],
        }

    return pd.DataFrame.from_dict(rows, orient="index")


def _run_parameter_grid(
    prices: pd.Series,
    parameter_combinations: Iterable[dict[str, Any]],
    strategy_function: Callable[..., pd.DataFrame],
    config: BacktestConfig,
    minimum_completed_trades: int = 1,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for parameters in parameter_combinations:
        strategy = strategy_function(prices=prices, **parameters)
        _, metrics, _ = run_backtest(prices, strategy["signal"], config)

        rows.append(
            {
                **parameters,
                "total_return": metrics["total_return"],
                "annualized_return": metrics["annualized_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "sortino_ratio": metrics["sortino_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "completed_trades": metrics["number_of_completed_trades"],
                "orders": metrics["number_of_orders"],
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("parameter search produced no results")

    eligible = results[
        results["completed_trades"] >= minimum_completed_trades
    ].copy()
    if eligible.empty:
        eligible = results.copy()

    eligible = eligible.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["sharpe_ratio"]
    )
    if eligible.empty:
        raise ValueError("parameter search produced no valid Sharpe ratios")

    return eligible.sort_values(
        ["sharpe_ratio", "annualized_return"], ascending=[False, False]
    ).reset_index(drop=True)


def optimize_moving_average(
    prices: pd.Series,
    short_windows: Iterable[int],
    long_windows: Iterable[int],
    config: BacktestConfig,
) -> pd.DataFrame:
    combinations = [
        {"short_window": short, "long_window": long}
        for short in short_windows
        for long in long_windows
        if short < long
    ]
    return _run_parameter_grid(
        prices,
        combinations,
        moving_average_strategy,
        config,
        minimum_completed_trades=1,
    )


def optimize_momentum(
    prices: pd.Series,
    lookback_windows: Iterable[int],
    config: BacktestConfig,
) -> pd.DataFrame:
    combinations = [
        {"lookback_window": lookback} for lookback in lookback_windows
    ]
    return _run_parameter_grid(
        prices,
        combinations,
        momentum_strategy,
        config,
        minimum_completed_trades=2,
    )


def optimize_mean_reversion(
    prices: pd.Series,
    lookback_windows: Iterable[int],
    entry_z_scores: Iterable[float],
    exit_z_scores: Iterable[float],
    config: BacktestConfig,
) -> pd.DataFrame:
    combinations = [
        {
            "lookback_window": lookback,
            "entry_z_score": entry,
            "exit_z_score": exit_value,
        }
        for lookback in lookback_windows
        for entry in entry_z_scores
        for exit_value in exit_z_scores
        if exit_value < entry
    ]
    return _run_parameter_grid(
        prices,
        combinations,
        mean_reversion_strategy,
        config,
        minimum_completed_trades=3,
    )


def run_strategy_suite(
    prices: pd.Series,
    config: BacktestConfig,
    moving_average_parameters: dict[str, Any],
    momentum_parameters: dict[str, Any],
    mean_reversion_parameters: dict[str, Any],
    signal_history_prices: pd.Series | None = None,
) -> dict[str, dict[str, Any]]:
    """Run all strategies for ``prices`` with optional earlier indicator history."""

    signal_history_prices = signal_history_prices if signal_history_prices is not None else prices

    strategy_frames = {
        "Moving Average": moving_average_strategy(
            signal_history_prices, **moving_average_parameters
        ),
        "Momentum": momentum_strategy(signal_history_prices, **momentum_parameters),
        "Mean Reversion": mean_reversion_strategy(
            signal_history_prices, **mean_reversion_parameters
        ),
        "Buy and Hold": buy_and_hold_strategy(signal_history_prices),
    }

    runs: dict[str, dict[str, Any]] = {}
    for name, strategy_frame in strategy_frames.items():
        period_signals = strategy_frame["signal"].reindex(prices.index).fillna(0.0)
        results, metrics, trades = run_backtest(prices, period_signals, config)
        runs[name] = {
            "results": results,
            "metrics": metrics,
            "trades": trades,
            "indicators": strategy_frame.reindex(prices.index),
        }

    return runs
