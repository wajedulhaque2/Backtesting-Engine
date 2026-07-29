import numpy as np
import pandas as pd
import pytest

from backtester import BacktestConfig, calculate_drawdown, run_backtest
from strategies import (
    mean_reversion_strategy,
    momentum_strategy,
    moving_average_strategy,
)


def dated_series(values):
    return pd.Series(
        values,
        index=pd.date_range("2024-01-01", periods=len(values), freq="D"),
        dtype=float,
    )


def test_signal_is_delayed_one_period():
    prices = dated_series([100, 110, 121])
    signals = dated_series([1, 1, 1])

    results, metrics, _ = run_backtest(
        prices,
        signals,
        BacktestConfig(transaction_cost_rate=0.0, execution_delay=1),
    )

    assert results["position"].tolist() == [0.0, 1.0, 1.0]
    assert results["strategy_return"].tolist() == pytest.approx([0.0, 0.1, 0.1])
    assert metrics["final_value"] == pytest.approx(12_100.0)


def test_transaction_costs_apply_on_entry_and_exit():
    prices = dated_series([100, 100, 100, 100])
    signals = dated_series([1, 1, 0, 0])

    results, metrics, trades = run_backtest(
        prices,
        signals,
        BacktestConfig(transaction_cost_rate=0.01, execution_delay=1),
    )

    assert results["turnover"].sum() == pytest.approx(2.0)
    assert metrics["number_of_orders"] == 2
    assert len(trades) == 1
    assert trades.iloc[0]["status"] == "Closed"
    assert metrics["final_value"] == pytest.approx(10_000 * 0.99 * 0.99)


def test_drawdown_uses_previous_peak():
    portfolio = dated_series([100, 120, 90, 108, 130])
    drawdown = calculate_drawdown(portfolio)

    assert drawdown.min() == pytest.approx(-0.25)
    assert drawdown.iloc[-1] == pytest.approx(0.0)


def test_long_and_short_positions_are_supported():
    prices = dated_series([100, 110, 99, 89.1])
    signals = dated_series([1, -1, -1, 0])

    results, metrics, _ = run_backtest(
        prices,
        signals,
        BacktestConfig(transaction_cost_rate=0.0, execution_delay=1),
    )

    assert results["position"].tolist() == [0.0, 1.0, -1.0, -1.0]
    assert metrics["final_value"] > 10_000


def test_moving_average_rejects_invalid_windows():
    prices = dated_series([100, 101, 102, 103])
    with pytest.raises(ValueError):
        moving_average_strategy(prices, short_window=20, long_window=10)


def test_momentum_signal_turns_positive_after_lookback_gain():
    prices = dated_series([100, 101, 102, 103])
    strategy = momentum_strategy(prices, lookback_window=2)
    assert strategy["signal"].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_mean_reversion_returns_binary_signal():
    prices = dated_series([100, 100, 100, 90, 92, 100, 102, 101])
    strategy = mean_reversion_strategy(
        prices, lookback_window=3, entry_z_score=1.0, exit_z_score=0.0
    )
    assert set(strategy["signal"].unique()).issubset({0.0, 1.0})


def test_invalid_positions_are_rejected():
    prices = dated_series([100, 101, 102])
    invalid_signals = dated_series([0, 0.5, 1])
    with pytest.raises(ValueError):
        run_backtest(prices, invalid_signals)
