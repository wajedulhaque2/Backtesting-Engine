import numpy as np
import pandas as pd
import pytest

from portfolio_backtester import PortfolioBacktestConfig, run_portfolio_backtest


def prices_frame(a, b=None):
    data = {"A": a}
    if b is not None:
        data["B"] = b
    return pd.DataFrame(
        data,
        index=pd.date_range("2024-01-01", periods=len(a), freq="D"),
        dtype=float,
    )


def test_multi_asset_weights_are_delayed_and_fractional():
    prices = prices_frame([100, 110, 121], [100, 100, 90])
    targets = pd.DataFrame(
        {"A": [0.5, 0.5, 0.5], "B": [-0.5, -0.5, -0.5]},
        index=prices.index,
    )
    config = PortfolioBacktestConfig(
        commission_rate=0,
        spread_rate=0,
        slippage_rate=0,
        annual_short_borrow_rate=0,
        annual_cash_rate=0,
        max_absolute_net_exposure=1.0,
    )
    results, _, executed = run_portfolio_backtest(prices, targets, config)
    assert executed.iloc[0].tolist() == [0.0, 0.0]
    assert executed.iloc[1].tolist() == [0.5, -0.5]
    assert results["strategy_return"].iloc[1] == pytest.approx(0.05)
    assert results["strategy_return"].iloc[2] == pytest.approx(0.10)


def test_turnover_costs_charge_each_changed_weight():
    prices = prices_frame([100, 100, 100])
    targets = pd.DataFrame({"A": [1.0, 0.0, 0.0]}, index=prices.index)
    config = PortfolioBacktestConfig(
        commission_rate=0.001,
        spread_rate=0.001,
        slippage_rate=0,
        annual_short_borrow_rate=0,
        annual_cash_rate=0,
        max_gross_exposure=1.0,
    )
    results, metrics, _ = run_portfolio_backtest(prices, targets, config)
    assert results["turnover"].sum() == pytest.approx(2.0)
    assert results["transaction_cost"].sum() == pytest.approx(0.004)
    assert metrics["final_value"] == pytest.approx(10_000 * 0.998 * 0.998)


def test_all_cash_portfolio_earns_cash_rate():
    prices = prices_frame([100, 100, 100])
    targets = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=prices.index)
    config = PortfolioBacktestConfig(
        commission_rate=0,
        spread_rate=0,
        slippage_rate=0,
        annual_cash_rate=0.10,
        annual_short_borrow_rate=0,
    )
    results, _, _ = run_portfolio_backtest(prices, targets, config)
    daily = (1.10 ** (1 / 252)) - 1
    assert results["strategy_return"].tolist() == pytest.approx(
        [daily, daily, daily]
    )


def test_short_positions_pay_borrow_cost():
    prices = prices_frame([100, 100, 100])
    targets = pd.DataFrame({"A": [-0.5, -0.5, -0.5]}, index=prices.index)
    config = PortfolioBacktestConfig(
        commission_rate=0,
        spread_rate=0,
        slippage_rate=0,
        annual_cash_rate=0,
        annual_short_borrow_rate=0.12,
    )
    results, _, executed = run_portfolio_backtest(prices, targets, config)
    daily_borrow = 1.12 ** (1 / 252) - 1
    assert executed.iloc[1, 0] == -0.5
    assert results["borrow_cost"].iloc[1] == pytest.approx(0.5 * daily_borrow)


def test_exposure_constraints_reject_excess_leverage():
    prices = prices_frame([100, 101, 102], [100, 99, 98])
    targets = pd.DataFrame(
        {"A": [1.0, 1.0, 1.0], "B": [-1.0, -1.0, -1.0]},
        index=prices.index,
    )
    config = PortfolioBacktestConfig(
        max_gross_exposure=1.5,
        max_absolute_net_exposure=1.0,
    )
    with pytest.raises(ValueError, match="gross exposure"):
        run_portfolio_backtest(prices, targets, config)


def test_market_neutral_pair_has_zero_net_and_unit_gross():
    prices = prices_frame([100, 105, 103, 110], [100, 100, 101, 99])
    targets = pd.DataFrame(
        {"A": [0.5] * 4, "B": [-0.5] * 4}, index=prices.index
    )
    config = PortfolioBacktestConfig(
        commission_rate=0,
        spread_rate=0,
        slippage_rate=0,
        annual_cash_rate=0,
        annual_short_borrow_rate=0,
    )
    results, _, _ = run_portfolio_backtest(prices, targets, config)
    active = results.iloc[1:]
    assert np.allclose(active["net_exposure"], 0.0)
    assert np.allclose(active["gross_exposure"], 1.0)
