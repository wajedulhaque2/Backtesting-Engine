"""Multi-asset target-weight portfolio backtester.

This module complements the original single-asset backtester. Strategies provide
one target weight per asset and date; the engine applies an execution delay,
portfolio constraints, financing, borrow costs and turnover-based trading costs.

The accounting model is intentionally transparent rather than broker-specific.
It is suitable for daily-bar research and portfolio construction experiments,
not intraday execution simulation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    initial_capital: float = 10_000.0
    commission_rate: float = 0.0001
    spread_rate: float = 0.0005
    slippage_rate: float = 0.0002
    annual_cash_rate: float = 0.0
    annual_short_borrow_rate: float = 0.03
    periods_per_year: int = 252
    execution_delay: int = 1
    max_gross_exposure: float = 1.5
    max_absolute_net_exposure: float = 1.0

    def validate(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be greater than zero")
        for name, value in (
            ("commission_rate", self.commission_rate),
            ("spread_rate", self.spread_rate),
            ("slippage_rate", self.slippage_rate),
            ("annual_short_borrow_rate", self.annual_short_borrow_rate),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.annual_cash_rate <= -1:
            raise ValueError("annual_cash_rate must be greater than -100%")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        if self.execution_delay < 0:
            raise ValueError("execution_delay cannot be negative")
        if self.max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be positive")
        if self.max_absolute_net_exposure < 0:
            raise ValueError("max_absolute_net_exposure cannot be negative")
        if self.max_absolute_net_exposure > self.max_gross_exposure:
            raise ValueError(
                "max_absolute_net_exposure cannot exceed max_gross_exposure"
            )

    @property
    def turnover_cost_rate(self) -> float:
        return self.commission_rate + self.spread_rate + self.slippage_rate


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if prices.empty or prices.shape[1] == 0:
        raise ValueError("prices must contain at least one asset")

    clean = prices.apply(pd.to_numeric, errors="coerce").astype(float)
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()
    clean = clean.dropna(how="any")
    if len(clean) < 2:
        raise ValueError("at least two complete price observations are required")
    if (clean <= 0).any().any():
        raise ValueError("all prices must be greater than zero")
    if clean.columns.duplicated().any():
        raise ValueError("price columns must be unique")
    return clean


def _prepare_weights(target_weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(target_weights, pd.DataFrame):
        raise TypeError("target_weights must be a pandas DataFrame")

    extra = set(target_weights.columns).difference(prices.columns)
    if extra:
        raise ValueError(f"target_weights contains unknown assets: {sorted(extra)}")

    weights = (
        target_weights.apply(pd.to_numeric, errors="coerce")
        .reindex(index=prices.index, columns=prices.columns)
        .fillna(0.0)
        .astype(float)
    )
    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("target_weights must contain only finite values")
    return weights


def _validate_exposure_constraints(
    weights: pd.DataFrame, config: PortfolioBacktestConfig
) -> None:
    gross = weights.abs().sum(axis=1)
    net = weights.sum(axis=1).abs()
    tolerance = 1e-12
    if bool((gross > config.max_gross_exposure + tolerance).any()):
        date = gross.idxmax()
        raise ValueError(
            f"gross exposure {gross.loc[date]:.4f} exceeds limit "
            f"{config.max_gross_exposure:.4f} at {date}"
        )
    if bool((net > config.max_absolute_net_exposure + tolerance).any()):
        date = net.idxmax()
        raise ValueError(
            f"absolute net exposure {net.loc[date]:.4f} exceeds limit "
            f"{config.max_absolute_net_exposure:.4f} at {date}"
        )


def _annualized_metrics(
    returns: pd.Series,
    portfolio: pd.Series,
    config: PortfolioBacktestConfig,
) -> dict[str, float]:
    if isinstance(returns.index, pd.DatetimeIndex) and len(returns.index) > 1:
        years = max(
            (returns.index[-1] - returns.index[0]).total_seconds()
            / (365.25 * 24 * 60 * 60),
            1.0 / config.periods_per_year,
        )
    else:
        years = max(
            len(returns) / config.periods_per_year,
            1.0 / config.periods_per_year,
        )

    final_value = float(portfolio.iloc[-1])
    total_return = final_value / config.initial_capital - 1.0
    annualized_return = (
        (final_value / config.initial_capital) ** (1.0 / years) - 1.0
    )
    volatility = float(returns.std(ddof=1))
    annualized_volatility = volatility * np.sqrt(config.periods_per_year)
    daily_cash = (1.0 + config.annual_cash_rate) ** (
        1.0 / config.periods_per_year
    ) - 1.0
    excess = returns - daily_cash
    sharpe = (
        float(excess.mean()) / volatility * np.sqrt(config.periods_per_year)
        if volatility > 0 and np.isfinite(volatility)
        else np.nan
    )
    downside = returns[returns < daily_cash] - daily_cash
    downside_dev = float(downside.std(ddof=1)) if len(downside) >= 2 else np.nan
    sortino = (
        float(excess.mean()) / downside_dev * np.sqrt(config.periods_per_year)
        if np.isfinite(downside_dev) and downside_dev > 0
        else np.nan
    )
    return {
        "final_value": final_value,
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
    }


def run_portfolio_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: PortfolioBacktestConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Backtest a multi-asset target-weight portfolio.

    Weights are fractions of portfolio equity and may be fractional or negative.
    A weight of 0.40 means 40% long exposure; -0.25 means 25% short exposure.

    Cash weight is defined as ``1 - net_exposure``. This is a self-financing
    accounting approximation: short-sale proceeds remain in the cash balance,
    while an explicit borrow charge is applied to short notional.
    """

    config = config or PortfolioBacktestConfig()
    config.validate()
    prices = _prepare_prices(prices)
    target_weights = _prepare_weights(target_weights, prices)

    executed_weights = target_weights.shift(config.execution_delay).fillna(0.0)
    _validate_exposure_constraints(executed_weights, config)

    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
    previous_weights = executed_weights.shift(1).fillna(0.0)
    turnover_by_asset = executed_weights.sub(previous_weights).abs()
    turnover = turnover_by_asset.sum(axis=1)

    gross_exposure = executed_weights.abs().sum(axis=1)
    net_exposure = executed_weights.sum(axis=1)
    short_notional = (-executed_weights).clip(lower=0.0).sum(axis=1)
    cash_weight = 1.0 - net_exposure

    gross_asset_return = executed_weights.mul(asset_returns).sum(axis=1)
    daily_cash_rate = (1.0 + config.annual_cash_rate) ** (
        1.0 / config.periods_per_year
    ) - 1.0
    daily_borrow_rate = (1.0 + config.annual_short_borrow_rate) ** (
        1.0 / config.periods_per_year
    ) - 1.0

    cash_return = cash_weight * daily_cash_rate
    transaction_cost = turnover * config.turnover_cost_rate
    borrow_cost = short_notional * daily_borrow_rate
    strategy_return = (
        gross_asset_return + cash_return - transaction_cost - borrow_cost
    )

    if bool((strategy_return <= -1.0).any()):
        raise ValueError("a daily net portfolio return was -100% or worse")

    portfolio = config.initial_capital * (1.0 + strategy_return).cumprod()
    running_peak = portfolio.cummax()
    drawdown = portfolio.div(running_peak).sub(1.0)

    results = pd.DataFrame(
        {
            "gross_asset_return": gross_asset_return,
            "cash_return": cash_return,
            "transaction_cost": transaction_cost,
            "borrow_cost": borrow_cost,
            "strategy_return": strategy_return,
            "portfolio": portfolio,
            "drawdown": drawdown,
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "cash_weight": cash_weight,
            "short_notional": short_notional,
            "turnover": turnover,
        },
        index=prices.index,
    )

    metrics = _annualized_metrics(strategy_return, portfolio, config)
    metrics.update(
        {
            "max_drawdown": float(drawdown.min()),
            "average_gross_exposure": float(gross_exposure.mean()),
            "average_absolute_net_exposure": float(net_exposure.abs().mean()),
            "total_turnover": float(turnover.sum()),
            "total_transaction_cost_fraction": float(transaction_cost.sum()),
            "total_borrow_cost_fraction": float(borrow_cost.sum()),
        }
    )

    return results, metrics, executed_weights
