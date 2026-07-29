"""Trading-strategy signal generators.

Each function returns a DataFrame whose ``signal`` column contains target
positions. The backtester delays those signals before applying returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    clean = pd.to_numeric(prices, errors="coerce").astype(float)
    if clean.dropna().empty:
        raise ValueError("prices contains no usable values")
    return clean


def buy_and_hold_strategy(prices: pd.Series) -> pd.DataFrame:
    """Remain fully invested throughout the available period."""

    prices = _clean_prices(prices)
    return pd.DataFrame({"signal": 1.0}, index=prices.index)


def moving_average_strategy(
    prices: pd.Series,
    short_window: int = 50,
    long_window: int = 200,
) -> pd.DataFrame:
    """Own the asset while the short average is above the long average."""

    prices = _clean_prices(prices)
    if short_window <= 0 or long_window <= 0:
        raise ValueError("moving-average windows must be positive")
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")

    short_average = prices.rolling(short_window, min_periods=short_window).mean()
    long_average = prices.rolling(long_window, min_periods=long_window).mean()
    signal = (short_average > long_average).astype(float)

    return pd.DataFrame(
        {
            "signal": signal,
            "short_average": short_average,
            "long_average": long_average,
        },
        index=prices.index,
    )


def momentum_strategy(
    prices: pd.Series,
    lookback_window: int = 126,
) -> pd.DataFrame:
    """Own the asset when its trailing lookback return is positive."""

    prices = _clean_prices(prices)
    if lookback_window <= 0:
        raise ValueError("lookback_window must be positive")

    momentum_return = prices.div(prices.shift(lookback_window)).sub(1.0)
    signal = (momentum_return > 0).astype(float)

    return pd.DataFrame(
        {
            "signal": signal,
            "momentum_return": momentum_return,
        },
        index=prices.index,
    )


def mean_reversion_strategy(
    prices: pd.Series,
    lookback_window: int = 20,
    entry_z_score: float = 2.0,
    exit_z_score: float = 0.0,
) -> pd.DataFrame:
    """Long-only mean reversion based on a rolling price z-score.

    Enter when price is unusually low relative to its rolling mean and leave
    when the z-score recovers to ``exit_z_score`` or higher.
    """

    prices = _clean_prices(prices)
    if lookback_window <= 1:
        raise ValueError("lookback_window must be greater than one")
    if entry_z_score <= 0:
        raise ValueError("entry_z_score must be positive")
    if exit_z_score >= entry_z_score:
        raise ValueError("exit_z_score must be smaller than entry_z_score")

    rolling_mean = prices.rolling(
        lookback_window, min_periods=lookback_window
    ).mean()
    rolling_std = prices.rolling(
        lookback_window, min_periods=lookback_window
    ).std(ddof=0)
    z_score = prices.sub(rolling_mean).div(rolling_std.replace(0.0, np.nan))

    signal_values: list[float] = []
    current_position = 0.0

    for value in z_score:
        if pd.isna(value):
            current_position = 0.0
        elif current_position == 0.0 and value <= -entry_z_score:
            current_position = 1.0
        elif current_position == 1.0 and value >= exit_z_score:
            current_position = 0.0

        signal_values.append(current_position)

    signal = pd.Series(signal_values, index=prices.index, dtype=float)

    return pd.DataFrame(
        {
            "signal": signal,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "z_score": z_score,
        },
        index=prices.index,
    )
