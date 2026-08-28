"""Walk-forward parameter selection for the original single-asset engine."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backtester import BacktestConfig, run_backtest


@dataclass(frozen=True)
class WalkForwardConfig:
    train_periods: int = 756
    test_periods: int = 252
    step_periods: int = 252
    minimum_completed_trades: int = 1
    score_metric: str = "sharpe_ratio"

    def validate(self) -> None:
        if self.train_periods < 2:
            raise ValueError("train_periods must be at least two")
        if self.test_periods < 1 or self.step_periods < 1:
            raise ValueError("test_periods and step_periods must be positive")
        if self.minimum_completed_trades < 0:
            raise ValueError("minimum_completed_trades cannot be negative")


def walk_forward_parameter_search(
    prices: pd.Series,
    parameter_combinations: Iterable[dict[str, Any]],
    strategy_function: Callable[..., pd.DataFrame],
    backtest_config: BacktestConfig | None = None,
    walk_forward_config: WalkForwardConfig | None = None,
) -> pd.DataFrame:
    """Select parameters on each training window and evaluate only the next window.

    Indicator history from the training window is retained when generating test
    signals, preventing artificial cold-start behaviour at each test boundary.
    """

    backtest_config = backtest_config or BacktestConfig()
    config = walk_forward_config or WalkForwardConfig()
    config.validate()
    params = list(parameter_combinations)
    if not params:
        raise ValueError("parameter_combinations cannot be empty")

    clean = pd.to_numeric(prices, errors="coerce").dropna().astype(float).sort_index()
    needed = config.train_periods + config.test_periods
    if len(clean) < needed:
        raise ValueError(f"at least {needed} observations are required")

    rows: list[dict[str, Any]] = []
    max_start = len(clean) - needed
    for start in range(0, max_start + 1, config.step_periods):
        train = clean.iloc[start : start + config.train_periods]
        test = clean.iloc[
            start + config.train_periods :
            start + config.train_periods + config.test_periods
        ]

        candidates: list[tuple[dict[str, Any], dict[str, float | int]]] = []
        for parameter_set in params:
            signal = strategy_function(prices=train, **parameter_set)["signal"]
            _, metrics, _ = run_backtest(train, signal, backtest_config)
            score = metrics.get(config.score_metric, np.nan)
            if (
                metrics["number_of_completed_trades"]
                >= config.minimum_completed_trades
                and np.isfinite(float(score))
            ):
                candidates.append((parameter_set, metrics))

        if not candidates:
            continue
        selected_params, train_metrics = max(
            candidates,
            key=lambda item: float(item[1][config.score_metric]),
        )

        signal_history = clean.iloc[
            start : start + config.train_periods + config.test_periods
        ]
        signal_frame = strategy_function(prices=signal_history, **selected_params)
        test_signal = signal_frame["signal"].reindex(test.index).fillna(0.0)
        _, test_metrics, _ = run_backtest(test, test_signal, backtest_config)

        rows.append(
            {
                "train_start": train.index[0],
                "train_end": train.index[-1],
                "test_start": test.index[0],
                "test_end": test.index[-1],
                "selected_parameters": dict(selected_params),
                "train_score": float(train_metrics[config.score_metric]),
                "test_sharpe_ratio": float(test_metrics["sharpe_ratio"]),
                "test_total_return": float(test_metrics["total_return"]),
                "test_max_drawdown": float(test_metrics["max_drawdown"]),
                "test_completed_trades": int(
                    test_metrics["number_of_completed_trades"]
                ),
            }
        )

    if not rows:
        raise ValueError("walk-forward search produced no eligible windows")
    return pd.DataFrame(rows)
