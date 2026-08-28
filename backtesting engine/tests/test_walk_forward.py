import pandas as pd

from backtester import BacktestConfig
from strategies import momentum_strategy
from walk_forward import WalkForwardConfig, walk_forward_parameter_search


def test_walk_forward_uses_multiple_unseen_windows():
    values = (
        list(range(100, 130))
        + list(range(130, 100, -1))
        + list(range(100, 140))
    )
    prices = pd.Series(
        values,
        index=pd.date_range("2020-01-01", periods=len(values), freq="D"),
        dtype=float,
    )
    result = walk_forward_parameter_search(
        prices,
        [{"lookback_window": 2}, {"lookback_window": 5}],
        momentum_strategy,
        BacktestConfig(transaction_cost_rate=0.0, execution_delay=1),
        WalkForwardConfig(
            train_periods=30,
            test_periods=15,
            step_periods=15,
            minimum_completed_trades=0,
        ),
    )
    assert len(result) >= 3
    assert set(result.columns) >= {
        "selected_parameters",
        "train_score",
        "test_total_return",
    }
    for index in range(len(result)):
        assert result.loc[index, "train_end"] < result.loc[index, "test_start"]
