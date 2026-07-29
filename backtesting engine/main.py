"""Complete research workflow for the reusable backtesting engine.

Edit the SETTINGS section, then run:

    python main.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from backtester import BacktestConfig
from market_data import download_close_prices
from research import (
    metrics_table,
    optimize_mean_reversion,
    optimize_momentum,
    optimize_moving_average,
    run_strategy_suite,
)


# ==================================================
# SETTINGS
# ==================================================

TICKER = "SPY"
START_DATE = "2010-01-01"
END_DATE = "2026-01-01"
SPLIT_DATE = "2019-01-01"

OUTPUT_DIRECTORY = Path("outputs")
SHOW_PLOTS = True
REFRESH_MARKET_DATA = False

CONFIG = BacktestConfig(
    initial_capital=10_000.0,
    transaction_cost_rate=0.001,
    annual_risk_free_rate=0.0,
    periods_per_year=252,
    execution_delay=1,
)

MOVING_AVERAGE_SHORT_WINDOWS = [20, 50, 100]
MOVING_AVERAGE_LONG_WINDOWS = [100, 150, 200, 250]
MOMENTUM_LOOKBACK_WINDOWS = [63, 126, 189, 252]
MEAN_REVERSION_LOOKBACK_WINDOWS = [10, 20, 40, 60]
MEAN_REVERSION_ENTRY_Z_SCORES = [1.0, 1.5, 2.0]
MEAN_REVERSION_EXIT_Z_SCORES = [0.0, 0.5]

PERCENT_COLUMNS = [
    "Total Return",
    "Annualized Return",
    "Annualized Volatility",
    "Maximum Drawdown",
    "Daily Win Rate",
    "Market Exposure",
    "Trade Win Rate",
    "Average Trade Return",
]


# ==================================================
# DISPLAY AND EXPORT HELPERS
# ==================================================


def display_summary(title: str, summary: pd.DataFrame) -> None:
    display = summary.copy()
    for column in PERCENT_COLUMNS:
        if column in display.columns:
            display[column] = display[column] * 100.0

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(display.round(2).to_string())
    print("\nPercentage columns are displayed as percentage points.")


def export_run_group(
    label: str,
    strategy_runs: dict[str, dict[str, Any]],
    output_directory: Path,
) -> pd.DataFrame:
    summary = metrics_table(strategy_runs)
    summary.to_csv(output_directory / f"{label}_summary.csv")

    for strategy_name, run in strategy_runs.items():
        safe_name = strategy_name.lower().replace(" ", "_")
        run["results"].to_csv(
            output_directory / f"{label}_{safe_name}_daily_results.csv"
        )
        run["trades"].to_csv(
            output_directory / f"{label}_{safe_name}_trades.csv", index=False
        )

    return summary


def save_portfolio_chart(
    title: str,
    strategy_runs: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 6))
    for name, run in strategy_runs.items():
        plt.plot(run["results"].index, run["results"]["portfolio"], label=name)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def save_drawdown_chart(
    title: str,
    strategy_runs: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 6))
    for name, run in strategy_runs.items():
        plt.plot(
            run["results"].index,
            run["results"]["drawdown"] * 100.0,
            label=name,
        )
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def save_indicator_chart(
    prices: pd.Series,
    selected_parameters: dict[str, dict[str, Any]],
    split_date: str,
    output_path: Path,
) -> None:
    from strategies import moving_average_strategy

    indicators = moving_average_strategy(
        prices, **selected_parameters["moving_average"]
    )

    plt.figure(figsize=(12, 6))
    plt.plot(prices.index, prices, label=f"{TICKER} adjusted close")
    plt.plot(
        indicators.index,
        indicators["short_average"],
        label=f"{selected_parameters['moving_average']['short_window']}-day average",
    )
    plt.plot(
        indicators.index,
        indicators["long_average"],
        label=f"{selected_parameters['moving_average']['long_window']}-day average",
    )
    plt.axvline(pd.Timestamp(split_date),
                linestyle="--", label="Train/test split")
    plt.title(f"{TICKER}: selected moving averages")
    plt.xlabel("Date")
    plt.ylabel("Adjusted price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# =================================================
# MAIN WORKFLOW
# ================================================

def main() -> dict[str, Any]:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    prices = download_close_prices(
        ticker=TICKER,
        start=START_DATE,
        end=END_DATE,
        refresh=REFRESH_MARKET_DATA,
    )

    training_prices = prices[prices.index < pd.Timestamp(SPLIT_DATE)].copy()
    testing_prices = prices[prices.index >= pd.Timestamp(SPLIT_DATE)].copy()

    if training_prices.empty or testing_prices.empty:
        raise ValueError(
            "SPLIT_DATE must leave data on both sides of the split")

    print(f"Ticker: {TICKER}")
    print(
        f"Training period: {training_prices.index[0].date()} "
        f"to {training_prices.index[-1].date()}"
    )
    print(
        f"Testing period:  {testing_prices.index[0].date()} "
        f"to {testing_prices.index[-1].date()}"
    )

    ma_search = optimize_moving_average(
        training_prices,
        MOVING_AVERAGE_SHORT_WINDOWS,
        MOVING_AVERAGE_LONG_WINDOWS,
        CONFIG,
    )
    momentum_search = optimize_momentum(
        training_prices,
        MOMENTUM_LOOKBACK_WINDOWS,
        CONFIG,
    )
    mean_reversion_search = optimize_mean_reversion(
        training_prices,
        MEAN_REVERSION_LOOKBACK_WINDOWS,
        MEAN_REVERSION_ENTRY_Z_SCORES,
        MEAN_REVERSION_EXIT_Z_SCORES,
        CONFIG,
    )

    ma_search.to_csv(OUTPUT_DIRECTORY /
                     "moving_average_parameter_search.csv", index=False)
    momentum_search.to_csv(
        OUTPUT_DIRECTORY / "momentum_parameter_search.csv", index=False
    )
    mean_reversion_search.to_csv(
        OUTPUT_DIRECTORY / "mean_reversion_parameter_search.csv", index=False
    )

    best_ma = ma_search.iloc[0]
    best_momentum = momentum_search.iloc[0]
    best_mean_reversion = mean_reversion_search.iloc[0]

    selected_parameters = {
        "moving_average": {
            "short_window": int(best_ma["short_window"]),
            "long_window": int(best_ma["long_window"]),
        },
        "momentum": {
            "lookback_window": int(best_momentum["lookback_window"]),
        },
        "mean_reversion": {
            "lookback_window": int(best_mean_reversion["lookback_window"]),
            "entry_z_score": float(best_mean_reversion["entry_z_score"]),
            "exit_z_score": float(best_mean_reversion["exit_z_score"]),
        },
    }

    with (OUTPUT_DIRECTORY / "selected_parameters.json").open("w", encoding="utf-8") as file:
        json.dump(selected_parameters, file, indent=2)

    print("\nSelected using training-period Sharpe ratio:")
    print(json.dumps(selected_parameters, indent=2))

    # Training search is in-sample. Testing is out-of-sample. Full-period results
    # are included only as a descriptive overview.
    full_runs = run_strategy_suite(
        prices,
        CONFIG,
        selected_parameters["moving_average"],
        selected_parameters["momentum"],
        selected_parameters["mean_reversion"],
        signal_history_prices=prices,
    )
    training_runs = run_strategy_suite(
        training_prices,
        CONFIG,
        selected_parameters["moving_average"],
        selected_parameters["momentum"],
        selected_parameters["mean_reversion"],
        signal_history_prices=training_prices,
    )
    testing_runs = run_strategy_suite(
        testing_prices,
        CONFIG,
        selected_parameters["moving_average"],
        selected_parameters["momentum"],
        selected_parameters["mean_reversion"],
        signal_history_prices=prices,
    )

    full_summary = export_run_group("full", full_runs, OUTPUT_DIRECTORY)
    training_summary = export_run_group(
        "training", training_runs, OUTPUT_DIRECTORY)
    testing_summary = export_run_group(
        "testing", testing_runs, OUTPUT_DIRECTORY)

    display_summary("FULL-PERIOD RESULTS — DESCRIPTIVE", full_summary)
    display_summary("TRAINING RESULTS — IN SAMPLE", training_summary)
    display_summary("TESTING RESULTS — OUT OF SAMPLE", testing_summary)

    save_portfolio_chart(
        f"{TICKER}: out-of-sample portfolio comparison",
        testing_runs,
        OUTPUT_DIRECTORY / "testing_portfolio_comparison.png",
    )
    save_drawdown_chart(
        f"{TICKER}: out-of-sample drawdown comparison",
        testing_runs,
        OUTPUT_DIRECTORY / "testing_drawdown_comparison.png",
    )
    save_indicator_chart(
        prices,
        selected_parameters,
        SPLIT_DATE,
        OUTPUT_DIRECTORY / "selected_moving_averages.png",
    )

    print(f"\nFiles saved to: {OUTPUT_DIRECTORY.resolve()}")
    print(
        "Treat the testing section as the main evidence. Good historical results "
        "do not guarantee future performance."
    )

    return {
        "prices": prices,
        "selected_parameters": selected_parameters,
        "parameter_search": {
            "moving_average": ma_search,
            "momentum": momentum_search,
            "mean_reversion": mean_reversion_search,
        },
        "runs": {
            "full": full_runs,
            "training": training_runs,
            "testing": testing_runs,
        },
        "summaries": {
            "full": full_summary,
            "training": training_summary,
            "testing": testing_summary,
        },
    }


if __name__ == "__main__":
    main()
