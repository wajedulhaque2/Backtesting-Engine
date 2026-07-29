"""Market-data adapter kept separate from the deterministic backtest engine."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def download_close_prices(
    ticker: str,
    start: str,
    end: str,
    cache_directory: str | Path = ".cache",
    refresh: bool = False,
) -> pd.Series:
    """Download adjusted daily close prices and cache them as CSV.

    ``auto_adjust=True`` makes yfinance return adjusted OHLC values, so the
    ``Close`` column is suitable for split- and distribution-aware return work.
    """

    if not ticker or not ticker.strip():
        raise ValueError("ticker cannot be empty")

    ticker = ticker.strip().upper()
    cache_directory = Path(cache_directory)
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_path = cache_directory / f"{ticker}_{start}_{end}.csv"

    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        if "Close" not in cached.columns or cached.empty:
            raise ValueError(f"Invalid cached market-data file: {cache_path}")
        prices = cached["Close"].astype(float)
        prices.name = ticker
        return prices.sort_index()

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for downloads. Run: pip install -r requirements.txt"
        ) from exc

    data = yf.Ticker(ticker).history(
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        actions=False,
    )

    if data.empty or "Close" not in data.columns:
        raise ValueError(
            f"No historical daily prices were returned for {ticker} from {start} to {end}."
        )

    prices = data["Close"].dropna().astype(float)
    if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    prices.index.name = "Date"
    prices.name = ticker
    prices.to_frame("Close").to_csv(cache_path)

    return prices.sort_index()
