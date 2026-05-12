"""
Data loading for the Latent Regime Detection system.
Author: Jesper Mathias Nielsen
"""

import numpy as np
import pandas as pd


def load_price_data(filepath: str) -> pd.DataFrame:
    """
    Load historical OHLCV price data from a CSV file.

    The CSV must contain at minimum a 'Date' column and a 'Close' column.
    Rows are sorted chronologically and indexed by date.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame with at least a 'Close' column.
    """
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df


def generate_synthetic_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic price data with embedded regime shifts for testing.

    Simulates three distinct periods: bull trend, bear trend, sideways drift.

    Parameters
    ----------
    n : int
        Total number of daily bars to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame with 'Close' prices.
    """
    rng = np.random.default_rng(seed)
    prices = [100.0]

    segment = n // 3
    # Bull regime: positive drift, moderate vol
    for _ in range(segment):
        prices.append(prices[-1] * np.exp(rng.normal(0.0008, 0.010)))
    # Bear regime: negative drift, higher vol
    for _ in range(segment):
        prices.append(prices[-1] * np.exp(rng.normal(-0.0006, 0.018)))
    # Sideways regime: near-zero drift, low vol
    for _ in range(n - 2 * segment):
        prices.append(prices[-1] * np.exp(rng.normal(0.0001, 0.006)))

    dates = pd.date_range(end="2024-12-31", periods=len(prices), freq="B")
    return pd.DataFrame({"Close": prices}, index=dates)
