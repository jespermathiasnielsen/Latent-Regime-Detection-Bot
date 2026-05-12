"""
Feature engineering for the Hidden Markov Model regime detector.
Author: Jesper Mathias Nielsen
"""

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Derive statistical features from price data for HMM fitting.

    Features computed:
    - Log returns
    - Rolling realised volatility (std of log returns over `window` days)
    - Momentum (% change over `window` days)

    Parameters
    ----------
    df : pd.DataFrame
        Date-indexed DataFrame containing a 'Close' column.
    window : int
        Look-back window in trading days.

    Returns
    -------
    pd.DataFrame
        Feature matrix with NaN rows dropped.
    """
    features = pd.DataFrame(index=df.index)
    features["returns"]    = np.log(df["Close"]).diff()
    features["volatility"] = features["returns"].rolling(window).std()
    features["momentum"]   = df["Close"].pct_change(window)
    return features.dropna()
