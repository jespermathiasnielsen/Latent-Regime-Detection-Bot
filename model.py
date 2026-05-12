"""
Hidden Markov Model for latent market regime detection.
Author: Jesper Mathias Nielsen
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def fit_hmm(
    features: pd.DataFrame,
    n_states: int = 3,
    random_state: int = 42,
) -> tuple:
    """
    Fit a Gaussian HMM to the feature matrix and infer hidden states.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix (returns, volatility, momentum).
    n_states : int
        Number of latent regimes to model.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple
        (hidden_states, state_probs, transition_matrix, fitted_model)
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=random_state,
    )
    model.fit(features)
    hidden_states = model.predict(features)
    state_probs   = model.predict_proba(features)
    return hidden_states, state_probs, model.transmat_, model


def map_regimes(
    features: pd.DataFrame,
    hidden_states: np.ndarray,
) -> tuple[list[str], dict[int, str]]:
    """
    Label each HMM state as 'bull', 'bear', or 'sideways' based on
    mean return and volatility within each state cluster.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix indexed by date.
    hidden_states : np.ndarray
        Integer state labels from the fitted HMM.

    Returns
    -------
    tuple[list[str], dict[int, str]]
        (regime_labels_per_row, state_to_regime_mapping)
    """
    regime_map: dict[int, str] = {}
    vol_low_threshold = features["volatility"].quantile(0.40)

    for state in np.unique(hidden_states):
        mask = hidden_states == state
        mean_ret = features["returns"][mask].mean()
        mean_vol = features["volatility"][mask].mean()

        if mean_vol < vol_low_threshold:
            regime = "sideways"
        elif mean_ret < 0:
            regime = "bear"
        else:
            regime = "bull"

        regime_map[state] = regime

    return [regime_map[s] for s in hidden_states], regime_map


def regime_statistics(
    features: pd.DataFrame,
    regimes: list[str],
    transmat: np.ndarray,
) -> None:
    """
    Print a summary of regime characteristics and persistence.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix used for fitting.
    regimes : list[str]
        Per-row regime labels.
    transmat : np.ndarray
        HMM transition probability matrix.
    """
    regime_series = pd.Series(regimes, index=features.index)

    print("─" * 50)
    print("Regime Summary")
    print("─" * 50)

    for regime in sorted(set(regimes)):
        mask = regime_series == regime
        count = mask.sum()
        pct   = 100 * count / len(regimes)
        mean_ret = features["returns"][mask.values].mean()
        mean_vol = features["volatility"][mask.values].mean()
        print(
            f"  {regime.upper():>10}  |  {count:4d} days ({pct:5.1f}%)  |"
            f"  avg return: {mean_ret:+.4f}  |  avg vol: {mean_vol:.4f}"
        )

    # Average regime duration
    runs = (regime_series != regime_series.shift()).cumsum()
    avg_dur = regime_series.groupby(runs).agg(["count", "first"])
    avg_dur = avg_dur.groupby("first")["count"].mean()
    print("\nAverage duration per regime (days):")
    for regime, dur in avg_dur.items():
        print(f"  {regime.upper():>10}  :  {dur:.1f}")

    print("\nTransition Matrix:")
    print(np.round(transmat, 3))
    print("─" * 50)
