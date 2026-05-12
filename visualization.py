"""
Visualisation for Hidden Markov Model market regime detection.
Author: Jesper Mathias Nielsen
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

REGIME_COLORS = {"bull": "#00FF7F", "bear": "#FF4444", "sideways": "#AAAAAA"}


def plot_price_with_regimes(df: pd.DataFrame, regimes: list[str]) -> None:
    """
    Price chart with background shading by detected market regime.

    Bull periods are green, bear red, sideways grey.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 6))

    df["Close"].plot(ax=ax, lw=1.8, color="white", label="Close Price")

    # Shade regime spans
    last_regime, start_idx = None, 0
    for i, regime in enumerate(regimes):
        if regime != last_regime:
            if last_regime is not None:
                ax.axvspan(
                    df.index[start_idx], df.index[i - 1],
                    color=REGIME_COLORS[last_regime], alpha=0.22,
                )
            start_idx, last_regime = i, regime
    ax.axvspan(df.index[start_idx], df.index[-1], color=REGIME_COLORS[last_regime], alpha=0.22)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=REGIME_COLORS["bull"],     alpha=0.6, label="Bull"),
        Patch(facecolor=REGIME_COLORS["bear"],     alpha=0.6, label="Bear"),
        Patch(facecolor=REGIME_COLORS["sideways"], alpha=0.6, label="Sideways"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    ax.set_title("Price with Detected Market Regimes", fontsize=13)
    ax.set_ylabel("Price")
    plt.tight_layout()


def plot_regime_probabilities(df: pd.DataFrame, state_probs: np.ndarray) -> None:
    """
    Stacked area chart showing the probability of each regime over time.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 4))

    colors = ["#00FF7F", "#FF4444", "#AAAAAA"]
    labels = [f"State {i}" for i in range(state_probs.shape[1])]
    ax.stackplot(df.index, state_probs.T, labels=labels, colors=colors[:state_probs.shape[1]], alpha=0.75)
    ax.set_title("Regime State Probabilities Over Time", fontsize=13)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    plt.tight_layout()


def plot_volatility_vs_regime(features: pd.DataFrame, regimes: list[str]) -> None:
    """
    Scatter plot of daily returns vs rolling volatility, coloured by regime.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.scatterplot(
        x=features["volatility"],
        y=features["returns"],
        hue=regimes,
        palette=REGIME_COLORS,
        alpha=0.65,
        ax=ax,
    )
    ax.set_title("Returns vs Volatility by Regime", fontsize=13)
    ax.set_xlabel("Rolling Volatility")
    ax.set_ylabel("Log Returns")
    ax.legend(title="Regime")
    plt.tight_layout()
