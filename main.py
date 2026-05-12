"""
Latent Market Regime Detection — entry point.

Detects hidden market regimes (bull, bear, sideways) using a Gaussian HMM
and visualises the results with price overlays, probability stacks, and
volatility scatter plots.

Author: Jesper Mathias Nielsen
Usage:
    python main.py                       # uses synthetic data
    python main.py --data path/to/data.csv
"""

import argparse
import matplotlib.pyplot as plt

from data import load_price_data, generate_synthetic_data
from features import compute_features
from model import fit_hmm, map_regimes, regime_statistics
from visualization import (
    plot_price_with_regimes,
    plot_regime_probabilities,
    plot_volatility_vs_regime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent Market Regime Detector")
    parser.add_argument("--data",    type=str,  default=None,  help="Path to CSV file (Date, Close)")
    parser.add_argument("--states",  type=int,  default=3,     help="Number of HMM states (default: 3)")
    parser.add_argument("--window",  type=int,  default=20,    help="Feature rolling window in days (default: 20)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.data:
        print(f"Loading data from {args.data} …")
        df = load_price_data(args.data)
    else:
        print("No data file supplied — using synthetic price series …")
        df = generate_synthetic_data(n=500)

    # ── Feature engineering ───────────────────────────────────────────────────
    features = compute_features(df, window=args.window)
    df = df.loc[features.index]

    # ── Fit HMM ───────────────────────────────────────────────────────────────
    print(f"Fitting Gaussian HMM with {args.states} states …")
    hidden_states, state_probs, transmat, model = fit_hmm(features, n_states=args.states)
    regimes, regime_map = map_regimes(features, hidden_states)
    df["regime"] = regimes

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\nState → Regime mapping: {regime_map}")
    regime_statistics(features, regimes, transmat)

    # ── Visualise ─────────────────────────────────────────────────────────────
    plot_price_with_regimes(df, regimes)
    plot_regime_probabilities(df, state_probs)
    plot_volatility_vs_regime(features, regimes)
    plt.show()


if __name__ == "__main__":
    main()
