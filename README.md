# Latent Market Regime Detection

**Author: [Jesper Mathias Nielsen](https://github.com/jespermathiasnielsen)**

Detects hidden market regimes — **bull, bear, and sideways** — from price data using a Gaussian Hidden Markov Model (HMM). Includes regime statistics, transition probabilities, and three diagnostic visualisations.

## What it does

Financial markets shift between structural states that are not directly observable. This system learns those latent states from three features — log returns, rolling volatility, and momentum — then labels each period as bull, bear, or sideways based on the statistical properties of each discovered cluster.

## Features

- Gaussian HMM with configurable number of states
- Feature engineering: log returns, rolling volatility, momentum
- Automatic regime labelling (bull / bear / sideways)
- Regime statistics: average duration, mean return/vol per regime, transition matrix
- Works with real CSV price data or built-in synthetic data for testing
- CLI arguments for data path, number of states, and rolling window

## Setup

```bash
git clone https://github.com/jespermathiasnielsen/Latent-Regime-Detection-Bot.git
cd Latent-Regime-Detection-Bot
pip install -r requirements.txt

# Run with synthetic data
python main.py

# Run with your own CSV (columns: Date, Close)
python main.py --data prices.csv --states 3 --window 20
```

## Visualisations

- **Price chart with regime shading** — green (bull), red (bear), grey (sideways) background spans
- **Regime probability stack** — shows confidence of each state over time
- **Returns vs Volatility scatter** — coloured by regime cluster

## Project structure

```
main.py          # Entry point with CLI argument parsing
data.py          # CSV loader + synthetic data generator
features.py      # Log returns, rolling vol, momentum
model.py         # HMM fitting, regime labelling, statistics
visualization.py # Three-chart output suite
```

## Concepts

- [Hidden Markov Model](https://en.wikipedia.org/wiki/Hidden_Markov_model)
- [Market regime](https://en.wikipedia.org/wiki/Market_trend)
- [Implied volatility](https://en.wikipedia.org/wiki/Implied_volatility)

## License

MIT
