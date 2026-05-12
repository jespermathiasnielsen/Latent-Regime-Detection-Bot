# Latent Market Regime Detection Bot

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Domain](https://img.shields.io/badge/Domain-Regime%20Detection%20%7C%20HMM-FF6F00?style=flat)

> Detect hidden market regimes — bull, bear, and sideways — from price data using a Gaussian Hidden Markov Model.

---

## Overview

Financial markets shift between structural states that are not directly observable. Prices in a bull regime have different return and volatility characteristics than prices in a bear or sideways regime — but the regime itself is latent (hidden).

This system learns those latent states from log returns, rolling volatility, and momentum, then labels each period and outputs regime statistics, transition probabilities, and three diagnostic visualisations.

---

## Features

- Gaussian HMM with configurable number of hidden states
- Three engineered features: log returns, rolling volatility, momentum
- Automatic regime labelling: bull · bear · sideways
- Regime statistics: mean return, mean vol, average duration, and transition matrix
- Accepts real CSV price data (`Date`, `Close` columns) or generates synthetic data
- CLI arguments for data path, number of states, and rolling window size
- Three diagnostic visualisations in a single figure

---

## Installation

```bash
git clone https://github.com/jespermathiasnielsen/Latent-Regime-Detection-Bot.git
cd Latent-Regime-Detection-Bot
pip install -r requirements.txt
```

**Requirements:** `numpy`, `pandas`, `hmmlearn`, `matplotlib`, `scikit-learn`

---

## Usage

```bash
# Synthetic data, 3 regimes (default)
python main.py

# Real price data
python main.py --data prices.csv

# Custom configuration
python main.py --data prices.csv --states 3 --window 20
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | `None` | Path to CSV with `Date` and `Close` columns |
| `--states` | `3` | Number of hidden states (regimes) |
| `--window` | `20` | Rolling window size for volatility and momentum |

### Input CSV Format

```
Date,Close
2020-01-01,3230.78
2020-01-02,3257.85
...
```

---

## How It Works

### Feature Engineering

Three features are computed from the raw price series:

| Feature | Computation | Role |
|---|---|---|
| Log returns | `ln(P_t / P_{t-1})` | Captures return magnitude and direction |
| Rolling volatility | Std of log returns over `window` periods | Distinguishes high/low volatility regimes |
| Momentum | Return over `window` periods | Captures trend direction |

### Hidden Markov Model

A Gaussian HMM assumes:
- The market occupies one of `K` hidden states at each timestep
- Each state emits observations drawn from a multivariate Gaussian
- The model transitions between states according to a learned transition matrix

Training fits the emission parameters (means and covariances per state) and the transition matrix to maximise the likelihood of the observed feature sequence via the Baum-Welch algorithm (EM).

### Regime Labelling

After fitting, each hidden state is mapped to a regime label based on the mean return and volatility of its emission distribution:
- **Bull:** high mean return, moderate volatility
- **Bear:** negative or low mean return, elevated volatility
- **Sideways:** near-zero mean return, low volatility

---

## Project Structure

```
main.py          # Entry point with CLI argument parsing
data.py          # CSV loader + synthetic data generator
features.py      # Log returns, rolling volatility, momentum
model.py         # HMM fitting, regime labelling, statistics
visualization.py # Three-chart diagnostic output
requirements.txt
README.md
```

---

## Output

Three-panel figure plus printed regime statistics:

| Panel | Content |
|---|---|
| Price + regime shading | Green (bull) · red (bear) · grey (sideways) background spans |
| Regime probability stack | Posterior probability of each state over time |
| Returns vs volatility | Scatter coloured by discovered regime cluster |

Terminal output includes the transition matrix and per-regime statistics table.

---

## References

- [Hidden Markov Model — Wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model)
- [Baum-Welch algorithm — Wikipedia](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm)
- [Market trend — Wikipedia](https://en.wikipedia.org/wiki/Market_trend)
- [hmmlearn documentation](https://hmmlearn.readthedocs.io)

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request. Ensure new code includes docstrings and follows the existing module structure.

---

## License

MIT — see [LICENSE](LICENSE) for details.

**Author:** [Jesper Mathias Nielsen](https://github.com/jespermathiasnielsen)
