# SSAT: Statistical Sports Analysis Toolkit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PyPI](https://img.shields.io/pypi/v/ssat)](https://pypi.org/project/ssat/)

SSAT is a comprehensive Python package for statistical sports analysis, providing both frequentist and Bayesian statistical models for analyzing and predicting sports match outcomes. The package is particularly focused on handball but can be adapted for other sports.

## 🚀 Key Features

### Statistical Models

- **Frequentist Models**: Bradley-Terry, GSSD, TOOR, ZSD, PRP, Poisson
- **Bayesian Models**: Poisson, Negative Binomial, Skellam variants with MCMC sampling
- **Model Comparison**: Built-in tools for comparing predictions across different approaches

### Analysis Capabilities

- **Team Ratings**: Detailed offensive/defensive capabilities analysis
- **Match Prediction**: Win/Draw/Loss probabilities with uncertainty quantification
- **Performance Evaluation**: Comprehensive model benchmarking and validation
- **Visualization**: Rich plotting utilities for model diagnostics and team analysis

### Data Integration

- **Sample Data**: Included handball datasets for immediate experimentation
- **Flexible Input**: Support for various data formats and structures
- **Extensible**: Easy integration with external data sources

## ✅ To-Do

- ### Streamlined fit and predic for both types of models

- ### Write documentation to show features like adding weights

## 📦 Installation

### Installation

```bash
pip install ssat
```

### `cmdStan` Intallation

To use the Bayesian models you will need to install `cmdStan` as described in the [`cmdStan` Installation Guide](https://mc-stan.org/cmdstanpy/installation.html#cmdstan-installation).

## 🏃‍♂️ Quick Start

### Frequentist Models Example

```python
import pandas as pd
from ssat.data import get_sample_handball_match_data
from ssat.frequentist import GSSD, BradleyTerry

# Load sample data
df = get_sample_handball_match_data()
league = "Starligue"
season = 2024
match_df = df.loc[(df["league"] == league) & (df["season"] == season)]

# Prepare data
X = match_df[["home_team", "away_team"]]
y = match_df["home_goals"] - match_df["away_goals"]  # spread
Z = match_df[["home_goals", "away_goals"]]

# Train-test split
train_size = int(len(match_df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
Z_train, Z_test = Z[:train_size], Z[train_size:]

# Fit models
bt_model = BradleyTerry()
bt_model.fit(X_train, y_train, Z_train)

gssd_model = GSSD()
gssd_model.fit(X_train, y_train, Z_train)

# Make predictions
test_fixtures = X_test.apply(lambda x: f"{x.home_team}-{x.away_team}", axis=1)

bt_probas = bt_model.predict_proba(X_test, point_spread=0, include_draw=True)
gssd_probas = gssd_model.predict_proba(X_test, point_spread=0, include_draw=True)

bt_probas_df = pd.DataFrame(
    bt_probas, columns=["Home", "Draw", "Away"], index=test_fixtures
)
gssd_probas_df = pd.DataFrame(
    gssd_probas, columns=["Home", "Draw", "Away"], index=test_fixtures
)
print(bt_probas_df.head())
print(gssd_probas_df.head())

# Get team ratings
bt_team_ratings = bt_model.get_team_ratings()
print(bt_team_ratings.head())

gssd_team_ratings = gssd_model.get_team_ratings()
print(gssd_team_ratings.head())
```

### Bayesian Models Example

```python
import pandas as pd
from ssat.bayesian import Poisson, Skellam
from ssat.data import get_sample_handball_match_data

# Load sample data
df = get_sample_handball_match_data()
league = "Starligue"
season = 2024
match_df = df.loc[(df["league"] == league) & (df["season"] == season)]

# Prepare data
X = match_df[["home_team", "away_team", "home_goals", "away_goals"]]
X = X.assign(goal_diff=X["home_goals"] - X["away_goals"])

# Train-test split
train_size = int(len(match_df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]

# Fit Bayesian models
poisson_model = Poisson()
poisson_model.fit(X_train, seed=42)

skellam_model = Skellam()
skellam_model.fit(X_train[["home_team", "away_team", "goal_diff"]], seed=42)

# Visualize model diagnostics
poisson_model.plot_trace()
poisson_model.plot_team_stats()

skellam_model.plot_trace()
skellam_model.plot_team_stats()

# Make predictions on new matches
test_fixtures = X_test.apply(lambda x: f"{x.home_team}-{x.away_team}", axis=1)

poisson_preds = poisson_model.predict(X_test)
poisson_probas = poisson_model.predict_proba(X_test)
poisson_probas.index = test_fixtures

skellam_preds = skellam_model.predict(X_test)
skellam_probas = skellam_model.predict_proba(X_test)
skellam_probas.index = test_fixtures

# Print results - notice how the Skellam assign a higher probability to draws
print(poisson_probas.head())
print(skellam_probas.head())
```

## 📊 Model Overview

### Frequentist Models

| Model | Description |
|-------|-------------|
| **Bradley-Terry** | Paired comparison with logistic regression |
| **GSSD** | Linear regression with offensive/defensive stats |
| **TOOR** | Team offense-offense rating |
| **ZSD** | Zero-score distribution modeling |
| **PRP** | Possession-based rating process |
| **Poisson** | Goal-scoring as Poisson process |

### Bayesian Models

| Model | Description |
|-------|-------------|
| **Poisson** | Bayesian goal-scoring with MCMC |
| **NegBinom** | Overdispersed goal modeling |
| **Skellam** | Direct goal difference modeling |
| **SkellamZero** | Zero-inflated Skellam |
| **Weighted variants** | Time-weighted model fitting |

## 📈 Example Notebooks

The repository contains comprehensive example notebooks:

- [`frequentist_example.ipynb`](ssat/notebooks/frequentist_example.ipynb): Complete frequentist model comparison with train-test evaluation
- [`bayesian_example.ipynb`](ssat/notebooks/bayesian_example.ipynb): Bayesian model usage with MCMC diagnostics and visualization

Both examples use real handball data and demonstrate:

- Proper train-test splitting
- Model performance evaluation
- Prediction comparison and visualization
- Team strength analysis

## 🔧 Other Usage

### Model Benchmarking

```python
import numpy as np
import pandas as pd
from ssat.data import get_sample_handball_match_data
from ssat.frequentist import BradleyTerry, GSSD

# Load sample data
df = get_sample_handball_match_data()
league = "Starligue"
season = 2024
match_df = df.loc[(df["league"] == league) & (df["season"] == season)]

# Prepare data
X = match_df[["home_team", "away_team"]]
y = match_df["home_goals"] - match_df["away_goals"]  # spread
Z = match_df[["home_goals", "away_goals"]]

# Train-test split
train_size = int(len(match_df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
Z_train, Z_test = Z[:train_size], Z[train_size:]

# Compare multiple models
models = [BradleyTerry(), GSSD()]
results = {}

for model in models:
    model.fit(X_train, y_train, Z_train)
    preds = model.predict(X_test)
    results[model.NAME] = np.mean(np.abs(preds - y_test))

print("Model Performance (MAE):")
for model_name, mae in results.items():
    print(f"{model_name}: {mae:.3f}")
```

### Custom Team Analysis

```python
import pandas as pd
from ssat.data import get_sample_handball_match_data
from ssat.frequentist import BradleyTerry, GSSD

# Load sample data
df = get_sample_handball_match_data()
league = "Starligue"
season = 2024
match_df = df.loc[(df["league"] == league) & (df["season"] == season)]


# Prepare data
X = match_df[["home_team", "away_team"]]
y = match_df["home_goals"] - match_df["away_goals"]  # spread
Z = match_df[["home_goals", "away_goals"]]

# Train-test split
train_size = int(len(match_df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
Z_train, Z_test = Z[:train_size], Z[train_size:]

# Fit Model
gssd_model = GSSD()
gssd_model.fit(X_train, y_train, Z_train)

# Detailed team strength analysis
team_stats = gssd_model.get_team_ratings()
print("Team Offensive/Defensive Breakdown:")
print(team_stats[['pfh', 'pah', 'pfa', 'paa']].head())

# Model coefficients
coeffs = team_stats.loc['Coefficients']
print(f"Home offense coefficient: {coeffs['pfh']:.3f}")
print(f"Home defense coefficient: {coeffs['pah']:.3f}")
```

## 📊 Data Format

SSAT expects data in the following format:

```python
# Required columns for match data
match_data = pd.DataFrame({
    'home_team': ['Team A', 'Team B', ...],
    'away_team': ['Team B', 'Team C', ...],
    'home_goals': [25, 30, ...],
    'away_goals': [23, 28, ...],
})
```

## 🛠️ Development

### Setup Development Environment

```bash
git clone https://github.com/bjrnsa/ssat.git
cd ssat
# Create and activate your virtual environment
pip install -e .
```

### Run Examples or check out the rendered notebooks in the `ssat/notebooks` folder

```bash
# Frequentist models example
python ssat/notebooks/frequentist_example.py

# Bayesian models example
python ssat/notebooks/bayesian_example.py
```

## 📝 Dependencies

### Core Dependencies

- **arviz**: Bayesian model diagnostics
- **cmdstanpy**: Stan interface for MCMC sampling
- **matplotlib**: Plotting and visualization
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **pyarrow**: Efficient data storage
- **scipy**: Statistical functions
- **seaborn**: Statistical visualization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use SSAT in your research, please cite:

```bibtex
@software{ssat2025,
  author = {Aagaard, Bjørn},
  title = {SSAT: Statistical Sports Analysis Toolkit},
  version = {0.0.3},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/bjrnsa/ssat}
}
```

## 🙏 Acknowledgments

- Statistical modeling concepts from Andrew Mack's "Statistical Sports Models in Excel"
- The Stan development team for excellent MCMC tools
- The scientific Python ecosystem contributors
