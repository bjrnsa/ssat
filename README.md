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

## 📦 Installation

### Basic Installation
```bash
pip install ssat
```

### Full Installation (Recommended)
```bash
pip install ssat[all]
```

### Optional Dependencies
```bash
# Development and notebooks
pip install ssat[dev]

# Data collection tools
pip install ssat[data]

# Machine learning extensions
pip install ssat[ml]
```

## 🏃‍♂️ Quick Start

### Frequentist Models Example
```python
import pandas as pd
from ssat.frequentist import BradleyTerry, GSSD

# Load sample data
match_df = pd.read_parquet("ssat/data/sample_handball_match_data.parquet")

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
bt_preds = bt_model.predict(X_test)
bt_probas = bt_model.predict_proba(X_test, point_spread=0, include_draw=True)

# Get team ratings
team_ratings = bt_model.get_team_ratings()
print(team_ratings.head())
```

### Bayesian Models Example
```python
from ssat.bayesian import Poisson, Skellam

# Prepare data for Bayesian models
poisson_data = match_df[["home_team", "away_team", "home_goals", "away_goals"]]
skellam_data = match_df[["home_team", "away_team", "home_goals", "away_goals"]]
skellam_data["goal_diff"] = skellam_data["home_goals"] - skellam_data["away_goals"]

# Fit Bayesian models
poisson_model = Poisson()
poisson_model.fit(poisson_data, seed=42)

skellam_model = Skellam()
skellam_model.fit(skellam_data[["home_team", "away_team", "goal_diff"]], seed=42)

# Visualize model diagnostics
poisson_model.plot_trace()
poisson_model.plot_team_stats()

# Make predictions on new matches
new_matches = pd.DataFrame({
    "home_team": ["Aalborg", "GOG"],
    "away_team": ["Skjern", "Kolding"]
})

poisson_preds = poisson_model.predict(new_matches)
poisson_probas = poisson_model.predict_proba(new_matches)
```

## 📊 Model Overview

### Frequentist Models

| Model | Description | Best For |
|-------|-------------|----------|
| **Bradley-Terry** | Paired comparison with logistic regression | Team rankings, simple win probabilities |
| **GSSD** | Linear regression with offensive/defensive stats | Detailed team performance analysis |
| **TOOR** | Team offense-offense rating | Offensive performance comparison |
| **ZSD** | Zero-score distribution modeling | Low-scoring sports analysis |
| **PRP** | Possession-based rating process | Possession-heavy sports |
| **Poisson** | Goal-scoring as Poisson process | Classic sports modeling |

### Bayesian Models

| Model | Description | Best For |
|-------|-------------|----------|
| **Poisson** | Bayesian goal-scoring with MCMC | Uncertainty quantification in predictions |
| **NegBinom** | Overdispersed goal modeling | High-variance scoring patterns |
| **Skellam** | Direct goal difference modeling | Spread betting, draw analysis |
| **SkellamZero** | Zero-inflated Skellam | Sports with frequent draws |
| **Weighted variants** | Time-weighted model fitting | Recent performance emphasis |

## 📈 Example Notebooks

The package includes comprehensive example notebooks:

- `frequentist_example.py`: Complete frequentist model comparison with train-test evaluation
- `bayesian_example.py`: Bayesian model usage with MCMC diagnostics and visualization

Both examples use real handball data and demonstrate:
- Proper train-test splitting
- Model performance evaluation
- Prediction comparison and visualization
- Team strength analysis

## 🔧 Advanced Usage

### Model Benchmarking
```python
from ssat.benchmark import model_benchmark
from sklearn.metrics import mean_absolute_error

# Compare multiple models
models = [BradleyTerry(), GSSD()]
results = {}

for model in models:
    model.fit(X_train, y_train, Z_train)
    preds = model.predict(X_test)
    results[model.NAME] = mean_absolute_error(y_test, preds)

print("Model Performance (MAE):")
for model_name, mae in results.items():
    print(f"{model_name}: {mae:.3f}")
```

### Custom Team Analysis
```python
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
    'datetime': ['2024-01-01', '2024-01-02', ...]  # for time-based analysis
})
```

## 🛠️ Development

### Setup Development Environment
```bash
git clone https://github.com/bjrnsa/ssat.git
cd ssat
pip install -e ".[all]"
```

### Run Examples
```bash
# Frequentist models example
python ssat/notebooks/frequentist_example.py

# Bayesian models example
python ssat/notebooks/bayesian_example.py
```

### Testing
```bash
# Run basic functionality tests
python -c "import ssat; print('Import successful')"
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

### Optional Dependencies
- **scikit-learn**: Machine learning utilities
- **jupyter**: Interactive notebooks
- **flashscore-scraper**: Sports data collection



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
