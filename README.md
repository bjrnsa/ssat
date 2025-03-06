# SSAT: Statistical Sports Analysis Toolkit

[![Python 3.10+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PyPI](https://img.shields.io/pypi/v/ssat)](https://pypi.org/project/ssat/)

SSAT is a Python package implementing statistical models for sports analytics. The package provides a collection of frequentist statistical models for analyzing and predicting sports match outcomes.

## Key Features

- **Multiple Statistical Models**:
  - Bradley-Terry Model: Paired comparison model for team rankings
  - TOOR (Team Offense-Offense Rating): Offensive performance analysis
  - GSSD (Goal Scoring Statistical Distribution): Goal distribution modeling
  - ZSD (Zero-Score Distribution): Special case handling for 0-0 outcomes
  - PRP (Possession-based Rating Process): Team rating based on possession metrics
  - Poisson Model: Classic goal-scoring probability distribution

- **Data Processing**: Integrated with flashscore-scraper for automated data collection
- **Visualization**: Comprehensive plotting utilities for model analysis
- **Model Comparison**: Tools for comparing predictions across different models

## Installation

```bash
pip install ssat
```

For full functionality including all optional dependencies:
```bash
pip install ssat[all]
```

### Dependencies

- Core: numpy, pandas, scipy
- Optional:
  - Development: ipykernel, ipywidgets, jupyter
  - Visualization: matplotlib, seaborn
  - Data Collection: flashscore-scraper, requests, beautifulsoup4
  - Machine Learning: scikit-learn, statsmodels
  - Bayesian (planned): arviz, cmdstanpy

## Quick Start

```python
import pandas as pd
from ssat.frequentist import BradleyTerry, Poisson

# Load data
df = pd.read_pickle("ssat/data/sample_handball_data.pkl")
X = df[["home_team", "away_team"]]
Z = df[["home_goals", "away_goals"]]
y = df["spread"]

# Initialize and fit models
bt_model = BradleyTerry()
poisson_model = Poisson()

# Fit models
bt_model.fit(X, y, Z)
poisson_model.fit(X, y, Z)

# Make predictions
bt_predictions = bt_model.predict(X)
poisson_predictions = poisson_model.predict(X)

# Predict probabilities
bt_probas = bt_model.predict_proba(X, Z, point_spread=0, include_draw=True)
poisson_probas = poisson_model.predict_proba(X, Z, point_spread=0, include_draw=True)
```

## Data Sources

Match data is collected using the [flashscore-scraper](https://github.com/flashscore/flashscore-scraper) package. The package includes sample handball data in `ssat/data/sample_handball_data.pkl` for testing and examples.

## API Documentation

### Base Model
All models inherit from `BaseModel` providing common functionality:
- `fit(X, y, Z)`: Fit the model to training data
- `predict(X)`: Predict match outcomes
- `predict_proba(X, Z, point_spread, include_draw)`: Predict outcome probabilities

### Specific Models

#### Bradley-Terry Model
```python
from ssat.frequentist import BradleyTerry

model = BradleyTerry()
model.fit(X, y, Z)
```
Implements paired comparison modeling for team strength estimation.

#### Poisson Model
```python
from ssat.frequentist import Poisson

model = Poisson()
model.fit(X, y, Z)
```
Models goal-scoring as a Poisson process.

[Additional model documentation available in the wiki]

## Development Roadmap

1. **Current Release (v0.0.1)**:
   - Frequentist models implementation
   - Basic data processing utilities
   - Example notebooks

2. **Upcoming Features**:
   - Bayesian implementations using Stan
   - Enhanced visualization tools
   - Additional sport-specific models
   - Performance optimization

3. **Future Plans**:
   - Real-time prediction updates
   - Web API integration
   - Additional sports support

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/bjrnsa/ssat.git
cd ssat
pip install -e ".[all]"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use SSAT in your research, please cite:

```bibtex
@software{ssat2025,
  author = {Aagaard, Bjørn},
  title = {SSAT: Statistical Sports Analysis Toolkit},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/bjrnsa/ssat}
}
```

## Acknowledgments

- Andrew Mack's "Statistical Sports Models in Excel" (ISBN: 978-1079013450)
- Contributors and maintainers of dependent packages
