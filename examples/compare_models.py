# %%
"""Model Comparison for Handball Match Prediction
Comparison of various Bayesian and frequentist models for sports outcome prediction
"""

import numpy as np
import pandas as pd

from ssat.bayesian import (
    NegBinom,
    Poisson,
    PoissonDecay,
    Skellam,
    SkellamDecay,
    SkellamZero,
)
from ssat.data import get_sample_handball_match_data
from ssat.frequentist import GSSD, PRP, TOOR, ZSD, BradleyTerry
from ssat.metrics import (
    average_rps,
    balanced_accuracy,
    calibration_error,
    ignorance_score,
    multiclass_brier_score,
    multiclass_log_loss,
)
from ssat.utils import dixon_coles_weights

# %%
# Configuration
np.random.seed(42)
LEAGUE = "Starligue"
SEASONS = [2024, 2025]
TRAIN_SPLIT = 0.8

# %%
# Load and filter data
df = get_sample_handball_match_data()
print(f"Available leagues: {list(df.league.unique())}")

match_df = df.loc[(df["league"] == LEAGUE) & (df["season"].isin(SEASONS))]
print(f"Dataset size: {len(match_df)} matches")

# %%
# Data preparation
goal_diff = match_df["home_goals"] - match_df["away_goals"]
outcomes = np.sign(goal_diff).replace(
    {-1: 2, 0: 1, 1: 0}
)  # 0=Home win, 1=Draw, 2=Away win

X = match_df[["home_team", "away_team"]]
Z = match_df[["home_goals", "away_goals"]]
y = goal_diff
dt = match_df["datetime"]

# Train-test split
train_size = int(len(match_df) * TRAIN_SPLIT)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
Z_train, Z_test = Z[:train_size], Z[train_size:]
dt_train, dt_test = dt[:train_size], dt[train_size:]
outcomes_test = outcomes[train_size:]

weights_train = dixon_coles_weights(dt_train)

print(f"Training set: {len(X_train)} matches")
print(f"Test set: {len(X_test)} matches")

# %%
# Initialize models
models = [
    ("Bradley-Terry", BradleyTerry()),
    ("PRP", PRP()),
    ("GSSD", GSSD()),
    ("TOOR", TOOR()),
    ("ZSD", ZSD()),
    ("Poisson", Poisson()),
    ("Negative Binomial", NegBinom()),
    ("Skellam", Skellam()),
    ("Skellam Zero", SkellamZero()),
    ("Skellam Decay", SkellamDecay()),
    ("Poisson Decay", PoissonDecay()),
]

# %%
# Model evaluation
results = []

for name, model in models:
    print(f"Training {name}...")

    try:
        model.fit(X=X_train, y=y_train, Z=Z_train, weights=weights_train)
    except (
        Exception
    ):  # Poisson and Negative Binomial fits on home and away goals separately
        model.fit(X=X_train, y=Z_train, Z=Z_train, weights=weights_train)

    preds_proba = model.predict_proba(X_test)

    # Calculate metrics
    metrics = {
        "Model": name,
        "Brier Score": multiclass_brier_score(outcomes_test, preds_proba),
        "Log Loss": multiclass_log_loss(outcomes_test, preds_proba),
        "RPS": average_rps(outcomes_test, preds_proba),
        "Calibration Error": calibration_error(outcomes_test, preds_proba),
        "Ignorance Score": ignorance_score(outcomes_test, preds_proba),
        "Balanced Accuracy": balanced_accuracy(outcomes_test, preds_proba),
    }
    results.append(metrics)

# %%
# Display results
results_df = pd.DataFrame(results).set_index("Model")
print("\nModel Performance Comparison:")
print("=" * 50)
results_df.round(4)

# %%
# Performance ranking (lower is better for most metrics, higher for accuracy)
ranking_metrics = [
    "Brier Score",
    "Log Loss",
    "RPS",
    "Calibration Error",
    "Ignorance Score",
]
accuracy_metrics = ["Balanced Accuracy"]

print("\nTop 3 Models by Metric:")
print("=" * 30)

for metric in ranking_metrics:
    top_3 = results_df[metric].nsmallest(3)
    print(f"\n{metric}:")
    for i, (model, score) in enumerate(top_3.items(), 1):
        print(f"  {i}. {model}: {score:.4f}")

for metric in accuracy_metrics:
    top_3 = results_df[metric].nlargest(3)
    print(f"\n{metric}:")
    for i, (model, score) in enumerate(top_3.items(), 1):
        print(f"  {i}. {model}: {score:.4f}")

# %%
