# %%
"""Bayesian Poisson Model for sports prediction."""

import numpy as np

from ssat.bayesian.nbinomial_models import NegBinom, NegBinomDecay
from ssat.bayesian.poisson_models import Poisson, PoissonDecay
from ssat.bayesian.skellam_models import (
    Skellam,
    SkellamDecay,
    SkellamZero,
    SkellamZeroDecay,
)

# %%
if __name__ == "__main__":
    from ssat.data import get_sample_handball_match_data

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load sample data
    df = get_sample_handball_match_data()
    league = "Starligue"
    season = 2024
    match_df = df.loc[(df["league"] == league) & (df["season"] == season)]
    goal_diff = match_df["home_goals"] - match_df["away_goals"]
    dt = match_df["datetime"]

    # Prepare data
    X = match_df[["home_team", "away_team"]]
    y = match_df[["home_goals", "away_goals"]].assign(
        goal_diff=lambda x: x["home_goals"] - x["away_goals"]
    )
    weights = np.random.normal(1, 0.1, len(match_df))

    # Train-test split
    train_size = int(len(match_df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    weights_train, weights_test = weights[:train_size], weights[train_size:]
    dt_train, dt_test = dt[:train_size], dt[train_size:]

    # Days since last match
    Z_train = (dt_train.max() - dt_train).dt.days.astype(int)
    Z_test = (dt_test.max() - dt_test).dt.days.astype(int)

    # instantiate all models
    models = [
        Poisson(),
        PoissonDecay(),
        NegBinom(),
        NegBinomDecay(),
        Skellam(),
        SkellamDecay(),
        SkellamZero(),
        SkellamZeroDecay(),
    ]
    # Fit model
    for model in models:
        print(model)
        name = model.__class__.__name__
        if "Skellam" in name:
            y_train_temp = y_train["goal_diff"]
        else:
            y_train_temp = y_train[["home_goals", "away_goals"]]

        if "Decay" in model.__class__.__name__:
            model.fit(X_train, y_train_temp, Z_train)
        else:
            model.fit(X_train, y_train_temp, weights=weights_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)

        model.plot_trace()
        model.plot_team_stats()
        # Days since last match

# %%
