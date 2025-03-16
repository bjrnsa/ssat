# %%
import os
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flashscore_scraper.data_loaders import Handball
from scipy import optimize
from scipy.stats import poisson as poisson_scipy
from scipy.stats import skellam as skellam_scipy

from ssat.bayesian.predictive_models import (
    Skellam,
    SkellamWeighted,
    SkellamZero,
    SkellamZeroWeighted,
)


def dixon_coles_func(dates, xi=0.0018, base_date=None, output_type="days_since_match"):
    """Calculates days since match based on the Dixon and Coles approach."""
    if base_date is None:
        base_date = max(dates)

    base_date = pd.Timestamp(base_date)

    # Calculate days since match
    diffs = np.array([(base_date - x).days if x < base_date else 0 for x in dates])

    if output_type == "days_since_match":
        return diffs
    elif output_type == "weights":
        # Calculate raw weights without normalization
        weights = np.exp(-xi * diffs / 365)
        return weights
    else:
        raise ValueError(f"Invalid output type: {output_type}")


db_path = Path(os.environ.get("DB_PATH", "database/database.db"))
loader = Handball(db_path=db_path)
loader_params = {
    # "league": "Herre Handbold Ligaen",
    # "country": "Europe",
    # "seasons": [2025],
    # "date_range": ("2022-01-01", "2025-03-16"),
    "include_additional_data": False,
}
df = loader.load_matches(**loader_params).sort_values("datetime")
df.set_index("flashscore_id", inplace=True)


last_train_date = "2024-11-11"

df = df.assign(
    weights=dixon_coles_func(
        df.datetime,
        xi=0.0,
        base_date=last_train_date,
        output_type="weights",
    ),
    days_since_match=dixon_coles_func(
        df.datetime,
        base_date=last_train_date,
        output_type="days_since_match",
    ),
    goal_diff_match=df.home_goals - df.away_goals,
)

feautres = ["home_team", "away_team", "goal_diff_match"]
train, test = (
    df.query("datetime < @last_train_date"),
    df.query("datetime >= @last_train_date"),
)

goals_ = train[["home_goals", "away_goals"]].values.flatten()
goal_diff_ = train["goal_diff_match"][train["goal_diff_match"].abs() < 30]


def poisson_model(params, data):
    mu = params
    loss = -np.sum(poisson_scipy.logpmf(data, mu))
    return loss


def skellam_model(params, data):
    mu1, mu2 = params
    loss = -np.sum(skellam_scipy.logpmf(data.values, mu1, mu2))
    return loss


def zero_inflated_skellam_model(params, data):
    mu1, mu2, gamma = params

    # Calculate regular Skellam log-probabilities
    skellam_logprob = skellam_scipy.logpmf(data.values, mu1, mu2)

    # Apply zero-inflation
    zero_mask = data.values == 0
    non_zero_mask = ~zero_mask

    # For zero values: log(gamma + (1-gamma)*p(0))
    zero_logprob = np.log(gamma + (1 - gamma) * np.exp(skellam_logprob[zero_mask]))

    # For non-zero values: log((1-gamma)*p(y))
    non_zero_logprob = np.log(1 - gamma) + skellam_logprob[non_zero_mask]

    # Combine log-probabilities
    total_logprob = np.sum(zero_logprob) + np.sum(non_zero_logprob)

    return -total_logprob


# Initial parameters
params0_poisson = [15]
params0_skellam = [15, 15]
params0_zero_inflated = [15, 15, 0.5]


# Optimize models
res_poisson = optimize.minimize(poisson_model, params0_poisson, args=(goals_))
res = optimize.minimize(skellam_model, params0_skellam, args=(goal_diff_))
res_zero_inflated = optimize.minimize(
    zero_inflated_skellam_model,
    params0_zero_inflated,
    args=(goal_diff_),
    bounds=[(0, None), (0, None), (0, 1)],
)

# Prepare data for plotting
value_counts = goal_diff_.value_counts(normalize=True).sort_index()
x_values = np.arange(value_counts.index.min(), value_counts.index.max() + 1)

# Calculate Skellam probabilities
mu1_skellam, mu2_skellam = res.x
skellam_probs = skellam_scipy.pmf(x_values, mu1_skellam, mu2_skellam)

# Calculate ZI-Skellam probabilities
zi_skellam_probs = np.zeros_like(x_values, dtype=float)
for i, x in enumerate(x_values):
    if x == 0:
        zi_skellam_probs[i] = (
            res_zero_inflated.x[2] + (1 - res_zero_inflated.x[2]) * skellam_probs[i]
        )
    else:
        zi_skellam_probs[i] = (1 - res_zero_inflated.x[2]) * skellam_probs[i]

# Plotting
plt.figure(figsize=(12, 8))

# Plot empirical distribution
empirical_values = [value_counts.get(x, 0) for x in x_values]
plt.bar(
    x_values, empirical_values, color="black", alpha=0.7, width=0.4, label="Empirical"
)

# Plot Skellam distribution
plt.bar(
    x_values + 0.2, skellam_probs, color="red", alpha=0.7, width=0.2, label="Skellam"
)

# Plot ZI-Skellam distribution
plt.bar(
    x_values + 0.4,
    zi_skellam_probs,
    color="green",
    alpha=0.7,
    width=0.2,
    label="ZI-Skellam",
)

# Set x-axis ticks to show all values
# plt.xticks(x_values)

# Labels and title
plt.xlabel("Goal Difference")
plt.ylabel("Density")
plt.title("Skellam Fit on Data")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot fitted vs empirical poisson
value_counts_poisson = pd.Series(goals_).value_counts(normalize=True).sort_index()
x_values_poisson = np.arange(
    value_counts_poisson.index.min(), value_counts_poisson.index.max() + 1
)

poisson_empirical_values = np.array(
    [value_counts_poisson.get(x, 0) for x in x_values_poisson]
)
mu_poisson = res_poisson.x[0]
poisson_probs = poisson_scipy.pmf(x_values_poisson, mu_poisson)

plt.figure(figsize=(12, 8))
plt.bar(
    x_values_poisson,
    poisson_empirical_values,
    color="black",
    alpha=0.7,
    width=0.4,
    label="Empirical",
)
plt.bar(
    x_values_poisson + 0.5,
    poisson_probs,
    color="blue",
    alpha=0.7,
    width=0.2,
    label="Poisson",
)

plt.xlabel("Goals")
plt.ylabel("Density")
plt.title("Poisson Fit on Data")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

train_outlier_free = train[train["goal_diff_match"].abs() < 15]
# %%
skellam = Skellam()
skellam.fit(
    train_outlier_free[feautres],
    train_outlier_free["weights"],
    seed=2,
)
skellam.plot_trace(
    var_names=[
        "intercept",
        "home_advantage",
        "attack_team",
        "defence_team",
        "tau",
        "sigma",
    ]
)
skellam.plot_team_stats()


skellam_weighted = SkellamWeighted()
skellam_weighted.fit(
    train_outlier_free[feautres], train_outlier_free["days_since_match"], seed=3
)
skellam_weighted.plot_trace(
    var_names=[
        "xi",
        "xi_logit",
        "home_advantage",
        "intercept",
        "tau",
        "sigma",
        "attack_team",
        "defence_team",
    ]
)
skellam_weighted.plot_team_stats()


skellam_zero = SkellamZero()
skellam_zero.fit(
    train_outlier_free[feautres],
    train_outlier_free["weights"],
    seed=2,
)
skellam_zero.plot_trace(
    var_names=[
        "zi",
        "home_advantage",
        "intercept",
        "tau",
        "sigma",
        "attack_team",
        "defence_team",
    ]
)
skellam_zero.plot_team_stats()


skellam_zero_weighted = SkellamZeroWeighted()
skellam_zero_weighted.fit(
    train_outlier_free[feautres],
    train_outlier_free["days_since_match"],
    seed=326,
)
skellam_zero_weighted.plot_trace(
    var_names=[
        "xi",
        "xi_logit",
        "zi",
        "home_advantage",
        "intercept",
        "tau",
        "sigma",
        "attack_team",
        "defence_team",
    ]
)
skellam_zero_weighted.plot_team_stats()
# %%


model_comparison = az.compare(
    {
        "skellam": skellam.inference_data,
        "skellam_weighted": skellam_weighted.inference_data,
        "skellam_zero": skellam_zero.inference_data,
        "skellam_zero_weighted": skellam_zero_weighted.inference_data,
    }
)
print(model_comparison)
# Extract predictions from each model
pred_goal_diff1 = skellam.predict(
    train_outlier_free[feautres],
    sampling_method="qskellam",
    func="median",
)
pred_goal_diff2 = skellam_weighted.predict(
    train_outlier_free[feautres],
    sampling_method="qskellam",
    func="median",
)
pred_goal_diff3 = skellam_zero.predict(
    train_outlier_free[feautres],
    sampling_method="qskellam",
    func="median",
)
pred_goal_diff4 = skellam_zero_weighted.predict(
    train_outlier_free[feautres],
    sampling_method="qskellam",
    func="median",
)

# Get actual goal differences
actual_goal_diff = np.array(train_outlier_free["goal_diff_match"])

# Create posterior predictive plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Plot histograms for each model
for ax, pred, title in zip(
    axes,
    [pred_goal_diff1, pred_goal_diff2, pred_goal_diff3, pred_goal_diff4],
    [
        "Basic Skellam",
        "Time-weighted Skellam",
        "Zero-inflated Skellam",
        "Zero-inflated Time-weighted",
    ],
):
    # Flatten predictions across all samples
    pred_flat = pred.values.flatten()

    # Plot histogram of predictions
    ax.hist(
        pred_flat,
        bins=range(
            min(min(pred_flat), min(actual_goal_diff)) - 1,
            max(max(pred_flat), max(actual_goal_diff)) + 2,
        ),
        alpha=0.5,
        label="Predicted",
    )

    # Plot histogram of actual data
    ax.hist(
        actual_goal_diff,
        bins=range(min(actual_goal_diff) - 1, max(actual_goal_diff) + 2),
        alpha=0.5,
        label="Actual",
    )

    # Add vertical line at zero
    ax.axvline(x=0, color="r", linestyle="--")

    # Count zeros in predictions and actual data
    zeros_pred = np.sum(pred_flat == 0) / len(pred_flat)
    zeros_actual = np.sum(actual_goal_diff == 0) / len(actual_goal_diff)

    ax.set_title(f"{title}\nZeros: Pred={zeros_pred:.2f}, Actual={zeros_actual:.2f}")
    ax.legend()

plt.tight_layout()
plt.show()


# %%
# %%
