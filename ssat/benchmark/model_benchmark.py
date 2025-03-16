# %%
import os
from pathlib import Path
from typing import Dict, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flashscore_scraper.data_loaders import Handball
from scipy import optimize
from scipy.stats import skellam as skellam_scipy
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
)

from ssat.bayesian.predictive_models import (
    Skellam,
    SkellamWeighted,
    SkellamZero,
    SkellamZeroWeighted,
)
from ssat.odds.implied import ImpliedOdds


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
    "league": "Herre Handbold Ligaen",
    # "country": "Europe",
    # "seasons": [2025],
    # "date_range": ("2022-01-01", "2025-03-16"),
    "include_additional_data": False,
}
df = loader.load_matches(**loader_params).sort_values("datetime")
df.set_index("flashscore_id", inplace=True)

# Remove Lemvig, Skive, Midtjylland
teams_to_remove = ["Lemvig", "Skive", "Midtjylland"]
df = df.query("home_team not in @teams_to_remove and away_team not in @teams_to_remove")

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


odds_df = loader.load_odds(df.index.tolist())
odds_df = odds_df.groupby("flashscore_id")[
    ["home_odds", "draw_odds", "away_odds"]
].max()
odds_df = odds_df.rename(
    columns={
        "home_odds": "home",
        "draw_odds": "draw",
        "away_odds": "away",
    }
)

implied = ImpliedOdds(["power"])
implied_odds = implied.get_implied_probabilities(odds_df)
implied_odds = implied_odds.pivot_table(
    index="match_id", columns="outcome", values="power"
)[["home", "draw", "away"]]


# Remove teams in test that are not present in train
test = test[test.home_team.isin(train.home_team) & test.away_team.isin(train.away_team)]
test_odds = odds_df.query("flashscore_id in @test.index")
test_implied_odds = implied_odds.query("match_id in @test.index")


predictions = {}
probas = {}

goal_diff_ = train["goal_diff_match"][train["goal_diff_match"].abs() < 15]


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
params0 = [15, 15]
params0_zero_inflated = [15, 15, 0.5]

skellam_model(params0, goal_diff_)

# Optimize models
res = optimize.minimize(skellam_model, params0, args=(goal_diff_))
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
func = "median"
sampling_method = "qskellam"

predictions[skellam.name] = skellam.predict(
    test[feautres], sampling_method=sampling_method, func=func
)
probas[skellam.name] = skellam.predict_proba(
    test[feautres], sampling_method=sampling_method, func=func
)

predictions[skellam_weighted.name] = skellam_weighted.predict(
    test[feautres], sampling_method=sampling_method, func=func
)
probas[skellam_weighted.name] = skellam_weighted.predict_proba(
    test[feautres], sampling_method=sampling_method, func=func
)

predictions[skellam_zero.name] = skellam_zero.predict(
    test[feautres], sampling_method=sampling_method, func=func
)
probas[skellam_zero.name] = skellam_zero.predict_proba(
    test[feautres], sampling_method=sampling_method, func=func
)
predictions[skellam_zero_weighted.name] = skellam_zero_weighted.predict(
    test[feautres], sampling_method=sampling_method, func=func
)
probas[skellam_zero_weighted.name] = skellam_zero_weighted.predict_proba(
    test[feautres], sampling_method=sampling_method, func=func
)

y_true_home = test["home_goals"]
y_true_away = test["away_goals"]
y_true_flat = test[["home_goals", "away_goals"]].values.flatten()
y_true_spread = test["home_goals"] - test["away_goals"]
y_true_outcome = np.sign(y_true_spread)


# %%
def evaluate_predictions(
    predictions: pd.Series,
    actuals: pd.Series,
) -> Dict[str, float]:
    """Evaluate spread predictions against actual results."""
    # Convert actuals to DataFrame if it's a Series

    # Calculate basic metrics
    results = []
    results.append(mean_absolute_error(actuals, predictions))
    results.append(mean_squared_error(actuals, predictions))
    results.append(root_mean_squared_error(actuals, predictions))

    return results


metrics = pd.DataFrame(index=["MAE", "MSE", "RMSE"])

for model, preds in predictions.items():
    if preds.shape[1] == 2:
        pred_home = preds.iloc[:, 0]
        pred_away = preds.iloc[:, 1]
        pred_flat = preds.values.flatten()
        pred_spread = pred_home - pred_away

        metrics.loc[:, f"{model}_home"] = evaluate_predictions(pred_home, y_true_home)
        metrics.loc[:, f"{model}_away"] = evaluate_predictions(pred_away, y_true_away)
        metrics.loc[:, f"{model}_flat"] = evaluate_predictions(pred_flat, y_true_flat)
        metrics.loc[:, f"{model}_spread"] = evaluate_predictions(
            pred_spread, y_true_spread
        )
    else:
        metrics.loc[:, f"{model}_spread"] = evaluate_predictions(preds, y_true_spread)

metrics.T


# %%
def evaluate_probability_predictions(
    predictions: pd.DataFrame, actuals: Union[pd.DataFrame, pd.Series], model_name: str
) -> Dict[str, float]:
    """Evaluate probability predictions against actual match outcomes."""
    # Convert actuals to DataFrame if it's a Series
    if isinstance(actuals, pd.Series):
        actuals_df = pd.DataFrame({"outcome": actuals})
    else:
        actuals_df = actuals.copy()

    # Ensure actuals has the right column name
    if "outcome" not in actuals_df.columns:
        if len(actuals_df.columns) == 1:
            actuals_df = actuals_df.rename(columns={actuals_df.columns[0]: "outcome"})
        else:
            raise ValueError("Actuals DataFrame must contain 'outcome' column")

    # Merge predictions with actuals
    merged = pd.merge(
        predictions, actuals_df[["outcome"]], left_index=True, right_index=True
    )

    # Get predicted class (highest probability)
    merged["predicted_class"] = merged[["home", "draw", "away"]].idxmax(axis=1)
    merged["predicted_class"] = merged["predicted_class"].map(
        {"home": 1, "draw": 0, "away": -1}
    )

    # Calculate standard classification metrics
    results = {
        "accuracy": accuracy_score(merged["outcome"], merged["predicted_class"]),
        "precision": precision_score(
            merged["outcome"],
            merged["predicted_class"],
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            merged["outcome"],
            merged["predicted_class"],
            average="weighted",
            zero_division=0,
        ),
        "f1": f1_score(
            merged["outcome"],
            merged["predicted_class"],
            average="weighted",
            zero_division=0,
        ),
    }

    # Calculate log loss
    results["log_loss"] = log_loss(
        merged["outcome"], merged[["home", "draw", "away"]].values
    )

    # Calculate Brier score (multi-class version)
    y_onehot = pd.get_dummies(merged["outcome"]).values
    results["brier_score"] = np.mean(
        np.sum((merged[["home", "draw", "away"]].values - y_onehot) ** 2, axis=1)
    )

    # Create a DataFrame to return results
    metrics_df = pd.DataFrame.from_dict(results, orient="index", columns=[model_name])
    return metrics_df


metrics_probas = []
for model, proba in probas.items():
    metrics_probas.append(
        evaluate_probability_predictions(proba, y_true_outcome, model)
    )

metrics_probas = pd.concat(metrics_probas, axis=1)
metrics_probas.T

# %%


def evaluate_betting_performance(
    predictions: pd.DataFrame,
    actuals: Union[pd.DataFrame, pd.Series],
    bookmaker_odds: pd.DataFrame,
    stake: float = 1.0,
    stake_type: str = "fixed",
    min_ev: float = 0.02,
    fraction: float = 1.0,
) -> Dict[str, float]:
    """Evaluate betting performance based on model probabilities and bookmaker odds."""
    # Prepare data
    betting_df = pd.DataFrame(index=predictions.index)

    # Calculate Kelly
    betting_df["kelly_home"] = (
        (bookmaker_odds["home"] - 1) * predictions["home"] - (1 - predictions["home"])
    ) / (bookmaker_odds["home"] - 1)
    betting_df["kelly_draw"] = (
        (bookmaker_odds["draw"] - 1) * predictions["draw"] - (1 - predictions["draw"])
    ) / (bookmaker_odds["draw"] - 1)
    betting_df["kelly_away"] = (
        (bookmaker_odds["away"] - 1) * predictions["away"] - (1 - predictions["away"])
    ) / (bookmaker_odds["away"] - 1)

    # Calculate expected values (EV)
    betting_df["ev_home"] = (predictions["home"] * bookmaker_odds["home"]) - 1
    betting_df["ev_draw"] = (predictions["draw"] * bookmaker_odds["draw"]) - 1
    betting_df["ev_away"] = (predictions["away"] * bookmaker_odds["away"]) - 1

    # Determine which outcome to bet on (highest EV that meets minimum edge)
    betting_df["bet_home"] = betting_df["ev_home"] > min_ev
    betting_df["bet_draw"] = betting_df["ev_draw"] > min_ev
    betting_df["bet_away"] = betting_df["ev_away"] > min_ev

    if stake_type == "fixed":
        # Determine bet size
        betting_df["bet_size_home"] = stake * fraction
        betting_df["bet_size_draw"] = stake * fraction
        betting_df["bet_size_away"] = stake * fraction
    elif stake_type == "kelly":
        # Determine bet size
        betting_df["bet_size_home"] = betting_df["kelly_home"] * stake * fraction
        betting_df["bet_size_draw"] = betting_df["kelly_draw"] * stake * fraction
        betting_df["bet_size_away"] = betting_df["kelly_away"] * stake * fraction

    # Calculate results for each potential bet
    betting_df["won_home"] = (actuals == 1) & betting_df["bet_home"]
    betting_df["won_draw"] = (actuals == 0) & betting_df["bet_draw"]
    betting_df["won_away"] = (actuals == -1) & betting_df["bet_away"]

    # Calculate profits
    betting_df["profit_home"] = np.where(
        betting_df["bet_home"],
        np.where(
            betting_df["won_home"],
            betting_df["bet_size_home"] * bookmaker_odds["home"]
            - betting_df["bet_size_home"],
            -betting_df["bet_size_home"],
        ),
        0,
    )

    betting_df["profit_draw"] = np.where(
        betting_df["bet_draw"],
        np.where(
            betting_df["won_draw"],
            betting_df["bet_size_draw"] * bookmaker_odds["draw"]
            - betting_df["bet_size_draw"],
            -betting_df["bet_size_draw"],
        ),
        0,
    )

    betting_df["profit_away"] = np.where(
        betting_df["bet_away"],
        np.where(
            betting_df["won_away"],
            betting_df["bet_size_away"] * bookmaker_odds["away"]
            - betting_df["bet_size_away"],
            -betting_df["bet_size_away"],
        ),
        0,
    )

    # Return results
    return betting_df


# model = "ens"
# proba = pd.concat(probas.values(), axis=1)["home"].mean(axis=1).to_frame(name="home")
# proba["draw"] = pd.concat(probas.values(), axis=1)["draw"].mean(axis=1)
# proba["away"] = pd.concat(probas.values(), axis=1)["away"].mean(axis=1)

backtests = []
for model, proba in probas.items():
    backtest = evaluate_betting_performance(
        proba,
        y_true_outcome,
        test_odds,
        stake=500,
        min_ev=0.4,
        fraction=1,
        stake_type="kelly",
    )
    backtest["model"] = model
    backtest.index = test.home_team + " - " + test.away_team
    backtests.append(backtest)


backtests = pd.concat(backtests, axis=0)

backtests.query("model == 'skellam_zero'")
backtests.groupby("model").sum().filter(regex="profit").idxmax()
print(f"{func} {sampling_method}")
print(backtests.groupby("model").sum().filter(regex="profit"))

# %%
