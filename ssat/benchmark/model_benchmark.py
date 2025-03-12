# %%
import os
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
from flashscore_scraper.data_loaders import Handball
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

from ssat.bayesian.predictive_models import Skellam, SkellamZero, SkellamZeroWeighted
from ssat.odds.implied import ImpliedOdds


def dixon_coles_weights(dates, xi=0.0018, base_date=None):
    """Calculates a decay curve based on the algorithm given by Dixon and Coles in their paper."""
    if base_date is None:
        base_date = max(dates)

    diffs = np.array([(base_date - x).days for x in dates])
    weights = np.exp(-xi * diffs)
    return weights


db_path = Path(os.environ.get("DB_PATH", "database/database.db"))
loader = Handball(db_path=db_path)
loader_params = {
    "league": "Herre Handbold Ligaen",
    # "date_range": (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01")),
}
df = loader.load_matches(**loader_params).sort_values("datetime")
df.set_index("flashscore_id", inplace=True)

feautres = ["home_team", "away_team", "home_goals", "away_goals"]
feautres_with_weights = feautres + ["weights", "days_since_match"]
train, test = df.iloc[:-20], df.iloc[-20:]

train["days_since_match"] = (train.datetime.max() - train.datetime).dt.days
train["weights"] = dixon_coles_weights(train.datetime, xi=0.0018)


odds_df = loader.load_odds(df.index.tolist())
odds_df = odds_df.groupby("flashscore_id")[
    ["home_odds", "draw_odds", "away_odds"]
].median()
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


models = [
    # SkellamZeroWeighted(),
    SkellamZero(),
    Skellam(),
]

predictions = {}
probas = {}
# for model in models:
# model.fit(train[feautres], seed=2)
# model.plot_trace()
# model.plot_team_stats()
# predictions[model.__class__.__name__] = model.predict(test[feautres])
# probas[model.__class__.__name__] = model.predict_proba(test[feautres])

train_weights = train[feautres + ["weights"]]
train_days = train[feautres + ["days_since_match"]]

for model in models:
    model.fit(train_weights, seed=234)
    predictions[model.__class__.__name__] = model.predict(test[feautres])
    probas[model.__class__.__name__] = model.predict_proba(test[feautres])

model_new = SkellamZeroWeighted()
model_new.fit(train_days, seed=12)
predictions[model_new.__class__.__name__] = model_new.predict(test[feautres])
probas[model_new.__class__.__name__] = model_new.predict_proba(test[feautres])

model_new.plot_trace(
    var_names=["xi", "gamma", "attack_team", "defence_team", "home_advantage"]
)
model_new.plot_team_stats()

y_true_home = test["home_goals"]
y_true_away = test["away_goals"]
y_true_flat = test[["home_goals", "away_goals"]].values.flatten()
y_true_spread = test["home_goals"] - test["away_goals"]
y_true_outcome = np.sign(y_true_spread)
y_true_outcome


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
            betting_df["bet_size_home"] * bookmaker_odds["home"],
            -betting_df["bet_size_home"],
        ),
        0,
    )

    betting_df["profit_draw"] = np.where(
        betting_df["bet_draw"],
        np.where(
            betting_df["won_draw"],
            betting_df["bet_size_draw"] * bookmaker_odds["draw"],
            -betting_df["bet_size_draw"],
        ),
        0,
    )

    betting_df["profit_away"] = np.where(
        betting_df["bet_away"],
        np.where(
            betting_df["won_away"],
            betting_df["bet_size_away"] * bookmaker_odds["away"],
            -betting_df["bet_size_away"],
        ),
        0,
    )

    # Return results
    return betting_df


backtests = []
for model, proba in probas.items():
    backtest = evaluate_betting_performance(
        proba,
        y_true_outcome,
        test_odds,
        stake=25,
        min_ev=0.08,
        fraction=1,
        stake_type="fixed",
    )
    backtest["model"] = model
    backtest.index = test.home_team + " - " + test.away_team
    backtests.append(backtest)


backtests = pd.concat(backtests, axis=0)


backtests.groupby("model").sum().filter(regex="profit").idxmax()
backtests.groupby("model").sum().filter(regex="profit")

# %%
