# %%
import os
from pathlib import Path
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flashscore_scraper.data_loaders import Handball

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

teams_to_remove = ["Lemvig", "Skive", "Midtjylland"]
df = df.query("home_team not in @teams_to_remove and away_team not in @teams_to_remove")
last_train_date = "2025-02-23"

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
df = df.join(implied_odds, on="flashscore_id")
feautres_skellam = ["home_team", "away_team", "goal_diff_match"]
feautres_poisson = ["home_team", "away_team", "home_goals", "away_goals"]
train, test = (
    df.query("datetime < @last_train_date"),
    df.query("datetime >= @last_train_date"),
)

goals_ = train[["home_goals", "away_goals"]]
goal_diff_ = train["goal_diff_match"][train["goal_diff_match"].abs() < 30]
# Remove Lemvig, Skive, Midtjylland
teams_to_remove = ["Lemvig", "Skive", "Midtjylland"]
df = df.query("home_team not in @teams_to_remove and away_team not in @teams_to_remove")


feautres = ["home_team", "away_team", "goal_diff_match"]
covars = ["weights", "home", "away"]
train, test = (
    df.query("datetime < @last_train_date"),
    df.query("datetime >= @last_train_date"),
)


# Remove teams in test that are not present in train
test = test[test.home_team.isin(train.home_team) & test.away_team.isin(train.away_team)]
test_odds = odds_df.query("flashscore_id in @test.index")
test_implied_odds = implied_odds.query("match_id in @test.index")


predictions = {}
probas = {}


skellam = Skellam()
skellam.fit(
    base_data=train[feautres_skellam],
    optional_data=train["weights"],
    seed=2,
)


skellam_weighted = SkellamWeighted()
skellam_weighted.fit(train[feautres_skellam], train["days_since_match"], seed=3)


skellam_zero = SkellamZero()
skellam_zero.fit(
    train[feautres_skellam],
    train["weights"],
    seed=2,
)


skellam_zero_weighted = SkellamZeroWeighted()
skellam_zero_weighted.fit(
    train[feautres_skellam],
    train["days_since_match"],
    seed=326,
)

# %%

pred_goal_skellam_dweibull = skellam_dweibull.predict(
    test[feautres_skellam],
    sampling_method="qskellam",
    func="median",
)
pred_goal_skellam = skellam.predict(
    test[feautres_skellam],
    sampling_method="qskellam",
    func="median",
)
pred_goal_skellam_weighted = skellam_weighted.predict(
    test[feautres_skellam],
    sampling_method="qskellam",
    func="median",
)
pred_goal_skellam_zero = skellam_zero.predict(
    test[feautres_skellam],
    sampling_method="qskellam",
    func="median",
)
pred_goal_skellam_zero_weighted = skellam_zero_weighted.predict(
    test[feautres_skellam],
    sampling_method="qskellam",
    func="median",
)


probas = {
    "skellam_covar": skellam_covar.predict_proba(
        test[feautres_skellam], sampling_method="qskellam"
    ),
    "skellam": skellam.predict_proba(
        test[feautres_skellam], sampling_method="qskellam"
    ),
    # "skellam_weighted": skellam_weighted.predict_proba(
    #     test[feautres_skellam], sampling_method="qskellam"
    # ),
    # "skellam_zero": skellam_zero.predict_proba(
    #     test[feautres_skellam], sampling_method="qskellam"
    # ),
    # "skellam_zero_weighted": skellam_zero_weighted.predict_proba(
    #     test[feautres_skellam], sampling_method="qskellam"
    # ),
}

# %%
# Create posterior predictive plots
actual_goal_diff = test["goal_diff_match"]
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
axes = axes.flatten()

# Plot histograms for each model
for ax, pred, title in zip(
    axes,
    [
        pred_goal_skellam_covar,
        pred_goal_skellam,
        pred_goal_skellam_weighted,
        pred_goal_skellam_zero,
        pred_goal_skellam_zero_weighted,
    ],
    [
        "Skellam Covar",
        "Skellam",
        "Skellam Weighted",
        "Skellam Zero",
        "Skellam Zero Weighted",
    ],
):
    # Flatten predictions across all samples
    pred_flat = np.array(pred).flatten()

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


y_true_outcome = np.sign(test["goal_diff_match"])
backtests = []
for model, proba in probas.items():
    backtest = evaluate_betting_performance(
        proba,
        y_true_outcome,
        test_odds,
        stake=500,
        min_ev=0.05,
        fraction=0.1,
        stake_type="kelly",
    )
    backtest["model"] = model
    backtest.index = test.home_team + " - " + test.away_team
    backtests.append(backtest)


backtests = pd.concat(backtests, axis=0)

backtests.query("model == 'skellam'")
backtests.groupby("model").sum().filter(regex="profit").idxmax()
print(backtests.groupby("model").sum().filter(regex="profit"))

# %%
