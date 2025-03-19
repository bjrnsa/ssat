# %%
import os
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
from flashscore_scraper.data_loaders import Handball

from ssat.bayesian.predictive_models import Skellam, SkellamZero
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
last_train_date = "2024-12-23"

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

feautres_skellam = ["home_team", "away_team", "goal_diff_match"]

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


probas_skellam = []
probas_skellam_zero = []

# Expading train set that starts from last_train_date
for week, week_df in df.groupby(pd.Grouper(key="datetime", freq="W")):
    if week_df.shape[0] == 0:
        continue
    if week < pd.Timestamp(last_train_date):
        continue

    print(f"Predicting week {week}")
    print(f"Fixtures: {week_df.home_team.unique()} x {week_df.away_team.unique()}")

    test = week_df
    train = df.query("datetime < @week and index not in @week_df.index")

    skellam = Skellam()
    skellam.fit(
        train[feautres_skellam],
        train["weights"],
        seed=2,
    )
    probas_skellam.append(
        skellam.predict_proba(test[feautres_skellam], sampling_method="qskellam")
    )

    skellam_zero = SkellamZero()
    skellam_zero.fit(
        train[feautres_skellam],
        train["weights"],
        seed=2,
    )
    probas_skellam_zero.append(
        skellam_zero.predict_proba(test[feautres_skellam], sampling_method="qskellam")
    )


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


probas_skellam_zero_df = pd.concat(probas_skellam_zero, axis=0)
probas_skellam_df = pd.concat(probas_skellam, axis=0)
test_odds = odds_df.loc[probas_skellam_df.index]
y_true_outcome = np.sign(df.loc[probas_skellam_df.index]["goal_diff_match"])
backtests = []

backtest = evaluate_betting_performance(
    probas_skellam_df,
    y_true_outcome,
    test_odds,
    stake=500,
    min_ev=0.5,
    fraction=0.1,
    stake_type="kelly",
)
backtest["model"] = "skellam_weighted"
backtest.index = (
    df.loc[probas_skellam_df.index].home_team
    + " - "
    + df.loc[probas_skellam_df.index].away_team
)
backtests.append(backtest)

backtest = evaluate_betting_performance(
    probas_skellam_zero_df,
    y_true_outcome,
    test_odds,
    stake=500,
    min_ev=0.5,
    fraction=0.1,
    stake_type="kelly",
)
backtest["model"] = "skellam_zero_weighted"
backtest.index = (
    df.loc[probas_skellam_zero_df.index].home_team
    + " - "
    + df.loc[probas_skellam_zero_df.index].away_team
)
backtests.append(backtest)

backtests = pd.concat(backtests, axis=0)

backtests.query("model == 'skellam_zero_weighted'")
backtests.groupby("model").sum().filter(regex="profit").idxmax()
print(backtests.groupby("model").sum().filter(regex="profit"))

# %%
