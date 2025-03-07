# %%
"""Example of a frequentist analysis using the bayesian module."""

from datetime import date
from os import name
from re import A
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from scipy.stats import skellam

from ssat.bayesian import (
    NegBinomHierarchical,
    PoissonHierarchical,
    SkellamHierarchical,
    SkellamHierarchicalDLC,
)
from ssat.odds import ImpliedOdds

skellam.logpmf(0, 1, 1)


def dixon_coles_weights(
    dates: Union[List[date], pd.Series], xi: float = 0.0018, base_date: date = None
) -> NDArray:
    """Calculates a decay curve based on the algorithm given by Dixon and Coles in their paper.

    Parameters
    ----------
    dates : Union[List[date], pd.Series]
        A list or pd.Series of dates to calculate weights for.
    xi : float, optional
        Controls the steepness of the decay curve. Defaults to 0.0018.
    base_date : date, optional
        The base date to start the decay from. If set to None, it uses the maximum date from the dates list. Defaults to None.

    Returns:
    -------
    NDArray
        An array of weights corresponding to the input dates.
    """
    if base_date is None:
        base_date = max(dates)

    diffs = np.array([(base_date - x).days for x in dates])
    weights = np.exp(-xi * diffs)
    return weights


# Load the data
df = pd.read_pickle("ssat/data/handball_data.pkl")
odds_df = pd.read_pickle("ssat/data/handball_odds.pkl")
odds_df = odds_df.groupby("flashscore_id").max().select_dtypes("number")
# df = pd.read_csv("ssat/notebooks/nhl2023_goals.csv")
# df["home_goals"] = df.filter(regex="home_").select_dtypes("number").sum(axis=1)
# df["away_goals"] = df.filter(regex="away_").select_dtypes("number").sum(axis=1)
# df["spread"] = df["home_goals"] - df["away_goals"]
implied = ImpliedOdds(methods=["power"])
implied_probs = implied.get_implied_probabilities(odds_df)
implied_probs = implied_probs.pivot_table(
    index="match_id", columns="outcome", values="power"
)
test_idx = 10

df = df.merge(implied_probs, left_index=True, right_index=True)

X = df[["home_team", "away_team"]].head(-test_idx)
Z = df[["home_goals", "away_goals", "home", "away"]].head(-test_idx)
y = df["spread"].head(-test_idx)

# X_test is the last 10 games
X_train = X
Z_train = Z
y_train = y


weights = dixon_coles_weights(df["datetime"].tail(-test_idx), xi=0.002)
skellam_dlc_model = SkellamHierarchicalDLC()
skellam_dlc_model.fit(
    X_train,
    Z_train,
    y_train,
    # weights=weights,
    draws=10000,
    warmup=2000,
    chains=3,
    plot_diagnostics=True,
)
model = SkellamHierarchical()
model.fit(
    X_train,
    Z_train[["home_goals", "away_goals"]],
    y_train,
    # weights=weights,
    draws=10000,
    warmup=2000,
    chains=3,
    plot_diagnostics=True,
)
team_ratings = model.get_team_ratings()
model.teams_

todays_fixtures_dlc = pd.DataFrame(
    {
        "home_team": ["Kolding"],
        "away_team": ["Skanderborg AGF"],
        "home": [0.17585894953444664],
        "away": [0.7409504021807284],
    }
)

todays_fixtures = pd.DataFrame(
    {
        "home_team": ["Kolding"],
        "away_team": ["Skanderborg AGF"],
    }
)
simulated_matches = model.predict(todays_fixtures, return_matches=True)
simulated_matches_dlc = skellam_dlc_model.predict(
    todays_fixtures_dlc, return_matches=True
)

odds_df.loc["nyyTYomn"]
implied_probs.loc["nyyTYomn"]

probas = model.predict_proba(todays_fixtures, threshold=0, include_draw=True)
probas_dls = skellam_dlc_model.predict_proba(
    todays_fixtures_dlc, threshold=0, include_draw=True
)
1 / probas_dls
1 / probas

# %%


def kelly_criterion(decimal_odds, win_probability):
    """Calculate the optimal bet size using Kelly Criterion with European decimal odds.

    Parameters:
    decimal_odds (float): European decimal odds (e.g., 2.5 means you win 2.5 times your stake)
    win_probability (float): Probability of winning (between 0 and 1)

    Returns:
    float: Fraction of bankroll to bet (negative means don't bet)
    """
    kelly_fraction = (win_probability * decimal_odds - 1) / (decimal_odds - 1)

    return max(0, kelly_fraction)


def expected_value(model_prob, actual_odds):
    return model_prob * actual_odds - 1


backtest_df = df.tail(test_idx)
backtest_odds = odds_df.loc[backtest_df.index]
backtest_probas = pd.DataFrame(
    skellam_dlc_model.predict_proba(
        backtest_df[["home_team", "away_team", "home", "away"]], threshold=0
    ),
    columns=backtest_odds.columns,
    index=backtest_odds.index,
)
backtest_probas = pd.DataFrame(
    model.predict_proba(backtest_df[["home_team", "away_team"]], threshold=0),
    columns=backtest_odds.columns,
    index=backtest_odds.index,
)
backtest_model_odds = pd.DataFrame(
    1 / backtest_probas, columns=backtest_odds.columns, index=backtest_odds.index
)

# Find good bets
good_bets = []
for idx, row in backtest_model_odds.iterrows():
    actual_odds = backtest_odds.loc[idx]
    for outcome in ["home_odds", "draw_odds", "away_odds"]:
        if row[outcome] < actual_odds[outcome]:
            model_prob = 1 / row[outcome]
            kelly = kelly_criterion(actual_odds[outcome], model_prob)
            ev = expected_value(model_prob, actual_odds[outcome])
            if kelly > 0:
                good_bets.append(
                    {
                        "match_index": idx,
                        "outcome": outcome,
                        "model_odds": row[outcome],
                        "actual_odds": actual_odds[outcome],
                        "kelly_criterion": kelly,
                        "expected_value": ev,
                    }
                )

# %%
good_bets_df = pd.DataFrame(good_bets)
bankroll_start = 1000
bankroll = bankroll_start
fraction = 0.25
pnl = []
roi = 0
rois = []
bet_sizes = []
bankroll_prev = bankroll

for idx, row in good_bets_df.iterrows():
    if row["outcome"] == "home_odds":
        bet_size = row["kelly_criterion"] * bankroll * fraction
        if backtest_df.loc[row["match_index"], "result"] == 1:
            bankroll += bet_size * row["actual_odds"]
        else:
            bankroll -= bet_size

    elif row["outcome"] == "away_odds":
        bet_size = row["kelly_criterion"] * bankroll * fraction
        if backtest_df.loc[row["match_index"], "result"] == -1:
            bankroll += bet_size * row["actual_odds"]
        else:
            bankroll -= bet_size
    else:
        bet_size = row["kelly_criterion"] * bankroll * fraction
        if backtest_df.loc[row["match_index"], "result"] == 0:
            bankroll += bet_size * row["actual_odds"]
        else:
            bankroll -= bet_size
    roi = bankroll / bankroll_prev - 1
    pnl.append(bankroll)
    rois.append(roi)
    bet_sizes.append(bet_size)
    bankroll_prev = bankroll

good_bets_df["bet_size"] = bet_sizes
good_bets_df["pnl"] = pnl
good_bets_df["roi"] = rois

# (good_bets_df["roi"] + 1).cumprod().plot()
good_bets_df["pnl"].plot()
good_bets_df

# %%
trace = model.inference_data
agg_strength = (
    trace.posterior["attack_team"].values - trace.posterior["defence_team"].values
)
agg_strength_df = pd.DataFrame(
    agg_strength.reshape(-1, agg_strength.shape[-1]), columns=model.teams_
).melt(var_name="Team", value_name="Strength")

ordered_team_names = sorted(model.teams_)
agg_strength_df["Team"] = agg_strength_df["Team"].map(
    lambda x: ordered_team_names[ordered_team_names.index(x)]
)
agg_strength_df.groupby("Team")["Strength"].mean().sort_values()
plt.figure(figsize=(20, 15))
for i, team in enumerate(ordered_team_names):
    plt.subplot((len(ordered_team_names) + 3) // 4, 4, i + 1)
    sns.kdeplot(
        data=agg_strength_df[agg_strength_df["Team"] == team], x="Strength", fill=True
    )
    plt.axvline(x=0, color="red", linestyle="dashed")
    plt.title(team)
    plt.xlabel("")
    plt.ylabel("")

plt.tight_layout()
plt.suptitle("Aggregate Strength Distributions by Team", fontsize=16, y=1.02)
plt.show()
