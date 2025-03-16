# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
from flashscore_scraper.data_loaders import Handball

from ssat.bayesian.predictive_models import SkellamZeroWeighted


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
    "country": "Europe",
    # "seasons": [2025],
    "include_additional_data": False,
}
df = loader.load_matches(**loader_params).sort_values("datetime")
df.set_index("flashscore_id", inplace=True)


df = df.assign(
    weights=dixon_coles_func(
        df.datetime,
        xi=0.0018,
        output_type="weights",
    ),
    days_since_match=dixon_coles_func(
        df.datetime,
        xi=0.0018,
        output_type="days_since_match",
    ),
    goal_diff_match=df.home_goals - df.away_goals,
)

feautres = ["home_team", "away_team", "goal_diff_match"]

skellam_zero_weighted = SkellamZeroWeighted()
skellam_zero_weighted.fit(df[feautres], df["days_since_match"], seed=2)
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

fixtures_params = {
    # "league": "Herre Handbold Ligaen",
    "country": "Europe",
    # "seasons": [2025],
}
fixtures = loader.load_fixtures(**fixtures_params)
# Get todays matches
fixtures = fixtures.set_index("flashscore_id")
fixtures = fixtures.sort_values("datetime")
fixtures = fixtures.query("datetime.dt.date == @pd.Timestamp.today().date()")

# %%
predicts = skellam_zero_weighted.predict(
    fixtures[["home_team", "away_team"]],
    sampling_method="qskellam",
    func="median",
)
predicted_proba = skellam_zero_weighted.predict_proba(
    fixtures[["home_team", "away_team"]]
)
predicted_proba.index.name = "flashscore_id"
predicted_odds = 1 / predicted_proba
odds_df = loader.load_odds(fixtures.index.tolist())
min_odds = odds_df.query("bookmaker_name == 'unibet'")
min_odds = min_odds.query("flashscore_id in @fixtures.index")
predicted_odds = predicted_odds.query("flashscore_id in @min_odds.index")
predicted_proba = predicted_proba.query("flashscore_id in @min_odds.index")

predicted_proba["matchup"] = fixtures.home_team + " - " + fixtures.away_team
predicted_odds["matchup"] = fixtures.home_team + " - " + fixtures.away_team
min_odds["matchup"] = fixtures.home_team + " - " + fixtures.away_team
min_odds = min_odds.rename(
    columns={
        "home_odds": "home",
        "draw_odds": "draw",
        "away_odds": "away",
    }
)
min_odds = min_odds.set_index("matchup")
predicted_odds = predicted_odds.set_index("matchup")
predicted_proba = predicted_proba.set_index("matchup")
min_odds = min_odds.drop(columns=["bookmaker_name"])
evs = {}
kellys = {}
for model_name in predicted_proba.columns:
    evs[model_name] = predicted_proba[model_name] * min_odds[model_name] - 1
    kellys[model_name] = (
        predicted_proba[model_name] * min_odds[model_name] - 1
    ) / min_odds[model_name]

evs = pd.DataFrame(evs)
kellys = pd.DataFrame(kellys)
good_matches = evs > 0.4

# Show only True from good_matches
bet_sizes = (kellys * good_matches * 500).clip(0, 500).astype(int)


# %%

good_matches.loc[evs > 0.4]


# %%
