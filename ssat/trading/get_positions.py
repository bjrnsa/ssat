# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
from flashscore_scraper.data_loaders import Handball

from ssat.bayesian.predictive_models import SkellamZero


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
    # "league": "Herre Handbold Ligaen",
    # "seasons": [2025],
    "country": "Europe",
}
df = loader.load_matches(**loader_params)
df["weights"] = dixon_coles_weights(df.datetime, xi=0.0018)
feautres = ["home_team", "away_team", "home_goals", "away_goals", "weights"]

model = SkellamZero()
model.fit(df[feautres], seed=2323)
# model.plot_trace()
# model.plot_team_stats()

fixtures = loader.load_fixtures(**loader_params)
# Get todays matches
fixtures = fixtures.set_index("flashscore_id")
fixtures = fixtures.sort_values("datetime")


# %%
predicted_proba = model.predict_proba(fixtures[["home_team", "away_team"]])
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
for model_name in predicted_proba.columns:
    evs[model_name] = predicted_proba[model_name] * min_odds[model_name] - 1

evs = pd.DataFrame(evs)

min_odds - predicted_odds.loc[min_odds.index].round(2)
min_odds
# %%
