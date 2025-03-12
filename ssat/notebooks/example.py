# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
from flashscore_scraper.data_loaders import Handball
from numpy.typing import NDArray

from ssat.bayesian.predictive_models import SkellamZero, SkellamZeroWeighted


def dixon_coles_weights(dates, xi=0.0018, base_date=None) -> NDArray:
    """Calculates a decay curve based on the algorithm given by Dixon and Coles in their paper.

    Parameters
    ----------
    dates : list
        A list or pd.Series of dates to calculate weights for
    x1 : float
        Controls the steepness of the decay curve
    base_date : date
        The base date to start the decay from. If set to None
        then it uses the maximum date
    """
    if base_date is None:
        base_date = max(dates)

    diffs = np.array([(base_date - x).days for x in dates])
    weights = np.exp(-xi * diffs)
    return weights


db_path = Path(os.environ.get("DB_PATH", "database/database.db"))
loader = Handball(db_path=db_path)
loader_params = {
    "country": "Herre Handbold Ligaen",
}
df = loader.load_matches(**loader_params).sort_values("datetime")
df["weights"] = dixon_coles_weights(df.index, xi=0.018)
df["days_since_match"] = (df.index.max() - df.index).days
df = df[
    [
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "days_since_match",
        "weights",
    ]
]
model_1 = SkellamZero()
model_2 = SkellamZeroWeighted()
preds = []
probas = []
matches = pd.read_pickle("ssat/data/handball_fixtures.pkl").sort_values(by="datetime")
matches = matches[["home_team", "away_team"]]
matches.index = matches.home_team + " - " + matches.away_team

model_1.fit(
    df[["home_team", "away_team", "home_goals", "away_goals", "weights"]],
    seed=111,
)

model_2.fit(
    df[["home_team", "away_team", "home_goals", "away_goals", "days_since_match"]],
    seed=111,
)


# Visualize results
model_1.plot_trace(var_names=["gamma", "attack_team", "defence_team", "home_advantage"])
model_2.plot_trace(
    var_names=["xi", "gamma", "attack_team", "defence_team", "home_advantage"]
)
model_2.plot_team_stats()

model_2.plot_team_stats()
