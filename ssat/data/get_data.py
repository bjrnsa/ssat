# %%
"""This script is used to load the data from the database."""

import os
from pathlib import Path

from flashscore_scraper.data_loaders import Handball, Volleyball

db_path = Path(os.environ.get("DB_PATH", "database/database.db"))
loader = Handball(db_path=db_path)
loader_params = {
    "league": "European Championship",
    # "seasons": ["2024/2025"],
    "date_range": None,
    "team_filters": None,
    # "include_additional_data": True,
}
df = loader.load_matches(**loader_params)
df.to_pickle("ssat/data/handball_data.pkl")
odds_df = loader.load_odds(df.index.tolist())
odds_df.to_pickle("ssat/data/handball_odds.pkl")

# %%
