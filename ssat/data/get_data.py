# %%
"""This script is used to load the data from the database."""

import os
from pathlib import Path

from flashscore_scraper.data_loaders import Handball

db_path = Path(os.environ.get("DB_PATH", "database/database.db"))
loader = Handball(db_path=db_path)
loader_params = {
    "league": "Herre Handbold Ligaen",
    "seasons": ["2023/2024"],
    "date_range": None,
    "team_filters": None,
    "include_additional_data": True,
}
df = loader.load_matches(**loader_params)
df.to_pickle("ssat/data/sample_handball_data.pkl")
