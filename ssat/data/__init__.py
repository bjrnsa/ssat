from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent


def get_sample_handball_match_data() -> pd.DataFrame:
    """Returns sample handball match data as a pandas DataFrame."""
    return pd.read_parquet(ROOT / "sample_handball_match_data.parquet")


def get_sample_handball_odds_data() -> pd.DataFrame:
    """Returns sample handball odds data as a pandas DataFrame."""
    return pd.read_parquet(ROOT / "sample_handball_odds_data.parquet")
