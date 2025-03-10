"""Bayesian Negative Binomial Model for sports prediction."""

from typing import Optional

import pandas as pd

from ssat.bayesian.base_model import BaseModel


class EloRating(BaseModel):
    """Bayesian Elo Rating Model for predicting match scores.

    This model uses Elo ratings to model goal scoring,
    accounting for both team attack and defense capabilities.
    """

    def __init__(
        self,
        stem: str = "elo_rating",
    ):
        """Initialize the Elo Rating model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "elo_rating".
        """
        super().__init__(stan_file=stem)


if __name__ == "__main__":
    # Example usage
    df = pd.read_pickle("ssat/data/handball_data.pkl")
    df = df[["home_team", "away_team", "result"]]

    model = EloRating()
    model.fit(df)
    # Visualize results
    model.plot_trace(var_names=["rating", "home_adv", "K", "draw_width"])
    model.plot_team_stats()

    # Make predictions
    matches = pd.DataFrame(
        {"home_team": ["Sonderjyske", "Aalborg"], "away_team": ["GOG", "Holstebro"]}
    )
    pred = model.predict(matches)
    proba = model.predict_proba(matches)
