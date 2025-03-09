"""Bayesian Skellam Model for sports prediction."""

from typing import Optional

import pandas as pd

from ssat.bayesian.base_model import BaseModel


class Skellam(BaseModel):
    """Bayesian Skellam Model for predicting match scores.
    
    This model uses a Skellam distribution (difference of two Poisson distributions)
    to directly model the goal difference between teams, accounting for both team
    attack and defense capabilities.
    """

    def __init__(
        self,
        stem: str = "skellam",
    ):
        """Initialize the Skellam model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "skellam".
        """
        super().__init__(stan_file=stem)


if __name__ == "__main__":
    # Example usage
    df = pd.read_pickle("ssat/data/handball_data.pkl")
    df = df[["home_team", "away_team", "home_goals", "away_goals"]]
    model = Skellam()
    model.fit(df)
    
    # Make predictions
    matches = pd.DataFrame({
        "home_team": ["Sonderjyske", "Aalborg", "Ringsted"],
        "away_team": ["GOG", "Holstebro", "Fredericia"]
    })
    pred = model.predict(matches)
    proba = model.predict_proba(matches)
    
    # Visualize results
    model.plot_trace()
    model.plot_team_stats()
