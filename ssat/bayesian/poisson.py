"""Bayesian Poisson Model for sports prediction."""

from typing import Optional

import pandas as pd

from ssat.bayesian.base_model import BaseModel


class Poisson(BaseModel):
    """Bayesian Poisson Model for predicting match scores.
    
    This model uses a Poisson distribution to model goal scoring,
    accounting for both team attack and defense capabilities.
    """

    def __init__(
        self,
        stem: str = "poisson",
    ):
        """Initialize the Poisson model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "poisson".
        """
        super().__init__(stan_file=stem)


if __name__ == "__main__":
    # Example usage
    df = pd.read_pickle("ssat/data/handball_data.pkl")
    df = df[["home_team", "away_team", "home_goals", "away_goals"]]
    model = Poisson()
    model.fit(df)
    
    # Make predictions
    matches = pd.DataFrame({
        "home_team": ["Sonderjyske", "Aalborg"],
        "away_team": ["GOG", "Holstebro"]
    })
    pred = model.predict(matches)
    proba = model.predict_proba(matches)
    
    # Visualize results
    model.plot_trace()
    model.plot_team_stats()
