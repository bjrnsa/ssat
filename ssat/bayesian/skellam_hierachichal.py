"""Bayesian Negative Binomial Hierarchical Model."""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ssat.bayesian.base_model import BaseModel, FitError


class SkellamHierarchical(BaseModel):
    """Bayesian Negative Binomial Hierarchical Model."""

    def __init__(
        self,
        stem: str = "skellam_hierachichal",
    ):
        """Initialize the Negative Binomial Hierarchical model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "skellam_hierachichal".
        """
        super().__init__(stan_file=stem)


if __name__ == "__main__":
    df = pd.read_pickle("ssat/data/handball_data.pkl")
    df = df[["home_team", "away_team", "home_goals", "away_goals"]]
    model = SkellamHierarchical()
    model.fit(df)
    matches = pd.DataFrame(
        {
            "home_team": ["Sonderjyske", "Aalborg", "Ringsted"],
            "away_team": ["GOG", "Holstebro", "Fredericia"],
        }
    )
    pred = model.predict(matches)
    proba = model.predict_proba(matches)
    model.plot_trace()
    model.plot_team_stats()
