"""Bayesian Poisson Hierarchical Model."""

from typing import Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ssat.bayesian.base_model import BaseModel, FitError


class PoissonHierarchical(BaseModel):
    """Bayesian Negative Binomial Hierarchical Model."""

    def __init__(
        self,
        stem: str = "poisson_hierarchical",
    ):
        """Initialize the Negative Binomial Hierarchical model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "poisson_hierarchical".
        """
        super().__init__(stan_file=stem)


if __name__ == "__main__":
    df = pd.read_pickle("ssat/data/handball_data.pkl")
    df = df[["home_team", "away_team", "home_goals", "away_goals"]]
    model = PoissonHierarchical()
    model.fit(df)
    pred = model.predict(df)
    proba = model.predict_proba(df)
    model.plot_trace()
    model.plot_team_stats()
