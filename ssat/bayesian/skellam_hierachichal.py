"""Bayesian Negative Binomial Hierarchical Model."""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ssat.bayesian.base_model import BaseModel, FitError
from ssat.bayesian.hierarchical_base import HierarchicalBaseModel


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
    df = pd.read_csv("ssat/data/afl_data.csv")
    df = df[["Home Team", "Away Team", "Home Pts", "Away Pts"]]
    model = SkellamHierarchical()
    model.fit(df)
    pred = model.predict(df)
    proba = model.predict_proba(df)
    pass
