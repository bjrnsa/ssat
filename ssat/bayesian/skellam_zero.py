"""Bayesian Zero-inflated Skellam Model for sports prediction."""

from datetime import date
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from ssat.bayesian.base_model import BaseModel


class SkellamZero(BaseModel):
    """Bayesian Zero-inflated Skellam Model for predicting match scores.

    This model uses a zero-inflated Skellam distribution to model goal differences,
    particularly suitable for low-scoring matches or competitions with frequent draws.
    The zero-inflation component explicitly models the probability of a draw.
    """

    def __init__(
        self,
        stem: str = "skellam_zero",
    ):
        """Initialize the Zero-inflated Skellam model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "skellam_zero".
        """
        super().__init__(stan_file=stem)


def dixon_coles_weights(
    dates: Union[List[date], pd.Series], xi: float = 0.0018, base_date: date = None
):
    """Calculates a decay curve based on the algorithm given by Dixon and Coles in their paper.

    Parameters
    ----------
    dates : Union[List[date], pd.Series]
        A list or pd.Series of dates to calculate weights for.
    xi : float, optional
        Controls the steepness of the decay curve. Defaults to 0.0018.
    base_date : date, optional
        The base date to start the decay from. If set to None, it uses the maximum date from the dates list. Defaults to None.

    Returns:
    -------
    NDArray
        An array of weights corresponding to the input dates.
    """
    if base_date is None:
        base_date = max(dates)

    diffs = np.array([(base_date - x).days for x in dates])
    weights = np.exp(-xi * diffs)
    return weights


if __name__ == "__main__":
    # Example usage
    df = pd.read_pickle("ssat/data/handball_data.pkl")
    df["weights"] = dixon_coles_weights(df["datetime"])
    df = df[["home_team", "away_team", "home_goals", "away_goals", "weights"]]
    model = SkellamZero()
    model.fit(df)

    # Visualize results
    model.plot_trace()
    model.plot_team_stats()

    # Make predictions
    matches = pd.DataFrame(
        {
            "home_team": [
                "Czech Republic",
                "Estonia",
                "Greece",
                "Montenegro",
                "Turkey",
                "Luxembourg",
                "Faroe Islands",
                "North Macedonia",
                "Georgia",
                "Finland",
                "Austria",
                "Poland",
                "Serbia",
                "Latvia",
                "Kosovo",
                "Israel",
            ],
            "away_team": [
                "Croatia",
                "Lithuania",
                "Iceland",
                "Hungary",
                "Switzerland",
                "Belgium",
                "Netherlands",
                "Slovenia",
                "Bosnia & Herzegovina",
                "Slovakia",
                "Germany",
                "Portugal",
                "Spain",
                "Italy",
                "Ukraine",
                "Romania",
            ],
        }
    )

    pred = model.predict(matches)
    proba = model.predict_proba(matches)
