# %%
"""This file contains the implementation of the Points Rating Prediction (PRP) model for predicting sports match outcomes."""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from ssat.frequentist.zsd_model import ZSD


class PRP(ZSD):
    """Points Rating Prediction (PRP) model for predicting sports match outcomes with scikit-learn-like API.

    This model extends the ZSD model and estimates offensive and defensive ratings for each team,
    plus adjustment factors. The model uses a simple additive approach for score prediction:
    score = adj_factor + offense - defense + avg_score

    Parameters
    ----------
    None

    Attributes:
    ----------
    teams : np.ndarray
        Unique team identifiers
    n_teams : int
        Number of teams in the dataset
    team_map : Dict[str, int]
        Mapping of team names to indices
    home_idx : np.ndarray
        Indices of home teams
    away_idx : np.ndarray
        Indices of away teams
    weights : np.ndarray
        Weights for rating optimization
    is_fitted : bool
        Whether the model has been fitted
    params : np.ndarray
        Optimized model parameters after fitting
        [0:n_teams] - Offensive ratings
        [n_teams:2*n_teams] - Defensive ratings
        [-2:] - Home/away adjustment factors
    """

    NAME = "PRP"

    def __init__(self) -> None:
        """Initialize PRP model."""
        super().__init__()

    def _predict_scores(
        self,
        home_idx: Union[int, np.ndarray, None] = None,
        away_idx: Union[int, np.ndarray, None] = None,
        offense_ratings: Union[np.ndarray, None] = None,
        defense_ratings: Union[np.ndarray, None] = None,
        factors: Union[np.ndarray, None] = None,
    ) -> Dict[str, np.ndarray]:
        """Calculate predicted scores using offensive/defensive ratings.

        Parameters
        ----------
        home_idx : Union[int, np.ndarray, None], default=None
            Index(es) of home team(s)
        away_idx : Union[int, np.ndarray, None], default=None
            Index(es) of away team(s)
        offense_ratings : Union[np.ndarray, None], default=None
            Optional offensive ratings to use
        defense_ratings : Union[np.ndarray, None], default=None
            Optional defensive ratings to use
        factors : Union[np.ndarray, None], default=None
            Optional (home_factor, away_factor) tuple

        Returns:
        -------
        Dict[str, np.ndarray]
            Dict with 'home' and 'away' predicted scores
        """
        if factors is None:
            factors = self.params[-2:]

        ratings = self._get_team_ratings(
            home_idx, away_idx, offense_ratings, defense_ratings
        )

        return {
            "home": self._goal_func(
                factors[0],
                ratings["home_offense"],
                ratings["away_defense"],
                factors[1],
                factor=0.5,
            ),
            "away": self._goal_func(
                factors[0],
                ratings["away_offense"],
                ratings["home_defense"],
                factors[1],
                factor=-0.5,
            ),
        }

    def _get_team_ratings(
        self,
        home_idx: Union[int, np.ndarray, None],
        away_idx: Union[int, np.ndarray, None],
        offense_ratings: Union[np.ndarray, None] = None,
        defense_ratings: Union[np.ndarray, None] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract offensive/defensive ratings from parameters.

        Parameters
        ----------
        home_idx : Union[int, np.ndarray, None]
            Index(es) of home team(s)
        away_idx : Union[int, np.ndarray, None]
            Index(es) of away team(s)
        offense_ratings : Union[np.ndarray, None], default=None
            Optional offensive ratings to use
        defense_ratings : Union[np.ndarray, None], default=None
            Optional defensive ratings to use

        Returns:
        -------
        Dict[str, np.ndarray]
            Dictionary with team ratings
        """
        if offense_ratings is None and defense_ratings is None:
            offense_ratings, defense_ratings = np.split(
                self.params[: 2 * self.n_teams], 2
            )

        assert offense_ratings is not None and defense_ratings is not None, (
            "offense_ratings and defense_ratings must be provided"
        )

        if home_idx is None:
            home_idx, away_idx = self.home_idx, self.away_idx

        return {
            "home_offense": offense_ratings[home_idx],
            "home_defense": defense_ratings[home_idx],
            "away_offense": offense_ratings[away_idx],
            "away_defense": defense_ratings[away_idx],
        }

    def _goal_func(
        self,
        home_advantage: float,
        offense_ratings: np.ndarray,
        defense_ratings: np.ndarray,
        avg_score: float,
        factor: float = 0.5,
    ) -> pd.DataFrame:
        """Calculate score prediction.

        Parameters
        ----------
        home_advantage : float
            Home advantage factor
        offense_ratings : np.ndarray
            Team's offensive ratings
        defense_ratings : np.ndarray
            Opponent's defensive ratings
        avg_score : float
            Average score factor
        factor : float, default=0.5
            Home/away adjustment multiplier

        Returns:
        -------
        np.ndarray
            Predicted score
        """
        return (
            (factor * home_advantage) + (offense_ratings + defense_ratings) + avg_score
        )

    def get_team_ratings(self) -> pd.DataFrame:
        """Get team ratings as a DataFrame.

        Returns:
        -------
        pd.DataFrame
            Team ratings with columns ['team', 'offense', 'defense']
        """
        self._check_is_fitted()

        offense_ratings = self.params[: self.n_teams]
        defense_ratings = self.params[self.n_teams : 2 * self.n_teams]

        return pd.DataFrame(
            {
                "team": self.teams,
                "offense": offense_ratings,
                "defense": defense_ratings,
            }
        ).set_index("team")

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[np.ndarray, pd.Series],
        Z: pd.DataFrame,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "PRP":
        """Fit the PRP model.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing match data with two columns:
            - First column: Home team names (string)
            - Second column: Away team names (string)
        y : Union[np.ndarray, pd.Series]
            Goal differences (home - away).
        Z : pd.DataFrame
            Additional data for the model, such as home_goals and away_goals.
            No column name checking is performed, only dimension validation.
        weights : Optional[np.ndarray], default=None
            Weights for rating optimization
        **kwargs : dict
            Additional optimization parameters (ftol, maxiter, etc.)

        Returns:
        -------
        self : PRP
            Fitted model
        """
        # Fit the model
        super().fit(X, y, Z, weights, **kwargs)

        return self
