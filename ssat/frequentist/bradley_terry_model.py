# %%
"""This module contains the implementation of the Bradley-Terry model for sports betting."""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.typing import NDArray
from scipy.optimize import minimize

from ssat.frequentist.base_model import BaseModel


class BradleyTerry(BaseModel):
    """Bradley-Terry model for predicting match outcomes with scikit-learn-like API.

    A probabilistic model that estimates team ratings and predicts match outcomes
    using maximum likelihood estimation. The model combines logistic regression for
    win probabilities with OLS regression for point spread predictions.

    Parameters
    ----------
    home_advantage : float, default=0.1
        Initial value for home advantage parameter.

    Attributes:
    ----------
    teams_ : np.ndarray
        Unique team identifiers
    n_teams_ : int
        Number of teams in the dataset
    team_map_ : Dict[str, int]
        Mapping of team names to indices
    params_ : np.ndarray
        Optimized model parameters after fitting
        [0:n_teams_] - Team ratings
        [-1] - Home advantage parameter
    intercept_ : float
        Point spread model intercept
    spread_coef_ : float
        Point spread model coefficient
    spread_error_ : float
        Standard error of spread predictions
    """

    NAME = "BT"

    def __init__(self, home_advantage: float = 0.1) -> None:
        """Initialize Bradley-Terry model."""
        self.home_advantage_ = home_advantage
        self.is_fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.Series],
        y: Union[np.ndarray, pd.Series],
        Z: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "BradleyTerry":
        """Fit the Bradley-Terry model.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing match data with two columns:
            - First column: Home team names (string)
            - Second column: Away team names (string)
        y : Union[np.ndarray, pd.Series]
            Goal differences (home - away).
        Z : Optional[pd.DataFrame], default=None
            BradleyTerry does not use Z, but it is included for compatibility.
        weights : Optional[np.ndarray], default=None
            Weights for rating optimization
        **kwargs : dict
            Additional optimization parameters (ftol, maxiter, etc.)

        Returns:
        -------
        self : BradleyTerry
            Fitted model
        """
        self._validate_X(X)

        # Extract team data using helper method
        self.home_team, self.away_team = self._extract_teams(X)
        self.goal_diff = np.asarray(y)

        # Vectorized result calculation
        self.result = np.sign(self.goal_diff).astype(int)

        # Team setup
        self.teams_ = np.unique(np.concatenate([self.home_team, self.away_team]))
        self.n_teams = len(self.teams_)
        self.team_map = {team: idx for idx, team in enumerate(self.teams_)}

        # Get team indices using helper method
        self.home_idx, self.away_idx = self._get_team_indices(
            self.home_team, self.away_team
        )

        # Set weights
        n_matches = len(self.goal_diff)
        self.weights = np.ones(n_matches) if weights is None else weights

        # Initialize parameters
        self._init_parameters()

        # Optimize parameters
        self.params = self._optimize_parameters(**kwargs)

        # Fit point spread model
        rating_diff = self._get_rating_difference()
        (self.intercept, self.spread_coef), self.spread_error_ = self._fit_ols(
            self.goal_diff, rating_diff
        )

        self.is_fitted = True
        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        Z: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        point_spread: int = 0,
    ) -> pd.DataFrame:
        """Generate predictions for new data.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Team pairs data with exactly 2 columns: [home_team, away_team]
        Z : Optional[Union[pd.DataFrame, np.ndarray]], default=None
            Additional match-level features (not used for prediction data)
        point_spread : int, default=0
            Point spread adjustment applied to goal differences

        Returns:
        -------
        np.ndarray
            Predicted goal differences with point_spread adjustment
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before making predictions")

        # Extract teams and get indices using helper methods
        home_teams, away_teams = self._extract_teams(X)
        home_idx, away_idx = self._get_team_indices(home_teams, away_teams)

        # Calculate all rating differences at once
        rating_diffs = self._get_rating_difference(home_idx, away_idx)

        # Calculate all predicted spreads vectorially
        predictions = self.intercept + self.spread_coef * rating_diffs

        return self._format_predictions(
            X,
            predictions,
            col_names=["goal_diff"],
        )

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        Z: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        point_spread: int = 0,
        include_draw: bool = True,
        outcome: Optional[str] = None,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Generate probability predictions for new data.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Team pairs data with exactly 2 columns: [home_team, away_team]
        Z : Optional[Union[pd.DataFrame, np.ndarray]], default=None
            Additional match-level features
        point_spread : int, default=0
            Point spread adjustment
        include_draw : bool, default=True
            Whether to include draw probability
        outcome : Optional[str], default=None
            Specific outcome to predict ('home', 'away', 'draw')
        threshold : float, default=0.5
            Threshold for draw prediction (for API consistency)

        Returns:
        -------
        np.ndarray
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before making predictions")

        if outcome not in [None, "home", "away", "draw"]:
            raise ValueError("outcome must be None, 'home', 'away', or 'draw'")

        if not include_draw and outcome == "draw":
            raise ValueError("Cannot predict draw when include_draw=False")

        predictions = self.predict(X, Z, point_spread=0).to_numpy().reshape(-1, 1)
        thresholds = np.array([point_spread + threshold, -point_spread - threshold])
        thresholds = np.tile(thresholds, (len(predictions), 1))

        if include_draw:
            probs = stats.norm.cdf(thresholds, predictions, self.spread_error_)
            home_probs = 1 - probs[:, 0]
            draw_probs = probs[:, 0] - probs[:, 1]
            away_probs = probs[:, 1]
        else:
            home_probs = 1 - stats.norm.cdf(
                point_spread, predictions, self.spread_error_
            )
            away_probs = 1 - home_probs

        # Handle specific outcome requests
        if outcome == "home":
            return self._format_predictions(X, home_probs, col_names=["home"])
        elif outcome == "away":
            return self._format_predictions(X, away_probs, col_names=["away"])
        elif outcome == "draw":
            return self._format_predictions(X, draw_probs, col_names=["draw"])

        # Handle include_draw parameter
        if include_draw:
            # Return all three probabilities
            result = np.stack([home_probs, draw_probs, away_probs]).T
            col_names = ["home", "draw", "away"]
        else:
            # Return only home/away, renormalized
            home_away_sum = home_probs + away_probs
            home_probs_norm = home_probs / home_away_sum
            away_probs_norm = away_probs / home_away_sum
            result = np.stack([home_probs_norm, away_probs_norm]).T
            col_names = ["home", "away"]

        return self._format_predictions(X, result, col_names=col_names)

    def _log_likelihood(self, params: NDArray[np.float64]) -> np.float64:
        """Calculate negative log likelihood for parameter optimization."""
        ratings: NDArray[np.float64] = params[:-1]
        home_advantage: np.float64 = params[-1]
        log_likelihood: np.float64 = np.float64(0.0)

        # Vectorized rating difference calculation
        win_probs: NDArray[np.float64] = self._logit_transform(
            home_advantage + ratings[self.home_idx] - ratings[self.away_idx]
        )

        # Vectorized calculation
        win_mask: NDArray[np.bool_] = self.result == 1
        loss_mask: NDArray[np.bool_] = self.result == -1
        draw_mask: NDArray[np.bool_] = ~(win_mask | loss_mask)

        log_likelihood += np.sum(self.weights[win_mask] * np.log(win_probs[win_mask]))
        log_likelihood += np.sum(
            self.weights[loss_mask] * np.log(1 - win_probs[loss_mask])
        )
        log_likelihood += np.sum(
            self.weights[draw_mask]
            * (np.log(win_probs[draw_mask]) + np.log(1 - win_probs[draw_mask]))
        )

        return -log_likelihood

    def _init_parameters(self) -> None:
        """Smart parameter initialization based on win rates."""
        # Initialize parameter array
        self.params = np.zeros(self.n_teams + 1)
        self.params[-1] = self.home_advantage_

        # Initialize team ratings based on win rates
        wins = np.zeros(self.n_teams)
        games = np.zeros(self.n_teams)

        for i, team_idx in enumerate(self.home_idx):
            games[team_idx] += 1
            if self.result[i] > 0:  # Home win
                wins[team_idx] += 1

        for i, team_idx in enumerate(self.away_idx):
            games[team_idx] += 1
            if self.result[i] < 0:  # Away win
                wins[team_idx] += 1

        # Avoid division by zero and extreme values
        win_rates = np.where(games > 0, wins / games, 0.5)
        win_rates = np.clip(win_rates, 0.01, 0.99)  # Avoid extreme values

        # Convert win rates to log-odds for better initialization
        self.params[:-1] = np.log(win_rates / (1 - win_rates))

    def _optimize_parameters(self, **kwargs) -> NDArray[np.float64]:
        """Optimize model parameters with fallback methods."""
        # Default options
        default_options = {"ftol": 1e-8, "maxiter": 500}
        options = {**default_options, **kwargs}

        objective = lambda p: self._log_likelihood(p) / len(self.result)
        methods = ["SLSQP", "L-BFGS-B"]

        # Try optimization methods in sequence
        for method in methods:
            result = minimize(
                fun=objective,
                x0=self.params,
                method=method,
                options=options,
            )
            if result.success:
                return result.x

        # If no method succeeded, return the last result anyway
        return result.x

    def _get_rating_difference(
        self,
        home_idx: Union[int, NDArray[np.int_], None] = None,
        away_idx: Union[int, NDArray[np.int_], None] = None,
    ) -> NDArray[np.float64]:
        """Calculate rating difference between teams."""
        if home_idx is None:
            home_idx, away_idx = self.home_idx, self.away_idx

        ratings: NDArray[np.float64] = self.params[:-1]
        home_advantage: np.float64 = self.params[-1]
        return self._logit_transform(
            home_advantage + ratings[home_idx] - ratings[away_idx]
        )

    def _fit_ols(self, y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit OLS no weights."""
        X = np.column_stack((np.ones(len(X)), X))

        # Use more efficient matrix operations
        coefficients = np.linalg.solve(X.T @ X, X.T @ y)
        residuals = y - X @ coefficients
        sse = np.sum((residuals**2))
        std_error = np.sqrt(sse / (X.shape[0] - X.shape[1]))

        return coefficients, std_error

    def get_team_ratings(self) -> pd.DataFrame:
        """Get team ratings as a DataFrame.

        Returns:
        -------
        pd.DataFrame
            DataFrame with team ratings
        """
        self._check_is_fitted()
        return pd.DataFrame(
            data=self.params,
            index=list(self.teams_) + ["Home Advantage"],
            columns=["rating"],
        )

    def get_params(self) -> dict:
        """Get the current parameters of the model.

        Returns:
        -------
        dict
            Dictionary containing model parameters
        """
        return {
            "home_advantage": self.home_advantage_,
            "params": self.params,
            "is_fitted": self.is_fitted,
        }

    def set_params(self, params: dict) -> None:
        """Set parameters for the model.

        Parameters
        ----------
        params : dict
            Dictionary containing model parameters, as returned by get_params()
        """
        self.home_advantage_ = params["home_advantage"]
        self.params = params["params"]
        self.is_fitted = params["is_fitted"]
