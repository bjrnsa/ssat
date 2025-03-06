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
        self.is_fitted_ = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        Z: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None,
    ) -> "BradleyTerry":
        """Fit the Bradley-Terry model.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing match data with two columns:
            - First column: Home team names (string)
            - Second column: Away team names (string)
            If y is None, X must have a third column with goal differences.
        y : Optional[Union[np.ndarray, pd.Series]], default=None
            Goal differences (home - away). If provided, this will be used instead of
            the third column in X.
        Z : Optional[pd.DataFrame], default=None
            Additional data for the model, such as home_goals and away_goals.
            No column name checking is performed, only dimension validation.
        weights : Optional[np.ndarray], default=None
            Weights for rating optimization

        Returns:
        -------
        self : BradleyTerry
            Fitted model
        """
        try:
            # Validate input dimensions and types
            self._validate_X(X)

            # Validate Z dimensions if provided
            if Z is not None and len(Z) != len(X):
                raise ValueError("Z must have the same number of rows as X")

            # Extract team data (first two columns)
            self.home_team_ = X.iloc[:, 0].to_numpy()
            self.away_team_ = X.iloc[:, 1].to_numpy()

            # Handle goal difference (y)
            if y is not None:
                self.spread_ = np.asarray(y)
            elif X.shape[1] >= 3:
                self.spread_ = X.iloc[:, 2].to_numpy()
            else:
                raise ValueError(
                    "Either y or a third column in X with goal differences must be provided"
                )

            # Validate goal difference
            if not np.issubdtype(self.spread_.dtype, np.number):
                raise ValueError("Goal differences must be numeric")

            # Derive result from spread_
            self.result_ = np.zeros_like(self.spread_, dtype=int)
            self.result_[self.spread_ > 0] = 1
            self.result_[self.spread_ < 0] = -1

            # Team setup
            self.teams_ = np.unique(np.concatenate([self.home_team_, self.away_team_]))
            self.n_teams_ = len(self.teams_)
            self.team_map_ = {team: idx for idx, team in enumerate(self.teams_)}

            # Create team indices
            self.home_idx_ = np.array(
                [self.team_map_[team] for team in self.home_team_]
            )
            self.away_idx_ = np.array(
                [self.team_map_[team] for team in self.away_team_]
            )

            # Set weights
            n_matches = len(X)
            self.weights_ = np.ones(n_matches) if weights is None else weights

            # Initialize parameters
            self.params_ = np.zeros(self.n_teams_ + 1)
            self.params_[-1] = self.home_advantage_

            # Optimize parameters
            self.params_ = self._optimize_parameters()

            # Fit point spread model
            rating_diff = self._get_rating_difference()
            (self.intercept_, self.spread_coef_), self.spread_error_ = self._fit_ols(
                self.spread_, rating_diff
            )

            self.is_fitted_ = True
            return self

        except Exception as e:
            self.is_fitted_ = False
            raise ValueError(f"Model fitting failed: {str(e)}") from e

    def predict(
        self, X: pd.DataFrame, Z: Optional[pd.DataFrame] = None, point_spread: int = 0
    ) -> np.ndarray:
        """Predict point spreads for matches.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing match data with two columns:
            - First column: Home team names (string)
            - Second column: Away team names (string)
        Z : Optional[pd.DataFrame], default=None
            Additional data for prediction. No column name checking is performed,
            only dimension validation.
        point_spread : float, default=0.0
            Point spread adjustment

        Returns:
        -------
        np.ndarray
            Predicted point spreads (goal differences)
        """
        self._check_is_fitted()
        self._validate_X(X, fit=False)

        # Validate Z dimensions if provided
        if Z is not None and len(Z) != len(X):
            raise ValueError("Z must have the same number of rows as X")

        home_teams = X.iloc[:, 0].to_numpy()
        away_teams = X.iloc[:, 1].to_numpy()

        predicted_spreads = np.zeros(len(X))

        for i, (home_team, away_team) in enumerate(zip(home_teams, away_teams)):
            # Validate teams
            self._validate_teams([home_team, away_team])

            # Calculate rating difference
            rating_diff = self._get_rating_difference(
                home_idx=self.team_map_[home_team],
                away_idx=self.team_map_[away_team],
            )

            # Calculate predicted spread
            predicted_spreads[i] = self.intercept_ + self.spread_coef_ * rating_diff

        return predicted_spreads

    def predict_proba(
        self,
        X: pd.DataFrame,
        Z: Optional[pd.DataFrame] = None,
        point_spread: int = 0,
        include_draw: bool = True,
        outcome: Optional[str] = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict match outcome probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing match data with two columns:
            - First column: Home team names (string)
            - Second column: Away team names (string)
        Z : Optional[pd.DataFrame], default=None
            Additional data for prediction. No column name checking is performed,
            only dimension validation.
        point_spread : float, default=0.0
            Point spread adjustment
        include_draw : bool, default=True
            Whether to include draw probability
        outcome: Optional[str], default=None
            Outcome to predict (home, draw, away)
        threshold: float, default=0.5
            Threshold for predicting draw outcome

        Returns:
        -------
        np.ndarray
            Array of shape (n_samples, n_classes) with probabilities
            If include_draw=True: [home, draw, away]
            If include_draw=False: [home, away]
        """
        self._check_is_fitted()
        self._validate_X(X, fit=False)

        # Validate Z dimensions if provided
        if Z is not None and len(Z) != len(X):
            raise ValueError("Z must have the same number of rows as X")

        home_teams = X.iloc[:, 0].to_numpy()
        away_teams = X.iloc[:, 1].to_numpy()

        if outcome is None:
            n_classes = 3 if include_draw else 2
            probabilities = np.zeros((len(X), n_classes))
        else:
            probabilities = np.zeros((len(X),))

        for i, (home_team, away_team) in enumerate(zip(home_teams, away_teams)):
            # Validate teams
            self._validate_teams([home_team, away_team])

            rating_diff = self._get_rating_difference(
                home_idx=self.team_map_[home_team],
                away_idx=self.team_map_[away_team],
            )

            # Calculate predicted spread
            predicted_spread = self.intercept_ + self.spread_coef_ * rating_diff

            # Calculate probabilities
            if include_draw:
                thresholds = np.array(
                    [point_spread + threshold, -point_spread - threshold]
                )
                probs = stats.norm.cdf(thresholds, predicted_spread, self.spread_error_)
                prob_home, prob_draw, prob_away = (
                    1 - probs[0],
                    probs[0] - probs[1],
                    probs[1],
                )
            else:
                prob_home = 1 - stats.norm.cdf(
                    point_spread, predicted_spread, self.spread_error_
                )
                prob_home, prob_away = prob_home, 1 - prob_home

            if outcome is not None:
                if outcome == "home":
                    probabilities[i] = prob_home
                elif outcome == "away":
                    probabilities[i] = prob_away
                elif outcome == "draw":
                    probabilities[i] = prob_draw
            else:
                if include_draw:
                    probabilities[i] = [prob_home, prob_draw, prob_away]
                else:
                    probabilities[i] = [prob_home, prob_away]

        if outcome:
            return probabilities.reshape(-1)

        return probabilities

    def _log_likelihood(self, params: NDArray[np.float64]) -> np.float64:
        """Calculate negative log likelihood for parameter optimization."""
        ratings: NDArray[np.float64] = params[:-1]
        home_advantage: np.float64 = params[-1]
        log_likelihood: np.float64 = np.float64(0.0)

        # Precompute home and away ratings
        home_ratings: NDArray[np.float64] = ratings[self.home_idx_]
        away_ratings: NDArray[np.float64] = ratings[self.away_idx_]
        win_probs: NDArray[np.float64] = self._logit_transform(
            home_advantage + home_ratings - away_ratings
        )

        # Vectorized calculation
        win_mask: NDArray[np.bool_] = self.result_ == 1
        loss_mask: NDArray[np.bool_] = self.result_ == -1
        draw_mask: NDArray[np.bool_] = ~(win_mask | loss_mask)

        log_likelihood += np.sum(self.weights_[win_mask] * np.log(win_probs[win_mask]))
        log_likelihood += np.sum(
            self.weights_[loss_mask] * np.log(1 - win_probs[loss_mask])
        )
        log_likelihood += np.sum(
            self.weights_[draw_mask]
            * (np.log(win_probs[draw_mask]) + np.log(1 - win_probs[draw_mask]))
        )

        return -log_likelihood

    def _optimize_parameters(self) -> NDArray[np.float64]:
        """Optimize model parameters using SLSQP."""
        result = minimize(
            fun=lambda p: self._log_likelihood(p) / len(self.result_),
            x0=self.params_,
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 200},
        )
        return result.x

    def _get_rating_difference(
        self,
        home_idx: Union[int, NDArray[np.int_], None] = None,
        away_idx: Union[int, NDArray[np.int_], None] = None,
    ) -> NDArray[np.float64]:
        """Calculate rating difference between teams."""
        if home_idx is None:
            home_idx, away_idx = self.home_idx_, self.away_idx_

        ratings: NDArray[np.float64] = self.params_[:-1]
        home_advantage: np.float64 = self.params_[-1]
        return self._logit_transform(
            home_advantage + ratings[home_idx] - ratings[away_idx]
        )

    def _logit_transform(
        self, x: Union[float, NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """Apply logistic transformation."""
        x_array = np.asarray(x, dtype=np.float64)
        return 1 / (1 + np.exp(-x_array))

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
            data=self.params_,
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
            "params": self.params_,
            "is_fitted": self.is_fitted_,
        }

    def set_params(self, params: dict) -> None:
        """Set parameters for the model.

        Parameters
        ----------
        params : dict
            Dictionary containing model parameters, as returned by get_params()
        """
        self.home_advantage_ = params["home_advantage"]
        self.params_ = params["params"]
        self.is_fitted_ = params["is_fitted"]
