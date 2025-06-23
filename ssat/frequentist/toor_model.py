# %%
"""This file contains the implementation of the Team OLS Optimized Rating (TOOR) model for predicting sports match outcomes."""

from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize

from ssat.frequentist.bradley_terry_model import BradleyTerry


class TOOR(BradleyTerry):
    """Team OLS Optimized Rating (TOOR) model with scikit-learn-like API.

    An extension of the Bradley-Terry model that uses team-specific coefficients
    for more accurate point spread predictions. The model combines traditional
    Bradley-Terry ratings with a team-specific regression approach.

    Parameters
    ----------
    home_advantage : float, default=0.1
        Initial value for home advantage parameter.

    Attributes:
    ----------
    Inherits all attributes from BradleyTerry plus:
    home_advantage_coef_ : float
        Home advantage coefficient for spread prediction
    home_team_coef_ : float
        Home team rating coefficient
    away_team_coef_ : float
        Away team rating coefficient
    """

    NAME = "TOOR"

    def __init__(self, home_advantage: float = 0.1) -> None:
        """Initialize TOOR model."""
        super().__init__(home_advantage=home_advantage)
        self.home_advantage = home_advantage
        self.home_team_coef = None
        self.away_team_coef = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[np.ndarray, pd.Series],
        Z: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "TOOR":
        """Fit the TOOR model.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing match data with two columns:
            - First column: Home team names (string)
            - Second column: Away team names (string)
        y : Union[np.ndarray, pd.Series]
            Goal differences (home - away).
        Z : Optional[pd.DataFrame], default=None
            Additional data for the model, such as home_goals and away_goals.
            No column name checking is performed, only dimension validation.
        weights : Optional[np.ndarray], default=None
            Weights for rating optimization
        **kwargs : dict
            Additional optimization parameters (ftol, maxiter, etc.)

        Returns:
        -------
        self : TOOR
            Fitted model
        """
        # First fit the Bradley-Terry model to get team ratings
        super().fit(X, y, Z, weights, **kwargs)

        # Optimize the three parameters using least squares with kwargs support
        default_options = {"ftol": 1e-8, "maxiter": 500}
        options = {**default_options, **kwargs}

        initial_guess = np.array([0.1, 1.0, -1.0])
        methods = ["L-BFGS-B", "SLSQP"]

        # Try optimization methods in sequence
        for method in methods:
            result = minimize(
                self._sse_function,
                initial_guess,
                method=method,
                options=options,
            )
            if result.success:
                break

        # Store the optimized coefficients
        self.home_advantage = result.x[0]  # home advantage
        self.home_team_coef = result.x[1]  # home team coefficient
        self.away_team_coef = result.x[2]  # away team coefficient

        # Calculate spread error
        predictions = (
            self.home_advantage
            + self.home_team_coef * self.params[self.home_idx]
            + self.away_team_coef * self.params[self.away_idx]
        )
        residuals = self.goal_diff - predictions
        sse = np.sum((residuals**2))
        self.spread_error_ = np.sqrt(sse / (len(self.goal_diff) - 3))

        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        Z: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        point_spread: int = 0,
    ) -> pd.DataFrame:
        """Predict point spreads for matches using team-specific coefficients.

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
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before making predictions")

        # Extract teams and get indices using helper methods
        home_teams, away_teams = self._extract_teams(X)
        home_idx, away_idx = self._get_team_indices(home_teams, away_teams)

        # Get team ratings vectorially
        home_ratings = self.params[home_idx]
        away_ratings = self.params[away_idx]

        # Calculate all predicted spreads vectorially using team-specific coefficients
        predicted_spreads = (
            self.home_advantage
            + self.home_team_coef * home_ratings
            + self.away_team_coef * away_ratings
        ) + point_spread

        return self._format_predictions(
            X,
            predicted_spreads,
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

    def get_params(self) -> dict:
        """Get the current parameters of the model.

        Returns:
        -------
        dict
            Dictionary containing model parameters
        """
        return {
            "home_advantage": self.home_advantage,
            "home_team_coef": self.home_team_coef,
            "away_team_coef": self.away_team_coef,
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
        self.home_advantage = params["home_advantage"]
        self.home_team_coef = params["home_team_coef"]
        self.away_team_coef = params["away_team_coef"]
        self.params = params["params"]
        self.is_fitted = params["is_fitted"]

    def get_team_ratings(self) -> pd.DataFrame:
        """Get team ratings as a DataFrame with home and away coefficients.

        Returns:
        -------
        pd.DataFrame
            DataFrame with team ratings multiplied by home and away coefficients
        """
        self._check_is_fitted()
        df = pd.DataFrame(
            {
                "home": self.params[:-1] * self.home_team_coef,
                "away": self.params[:-1] * self.away_team_coef,
            },
            index=self.teams_,
        )
        df.loc["Home Advantage"] = [self.home_advantage, np.nan]
        return df

    def _sse_function(self, parameters: np.ndarray) -> float:
        """Calculate sum of squared errors for parameter optimization.

        Parameters
        ----------
        parameters : np.ndarray
            Array of [home_advantage, home_team_coef, away_team_coef]

        Returns:
        -------
        float
            Sum of squared errors
        """
        home_adv, home_team_coef, away_team_coef = parameters

        # Get logistic ratings from Bradley-Terry optimization
        logistic_ratings = self.params[:-1]  # Exclude home advantage parameter

        # Calculate predictions
        predictions = (
            home_adv
            + home_team_coef * logistic_ratings[self.home_idx]
            + away_team_coef * logistic_ratings[self.away_idx]
        )

        # Calculate weighted squared errors
        errors = self.goal_diff - predictions
        sse = np.sum(errors**2 * self.weights)

        return sse
