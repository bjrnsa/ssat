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
        self.home_advantage_ = home_advantage
        self.home_team_coef_ = None
        self.away_team_coef_ = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        Z: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None,
    ) -> "TOOR":
        """Fit the TOOR model.

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
        self : TOOR
            Fitted model
        """
        try:
            # First fit the Bradley-Terry model to get team ratings
            super().fit(X, y, Z, weights)

            # Optimize the three parameters using least squares
            initial_guess = np.array([0.1, 1.0, -1.0])
            result = minimize(
                self._sse_function,
                initial_guess,
                method="L-BFGS-B",
                options={"ftol": 1e-10, "maxiter": 200},
            )

            # Store the optimized coefficients
            self.home_advantage_ = result.x[0]  # home advantage
            self.home_team_coef_ = result.x[1]  # home team coefficient
            self.away_team_coef_ = result.x[2]  # away team coefficient

            # Calculate spread error
            predictions = (
                self.home_advantage_
                + self.home_team_coef_ * self.params_[self.home_idx_]
                + self.away_team_coef_ * self.params_[self.away_idx_]
            )
            residuals = self.spread_ - predictions
            sse = np.sum((residuals**2))
            self.spread_error_ = np.sqrt(sse / (X.shape[0] - X.shape[1]))

            return self

        except Exception as e:
            self.is_fitted_ = False
            raise ValueError(f"Model fitting failed: {str(e)}") from e

    def predict(
        self,
        X: pd.DataFrame,
        Z: Optional[pd.DataFrame] = None,
        point_spread: int = 0,
    ) -> np.ndarray:
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

            # Get team ratings
            home_rating = self.params_[self.team_map_[home_team]]
            away_rating = self.params_[self.team_map_[away_team]]

            # Calculate predicted spread using team-specific coefficients
            predicted_spreads[i] = (
                self.home_advantage_
                + self.home_team_coef_ * home_rating
                + self.away_team_coef_ * away_rating
            )

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

            # Get team ratings
            home_rating = self.params_[self.team_map_[home_team]]
            away_rating = self.params_[self.team_map_[away_team]]

            # Calculate predicted spread using team-specific coefficients
            predicted_spread = (
                self.home_advantage_
                + self.home_team_coef_ * home_rating
                + self.away_team_coef_ * away_rating
            )

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

    def get_params(self) -> dict:
        """Get the current parameters of the model.

        Returns:
        -------
        dict
            Dictionary containing model parameters
        """
        return {
            "home_advantage": self.home_advantage_,
            "home_team_coef": self.home_team_coef_,
            "away_team_coef": self.away_team_coef_,
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
        self.home_team_coef_ = params["home_team_coef"]
        self.away_team_coef_ = params["away_team_coef"]
        self.params_ = params["params"]
        self.is_fitted_ = params["is_fitted"]

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
                "home": self.params_[:-1] * self.home_team_coef_,
                "away": self.params_[:-1] * self.away_team_coef_,
            },
            index=self.teams_,
        )
        df.loc["Home Advantage"] = [self.home_advantage_, np.nan]
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
        logistic_ratings = self.params_[:-1]  # Exclude home advantage parameter

        # Calculate predictions
        predictions = (
            home_adv
            + home_team_coef * logistic_ratings[self.home_idx_]
            + away_team_coef * logistic_ratings[self.away_idx_]
        )

        # Calculate weighted squared errors
        errors = self.spread_ - predictions
        sse = np.sum(errors**2 * self.weights_)

        return sse
