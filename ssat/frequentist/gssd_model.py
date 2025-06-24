# %%
"""Generalized Scores Standard Deviation (GSSD) model."""

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ssat.frequentist.base_model import BaseModel


class GSSD(BaseModel):
    """Generalized Scores Standard Deviation (GSSD) model with scikit-learn-like API.

    A model that predicts match outcomes using team-specific offensive and defensive ratings.
    The model uses weighted OLS regression to estimate team performance parameters and
    calculates win/draw/loss probabilities using a normal distribution.

    Parameters
    ----------
    None

    Attributes:
    ----------
    teams_ : np.ndarray
        Unique team identifiers
    team_ratings_ : Dict[str, np.ndarray]
        Dictionary mapping teams to their offensive/defensive ratings
    is_fitted_ : bool
        Whether the model has been fitted
    spread_error_ : float
        Standard error of the model predictions
    intercept_ : float
        Model intercept term
    pfh_coef_ : float
        Coefficient for home team's offensive rating
    pah_coef_ : float
        Coefficient for home team's defensive rating
    pfa_coef_ : float
        Coefficient for away team's offensive rating
    paa_coef_ : float
        Coefficient for away team's defensive rating
    """

    NAME = "GSSD"

    def __init__(self) -> None:
        """Initialize GSSD model."""
        super().__init__()
        self.is_fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[np.ndarray, pd.Series],
        Z: Union[pd.DataFrame, np.ndarray],
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "GSSD":
        """Fit the GSSD model.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing match data with two columns:
            - First column: Home team names (string)
            - Second column: Away team names (string)
        y : Union[np.ndarray, pd.Series]
            Goal differences (home - away).
        Z : Union[pd.DataFrame, np.ndarray]
            Additional data for the model, such as home_goals and away_goals.
            No column name checking is performed, only dimension validation.
        weights : Optional[np.ndarray], default=None
            Weights for rating optimization
        **kwargs : dict
            Additional optimization parameters (ftol, maxiter, etc.)

        Returns:
        -------
        self : GSSD
            Fitted model
        """
        # Validate input dimensions and types
        self._validate_X(X)
        self._validate_Z(X, Z, True)

        # Extract team data using helper method
        self.home_team, self.away_team = self._extract_teams(X)

        # Handle goal difference (y)
        self.goal_diff = np.asarray(y)

        # Extract home_goals and away_goals from Z
        if isinstance(Z, np.ndarray):
            self.home_goals = Z[:, 0]
            self.away_goals = Z[:, 1]
        else:
            self.home_goals = Z.iloc[:, 0].to_numpy()
            self.away_goals = Z.iloc[:, 1].to_numpy()

        # Team setup
        self.teams = np.unique(np.concatenate([self.home_team, self.away_team]))
        self.n_teams = len(self.teams)
        self.team_map = {team: idx for idx, team in enumerate(self.teams)}

        # Get team indices using helper method
        self.home_idx, self.away_idx = self._get_team_indices(
            self.home_team, self.away_team
        )

        # Set weights
        n_matches = len(self.goal_diff)
        self.weights = np.ones(n_matches) if weights is None else weights

        # Calculate team statistics
        self._calculate_team_statistics()

        # Initialize and optimize parameters
        self._init_parameters()
        result = self._optimize_parameters(**kwargs)

        # Store model parameters
        self.intercept = result[0]
        self.pfh_coef = result[1]
        self.pah_coef = result[2]
        self.pfa_coef = result[3]
        self.paa_coef = result[4]

        # Calculate spread error
        features = np.column_stack((self.pfh, self.pah, self.pfa, self.paa))
        predictions = self._get_predictions(features)
        residuals = self.goal_diff - predictions
        sse = np.sum((residuals**2))
        self.spread_error_ = np.sqrt(sse / (len(self.goal_diff) - 5))

        self.is_fitted = True
        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        Z: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        point_spread: int = 0,
        format_predictions: bool = False,
    ) -> pd.DataFrame:
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
        if not self.is_fitted:
            raise ValueError("Model must be fit before making predictions")

        # Extract teams and get indices using helper methods
        home_teams, away_teams = self._extract_teams(X)
        home_idx, away_idx = self._get_team_indices(home_teams, away_teams)

        # Get team ratings vectorially
        team_ratings_array = np.array([self.team_ratings[team] for team in self.teams])

        home_off = team_ratings_array[home_idx, 0]  # pfh for home teams
        home_def = team_ratings_array[home_idx, 1]  # pah for home teams
        away_off = team_ratings_array[away_idx, 2]  # pfa for away teams
        away_def = team_ratings_array[away_idx, 3]  # paa for away teams

        # Calculate all predicted spreads vectorially
        predictions = (
            self.intercept
            + home_off * self.pfh_coef
            + home_def * self.pah_coef
            + away_off * self.pfa_coef
            + away_def * self.paa_coef
        ) + point_spread

        if format_predictions:
            return self._format_predictions(
                X,
                predictions,
                col_names=["goal_diff"],
            )
        return predictions

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        Z: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        point_spread: int = 0,
        include_draw: bool = True,
        outcome: Optional[str] = None,
        threshold: float = 0.5,
        format_predictions: bool = False,
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

        predictions = self.predict(X, Z, point_spread=0).reshape(-1, 1)
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

        if format_predictions:
            return self._format_predictions(X, result, col_names=col_names)

        return result

    def get_params(self) -> dict:
        """Get the current parameters of the model.

        Returns:
        -------
        dict
            Dictionary containing model parameters
        """
        return {
            "intercept": self.intercept,
            "pfh_coef": self.pfh_coef,
            "pah_coef": self.pah_coef,
            "pfa_coef": self.pfa_coef,
            "paa_coef": self.paa_coef,
            "params": self.team_ratings,
            "is_fitted": self.is_fitted,
        }

    def set_params(self, params: dict) -> None:
        """Set parameters for the model.

        Parameters
        ----------
        params : dict
            Dictionary containing model parameters, as returned by get_params()
        """
        self.intercept = params["intercept"]
        self.pfh_coef = params["pfh_coef"]
        self.pah_coef = params["pah_coef"]
        self.pfa_coef = params["pfa_coef"]
        self.paa_coef = params["paa_coef"]
        self.team_ratings = params["params"]
        self.is_fitted = params["is_fitted"]

    def get_team_ratings(self) -> pd.DataFrame:
        """Get team ratings as a DataFrame.

        Returns:
        -------
        pd.DataFrame
            DataFrame with team ratings and model coefficients
        """
        self._check_is_fitted()

        # Get team ratings
        ratings_df = pd.DataFrame(
            self.team_ratings, index=["pfh", "pah", "pfa", "paa"]
        ).T

        # Add coefficients as a new row
        coeffs = {
            "pfh": self.pfh_coef,
            "pah": self.pah_coef,
            "pfa": self.pfa_coef,
            "paa": self.paa_coef,
        }
        ratings_df.loc["Coefficients"] = pd.Series(coeffs)
        ratings_df.loc["Intercept"] = self.intercept

        return ratings_df.round(2)

    def _init_parameters(self) -> None:
        """Initialize model parameters with default values."""
        self.initial_params = np.array([0.1, 1.0, 1.0, -1.0, -1.0])

    def _optimize_parameters(self, **kwargs) -> pd.DataFrame:
        """Optimize model parameters with fallback methods.

        Parameters
        ----------
        **kwargs : dict
            Additional optimization parameters (ftol, maxiter, etc.)

        Returns:
        -------
        np.ndarray
            Optimized parameters [intercept, pfh_coef, pah_coef, pfa_coef, paa_coef]
        """
        # Default optimization options
        default_options = {"ftol": 1e-8, "maxiter": 500}
        options = {**default_options, **kwargs}
        methods = ["L-BFGS-B", "SLSQP"]

        # Try optimization methods in sequence
        for method in methods:
            result = minimize(
                self._sse_function,
                self.initial_params,
                method=method,
                options=options,
            )
            if result.success:
                return result.x

        # If no method succeeded, return the last result anyway
        return result.x

    def _sse_function(self, parameters: np.ndarray) -> float:
        """Calculate sum of squared errors for parameter optimization.

        Parameters
        ----------
        parameters : np.ndarray
            Array of [intercept, pfh_coef, pah_coef, pfa_coef, paa_coef]

        Returns:
        -------
        float
            Sum of squared errors
        """
        intercept, pfh_coef, pah_coef, pfa_coef, paa_coef = parameters

        # Vectorized calculation of predictions
        predictions = (
            intercept
            + pfh_coef * self.pfh
            + pah_coef * self.pah
            + pfa_coef * self.pfa
            + paa_coef * self.paa
        )

        # Calculate weighted squared errors
        errors = self.goal_diff - predictions
        sse = np.sum(self.weights * (errors**2))

        return sse

    def _get_predictions(self, features: np.ndarray) -> pd.DataFrame:
        """Calculate predictions using current model parameters.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix for predictions.

        Returns:
        -------
        np.ndarray
            Predicted values.
        """
        return (
            self.intercept
            + self.pfh_coef * features[:, 0]
            + self.pah_coef * features[:, 1]
            + self.pfa_coef * features[:, 2]
            + self.paa_coef * features[:, 3]
        )

    def _calculate_team_statistics(self) -> None:
        """Calculate and store all team-related statistics."""
        # Create a temporary DataFrame for calculations
        df = pd.DataFrame(
            {
                "home_team": self.home_team,
                "away_team": self.away_team,
                "home_goals": self.home_goals,
                "away_goals": self.away_goals,
            }
        )

        # Calculate mean points for home/away scenarios
        home_stats = df.groupby("home_team").agg(
            {"home_goals": "mean", "away_goals": "mean"}
        )
        away_stats = df.groupby("away_team").agg(
            {"away_goals": "mean", "home_goals": "mean"}
        )

        # Store transformed statistics
        self.pfh = df.groupby("home_team")["home_goals"].transform("mean").to_numpy()
        self.pah = df.groupby("home_team")["away_goals"].transform("mean").to_numpy()
        self.pfa = df.groupby("away_team")["away_goals"].transform("mean").to_numpy()
        self.paa = df.groupby("away_team")["home_goals"].transform("mean").to_numpy()

        # Create team ratings dictionary
        self.team_ratings = {}
        for team in self.teams:
            if team in home_stats.index and team in away_stats.index:
                self.team_ratings[team] = np.array(
                    [
                        home_stats.loc[team, "home_goals"],
                        home_stats.loc[team, "away_goals"],
                        away_stats.loc[team, "away_goals"],
                        away_stats.loc[team, "home_goals"],
                    ]
                )
            elif team in home_stats.index:
                # Team only played home games
                self.team_ratings[team] = np.array(
                    [
                        home_stats.loc[team, "home_goals"],
                        home_stats.loc[team, "away_goals"],
                        0.0,
                        0.0,
                    ]
                )
            elif team in away_stats.index:
                # Team only played away games
                self.team_ratings[team] = np.array(
                    [
                        0.0,
                        0.0,
                        away_stats.loc[team, "away_goals"],
                        away_stats.loc[team, "home_goals"],
                    ]
                )
            else:
                # Team hasn't played any games (shouldn't happen with proper data)
                self.team_ratings[team] = np.zeros(4)
