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
        self.is_fitted_ = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        Z: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None,
    ) -> "GSSD":
        """Fit the GSSD model.

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
        self : GSSD
            Fitted model
        """
        try:
            # Validate input dimensions and types
            self._validate_X(X)

            # Validate Z is provided and has required dimensions
            if Z is None:
                raise ValueError(
                    "Z must be provided with home_goals and away_goals data"
                )
            if len(Z) != len(X):
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

            # Extract home_goals and away_goals from Z
            self.home_goals_ = Z.iloc[:, 0].to_numpy()
            self.away_goals_ = Z.iloc[:, 1].to_numpy()

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

            # Calculate team statistics
            self._calculate_team_statistics(X)

            # Prepare features and fit model
            features = np.column_stack((self.pfh_, self.pah_, self.pfa_, self.paa_))
            initial_guess = np.array([0.1, 1.0, 1.0, -1.0, -1.0])
            result = minimize(
                self._sse_function,
                initial_guess,
                method="L-BFGS-B",
                options={"ftol": 1e-10, "maxiter": 200},
            )

            # Store model parameters
            self.const_ = result.x[0]
            self.pfh_coef_ = result.x[1]
            self.pah_coef_ = result.x[2]
            self.pfa_coef_ = result.x[3]
            self.paa_coef_ = result.x[4]

            # Calculate spread error
            predictions = self._get_predictions(features)
            residuals = self.spread_ - predictions
            sse = np.sum((residuals**2))
            self.spread_error_ = np.sqrt(sse / (X.shape[0] - X.shape[1]))

            self.is_fitted_ = True
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

        home_teams = X.iloc[:, 0].to_numpy()
        away_teams = X.iloc[:, 1].to_numpy()

        predicted_spreads = np.zeros(len(X))

        for i, (home_team, away_team) in enumerate(zip(home_teams, away_teams)):
            # Validate teams
            self._validate_teams([home_team, away_team])

            # Get team ratings
            home_off, home_def = self.team_ratings_[home_team][:2]
            away_off, away_def = self.team_ratings_[away_team][2:]

            # Calculate spread
            predicted_spreads[i] = (
                self.const_
                + home_off * self.pfh_coef_
                + home_def * self.pah_coef_
                + away_off * self.pfa_coef_
                + away_def * self.paa_coef_
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
            home_off, home_def = self.team_ratings_[home_team][:2]
            away_off, away_def = self.team_ratings_[away_team][2:]

            # Calculate spread
            predicted_spread = (
                self.const_
                + home_off * self.pfh_coef_
                + home_def * self.pah_coef_
                + away_off * self.pfa_coef_
                + away_def * self.paa_coef_
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
            "intercept": self.intercept_,
            "pfh_coef": self.pfh_coef_,
            "pah_coef": self.pah_coef_,
            "pfa_coef": self.pfa_coef_,
            "paa_coef": self.paa_coef_,
            "params": self.team_ratings_,
            "is_fitted": self.is_fitted_,
        }

    def set_params(self, params: dict) -> None:
        """Set parameters for the model.

        Parameters
        ----------
        params : dict
            Dictionary containing model parameters, as returned by get_params()
        """
        self.intercept_ = params["intercept"]
        self.pfh_coef_ = params["pfh_coef"]
        self.pah_coef_ = params["pah_coef"]
        self.pfa_coef_ = params["pfa_coef"]
        self.paa_coef_ = params["paa_coef"]
        self.team_ratings_ = params["params"]
        self.is_fitted_ = params["is_fitted"]

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
            + pfh_coef * self.pfh_
            + pah_coef * self.pah_
            + pfa_coef * self.pfa_
            + paa_coef * self.paa_
        )

        # Calculate weighted squared errors
        errors = self.spread_ - predictions
        sse = np.sum(self.weights_ * (errors**2))

        return sse

    def _get_predictions(self, features: np.ndarray) -> np.ndarray:
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
            self.const_
            + self.pfh_coef_ * features[:, 0]
            + self.pah_coef_ * features[:, 1]
            + self.pfa_coef_ * features[:, 2]
            + self.paa_coef_ * features[:, 3]
        )

    def _calculate_team_statistics(self, X: pd.DataFrame) -> None:
        """Calculate and store all team-related statistics.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing match data
        """
        # Create a temporary DataFrame for calculations
        df = pd.DataFrame(
            {
                "home_team": self.home_team_,
                "away_team": self.away_team_,
                "home_goals": self.home_goals_,
                "away_goals": self.away_goals_,
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
        self.pfh_ = df.groupby("home_team")["home_goals"].transform("mean").to_numpy()
        self.pah_ = df.groupby("home_team")["away_goals"].transform("mean").to_numpy()
        self.pfa_ = df.groupby("away_team")["away_goals"].transform("mean").to_numpy()
        self.paa_ = df.groupby("away_team")["home_goals"].transform("mean").to_numpy()

        # Create team ratings dictionary
        self.team_ratings_ = {}
        for team in self.teams_:
            if team in home_stats.index and team in away_stats.index:
                self.team_ratings_[team] = np.array(
                    [
                        home_stats.loc[team, "home_goals"],
                        home_stats.loc[team, "away_goals"],
                        away_stats.loc[team, "away_goals"],
                        away_stats.loc[team, "home_goals"],
                    ]
                )
            elif team in home_stats.index:
                # Team only played home games
                self.team_ratings_[team] = np.array(
                    [
                        home_stats.loc[team, "home_goals"],
                        home_stats.loc[team, "away_goals"],
                        0.0,  # Default value for missing away offensive rating
                        0.0,  # Default value for missing away defensive rating
                    ]
                )
            elif team in away_stats.index:
                # Team only played away games
                self.team_ratings_[team] = np.array(
                    [
                        0.0,  # Default value for missing home offensive rating
                        0.0,  # Default value for missing home defensive rating
                        away_stats.loc[team, "away_goals"],
                        away_stats.loc[team, "home_goals"],
                    ]
                )
            else:
                # Team hasn't played any games (shouldn't happen with proper data)
                self.team_ratings_[team] = np.zeros(4)

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
            self.team_ratings_, index=["pfh", "pah", "pfa", "paa"]
        ).T

        # Add coefficients as a new row
        coeffs = {
            "pfh": self.pfh_coef_,
            "pah": self.pah_coef_,
            "pfa": self.pfa_coef_,
            "paa": self.paa_coef_,
        }
        ratings_df.loc["Coefficients"] = pd.Series(coeffs)
        ratings_df.loc["Intercept"] = self.const_

        return ratings_df.round(2)
