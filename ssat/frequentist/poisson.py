# %%
"""This file contains the implementation of the Poisson model for predicting games outcomes."""

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ssat.frequentist.base_model import BaseModel


class Poisson(BaseModel):
    """Poisson model for predicting games outcomes.

    A probabilistic model that estimates team attack/defense strengths and predicts
    match outcomes using maximum likelihood estimation with a Poisson distribution.

    Parameters
    ----------
    home_advantage : float, default=0.25
        Initial value for home advantage parameter

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
        [0:n_teams_] - Team attack ratings
        [n_teams_:2*n_teams_] - Team defense ratings
        [-1] - Home advantage parameter
    """

    NAME = "Poisson"

    def __init__(self, home_advantage: float = 0.25) -> None:
        """Initialize Poisson model."""
        super().__init__()
        self.home_advantage_ = home_advantage
        self.is_fitted_ = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        Z: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None,
    ) -> "Poisson":
        """Fit a Poisson model.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing match data with two columns:
            - First column: Home team names (string)
            - Second column: Away team names (string)
        y : Optional[Union[np.ndarray, pd.Series]], default=None
            Not used in this model but included for API consistency
        Z : pd.DataFrame
            Additional data for the model with home_goals and away_goals
            Must be provided with exactly 2 columns
        weights : Optional[np.ndarray], default=None
            Weights for rating optimization

        Returns:
        -------
        self : Poisson
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

            # Initialize parameters
            self.params_ = np.zeros(2 * self.n_teams_ + 1)
            # Set attack ratings to average 1.0 to satisfy sum = n_teams_ constraint
            self.params_[: self.n_teams_] = 1.0
            # Set defense ratings to 0.0 to satisfy sum = 0 constraint
            self.params_[self.n_teams_ : -1] = 0.0
            # Set home advantage
            self.params_[-1] = self.home_advantage_

            self.params_ = self._optimize_parameters()

            self._calculate_spread_error(X)

            self.is_fitted_ = True
            return self

        except Exception as e:
            self.is_fitted_ = False
            raise ValueError(f"Model fitting failed: {str(e)}")

    def _optimize_parameters(self) -> np.ndarray:
        """Optimize model parameters using SLSQP."""
        bounds = [(-100, 100)] * (2 * self.n_teams_) + [(0, 4)] * 1

        result = minimize(
            fun=self._log_likelihood,
            x0=self.params_,
            method="SLSQP",
            bounds=bounds,
            options={"ftol": 1e-10, "maxiter": 200},
        )

        return result.x

    def _get_team_ratings(
        self,
        home_idx: Union[int, np.ndarray, None] = None,
        away_idx: Union[int, np.ndarray, None] = None,
    ) -> dict:
        """Get team ratings for given indices.

        Parameters
        ----------
        home_idx : Union[int, np.ndarray, None], default=None
            Index(es) of home team(s)
        away_idx : Union[int, np.ndarray, None], default=None
            Index(es) of away team(s)

        Returns:
        -------
        dict
            Dictionary containing team ratings
        """
        if home_idx is None:
            home_idx, away_idx = self.home_idx_, self.away_idx_

        attack_ratings = self.params_[: self.n_teams_]
        defense_ratings = self.params_[self.n_teams_ : 2 * self.n_teams_]
        home_advantage = self.params_[-1]

        return {
            "home_attack": attack_ratings[home_idx],
            "away_attack": attack_ratings[away_idx],
            "home_defense": defense_ratings[home_idx],
            "away_defense": defense_ratings[away_idx],
            "home_advantage": home_advantage,
        }

    def _calculate_expected_goals(self, ratings: dict) -> tuple:
        """Calculate expected goals using team ratings.

        Parameters
        ----------
        ratings : dict
            Dictionary of team ratings from _get_team_ratings

        Returns:
        -------
        tuple
            (home_goals, away_goals)
        """
        home_goals = np.exp(
            ratings["home_advantage"] + ratings["home_attack"] + ratings["away_defense"]
        )
        away_goals = np.exp(ratings["away_attack"] + ratings["home_defense"])
        return home_goals, away_goals

    def _calculate_spread_error(self, X: pd.DataFrame) -> None:
        """Calculate spread error."""
        home_teams = X.iloc[:, 0].to_numpy()
        away_teams = X.iloc[:, 1].to_numpy()
        predictions = np.zeros(len(X))

        for i, (home_team, away_team) in enumerate(zip(home_teams, away_teams)):
            # Validate teams
            self._validate_teams([home_team, away_team])

            ratings = self._get_team_ratings(
                home_idx=self.team_map_[home_team], away_idx=self.team_map_[away_team]
            )
            home_goals, away_goals = self._calculate_expected_goals(ratings)
            predictions[i] = home_goals - away_goals

        # Calculate spread error
        residuals = self.spread_ - predictions
        sse = np.sum((residuals**2))
        self.spread_error_ = np.sqrt(sse / (len(X) - X.shape[1]))

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
        """
        self._check_is_fitted()
        self._validate_X(X, fit=False)

        home_teams = X.iloc[:, 0].to_numpy()
        away_teams = X.iloc[:, 1].to_numpy()
        predicted_spreads = np.zeros(len(X))

        for i, (home_team, away_team) in enumerate(zip(home_teams, away_teams)):
            # Validate teams
            self._validate_teams([home_team, away_team])

            ratings = self._get_team_ratings(
                home_idx=self.team_map_[home_team], away_idx=self.team_map_[away_team]
            )
            home_goals, away_goals = self._calculate_expected_goals(ratings)
            predicted_spreads[i] = home_goals - away_goals

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

            ratings = self._get_team_ratings(
                home_idx=self.team_map_[home_team], away_idx=self.team_map_[away_team]
            )
            home_goals, away_goals = self._calculate_expected_goals(ratings)
            predicted_spread = home_goals - away_goals
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
        self._check_is_fitted()
        return {
            "teams": self.teams_,
            "team_map": self.team_map_,
            "params": self.params_,
            "is_fitted": self.is_fitted_,
            "home_advantage": self.home_advantage_,
        }

    def set_params(self, params: dict) -> None:
        """Set parameters for the model.

        Parameters
        ----------
        params : dict
            Dictionary containing model parameters, as returned by get_params()
        """
        self.teams_ = params["teams"]
        self.team_map_ = params["team_map"]
        self.params_ = params["params"]
        self.is_fitted_ = params["is_fitted"]
        self.home_advantage_ = params["home_advantage"]

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

    def _log_likelihood(self, params: np.ndarray) -> np.float64:
        """Calculate negative log likelihood for parameter optimization.

        Parameters
        ----------
        params : np.ndarray
            Model parameters to optimize
            [0:n_teams_] - Attack ratings
            [n_teams_:2*n_teams_] - Defense ratings
            [-1] - Home advantage parameter

        Returns:
        -------
        float
            Negative log likelihood
        """
        # Extract parameters
        attack_ratings = params[: self.n_teams_]
        defense_ratings = params[self.n_teams_ : 2 * self.n_teams_]
        home_advantage = params[-1]
        log_likelihood: np.float64 = np.float64(0.0)

        # Get team ratings for home and away teams
        home_attack = attack_ratings[self.home_idx_]
        away_attack = attack_ratings[self.away_idx_]
        home_defense = defense_ratings[self.home_idx_]
        away_defense = defense_ratings[self.away_idx_]

        # Calculate expected goals
        home_exp = np.exp(home_advantage + home_attack + away_defense)
        away_exp = np.exp(away_attack + home_defense)

        # Calculate log probabilities
        home_llk = stats.poisson.logpmf(self.home_goals_, home_exp)
        away_llk = stats.poisson.logpmf(self.away_goals_, away_exp)

        # Sum log likelihood with weights
        log_likelihood = np.sum(self.weights_ * (home_llk + away_llk))

        return -log_likelihood
