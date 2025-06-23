# %%
"""Z-Score Deviation (ZSD) model."""

import warnings
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ssat.frequentist.base_model import BaseModel

# Suppress the specific warning
warnings.filterwarnings(
    "ignore", message="delta_grad == 0.0. Check if the approximated function is linear."
)


class ZSD(BaseModel):
    """Z-Score Deviation (ZSD) model for predicting sports match outcomes with scikit-learn-like API.

    The model uses weighted optimization to estimate team performance parameters and
    calculates win/draw/loss probabilities using a normal distribution.

    Parameters
    ----------
    None

    Attributes:
    ----------
    teams_ : np.ndarray
        Unique team identifiers
    n_teams_ : int
        Number of teams in the dataset
    team_map_ : Dict[str, int]
        Mapping of team names to indices
    home_idx_ : np.ndarray
        Indices of home teams
    away_idx_ : np.ndarray
        Indices of away teams
    weights_ : np.ndarray
        Weights for rating optimization
    is_fitted_ : bool
        Whether the model has been fitted
    params_ : np.ndarray
        Optimized model parameters after fitting
        [0:n_teams_] - Offensive ratings
        [n_teams_:2*n_teams_] - Defensive ratings
        [-2:] - Home/away adjustment factors
    mean_home_score_ : float
        Mean home team score
    std_home_score_ : float
        Standard deviation of home team scores
    mean_away_score_ : float
        Mean away score
    std_away_score_ : float
        Standard deviation of away team scores
    intercept_ : float
        Spread model intercept
    spread_coefficient_ : float
        Spread model coefficient
    spread_error_ : float
        Standard error of spread predictions

    Note:
    ----
    The model ensures that both offensive and defensive ratings sum to zero
    through optimization constraints, making the ratings interpretable as
    relative performance measures.
    """

    NAME = "ZSD"

    def __init__(self) -> None:
        """Initialize ZSD model."""
        self.is_fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[np.ndarray, pd.Series],
        Z: Union[pd.DataFrame, np.ndarray],
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "ZSD":
        """Fit the ZSD model.

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
        self : ZSD
            Fitted model
        """
        # Validate input dimensions and types
        self._validate_X(X)
        self._validate_Z(X, Z, True)

        # Extract team data using helper method
        self.home_team, self.away_team = self._extract_teams(X)

        # Handle goal difference (y)
        self.goal_diff = np.asarray(y)

        # Validate goal difference
        if not np.issubdtype(self.goal_diff.dtype, np.number):
            raise ValueError("Goal differences must be numeric")

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
        n_matches = len(X)
        self.weights = np.ones(n_matches) if weights is None else weights

        # Calculate scoring statistics
        self._calculate_scoring_statistics()

        # Initialize and optimize parameters
        self._init_parameters()
        self.params = self._optimize_parameters(**kwargs)

        # Fit spread model
        pred_scores = self._predict_scores()
        predictions = pred_scores["home"] - pred_scores["away"]
        residuals = self.goal_diff - predictions
        sse = np.sum((residuals**2))
        self.spread_error = np.sqrt(sse / (X.shape[0] - X.shape[1]))

        self.is_fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        Z: Optional[pd.DataFrame] = None,
        point_spread: int = 0,
    ) -> pd.DataFrame:
        """Predict point spreads for matches.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing match data with two columns:
            - First column: Home team names (string)
            - Second column: Away team names (string)
        Z : pd.DataFrame
            Additional data for prediction. Not used in this method but included for API consistency.
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

        # Vectorized prediction calculation
        pred_scores = self._predict_scores(home_idx, away_idx)
        predicted_spreads = pred_scores["home"] - pred_scores["away"] + point_spread

        return self._format_predictions(
            X,
            predicted_spreads,
            col_names=["goal_diff"],
        )

    def predict_proba(
        self,
        X: pd.DataFrame,
        Z: Optional[pd.DataFrame] = None,
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
            probs = stats.norm.cdf(thresholds, predictions, self.spread_error)
            home_probs = 1 - probs[:, 0]
            draw_probs = probs[:, 0] - probs[:, 1]
            away_probs = probs[:, 1]
        else:
            home_probs = 1 - stats.norm.cdf(
                point_spread, predictions, self.spread_error
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

    def _init_parameters(self) -> None:
        """Initialize model parameters with smart defaults."""
        # Initialize parameters based on team performance
        self.initial_params = np.random.normal(0, 0.1, 2 * self.n_teams + 2)

        # Set constraints for optimization
        self.constraints = [
            {"type": "eq", "fun": lambda p: np.mean(p[: self.n_teams])},
            {
                "type": "eq",
                "fun": lambda p: np.mean(p[self.n_teams : 2 * self.n_teams]),
            },
        ]

        # Set bounds for optimization
        self.bounds = [(-50, 50)] * (2 * self.n_teams) + [(-np.inf, np.inf)] * 2

    def _optimize_parameters(self, **kwargs) -> pd.DataFrame:
        """Optimize model parameters with fallback methods.

        Parameters
        ----------
        **kwargs : dict
            Additional optimization parameters (ftol, maxiter, etc.)

        Returns:
        -------
        np.ndarray
            Optimized parameters

        Raises:
        ------
        RuntimeError
            If optimization fails with all methods
        """
        # Default optimization options
        default_options = {"maxiter": 100000, "ftol": 1e-8}
        options = {**default_options, **kwargs}

        # Try different optimization methods
        methods = ["SLSQP", "trust-constr"]

        for method in methods:
            try:
                result = minimize(
                    fun=self._sse_function,
                    x0=self.initial_params,
                    method=method,
                    constraints=self.constraints,
                    bounds=self.bounds,
                    options=options,
                )

                if result.success:
                    return result.x

            except Exception:
                # Continue to next method if current one fails
                continue

        # If no method succeeded, raise an error
        raise RuntimeError("Optimization failed with all attempted methods")

    def _sse_function(self, params: np.ndarray) -> np.float64:
        """Calculate the weighted sum of squared errors for given parameters.

        Parameters
        ----------
        params : np.ndarray
            Model parameters

        Returns:
        -------
        float
            Weighted sum of squared errors
        """
        # Unpack parameters efficiently
        pred_scores = self._predict_scores(
            self.home_idx,
            self.away_idx,
            *np.split(params, [self.n_teams, 2 * self.n_teams]),
        )
        squared_errors = (self.home_goals - pred_scores["home"]) ** 2 + (
            self.away_goals - pred_scores["away"]
        ) ** 2
        return np.sum(squared_errors * self.weights, axis=0)

    def _predict_scores(
        self,
        home_idx: Union[int, np.ndarray, None] = None,
        away_idx: Union[int, np.ndarray, None] = None,
        home_ratings: Union[np.ndarray, None] = None,
        away_ratings: Union[np.ndarray, None] = None,
        factors: Union[np.ndarray, None] = None,
    ) -> Dict[str, np.ndarray]:
        """Calculate predicted scores using team ratings and factors.

        Parameters
        ----------
        home_idx : Union[int, np.ndarray, None], default=None
            Index(es) of home team(s)
        away_idx : Union[int, np.ndarray, None], default=None
            Index(es) of away team(s)
        home_ratings : Union[np.ndarray, None], default=None
            Optional home ratings to use
        away_ratings : Union[np.ndarray, None] = None
            Optional away ratings to use
        factors : Union[Tuple[float, float], None], default=None
            Optional (home_factor, away_factor) tuple

        Returns:
        -------
        Dict[str, np.ndarray]
            Dict with 'home' and 'away' predicted scores
        """
        if factors is None:
            factors = self.params[-2:]

        ratings = self._get_team_ratings(home_idx, away_idx, home_ratings, away_ratings)

        return {
            "home": self._transform_to_score(
                self._parameter_estimate(
                    factors[0], ratings["home_rating"], ratings["away_rating"]
                ),
                self.mean_home_score,
                self.std_home_score,
            ),
            "away": self._transform_to_score(
                self._parameter_estimate(
                    factors[1], ratings["home_away_rating"], ratings["away_home_rating"]
                ),
                self.mean_away_score,
                self.std_away_score,
            ),
        }

    def _get_team_ratings(
        self,
        home_idx: Union[int, np.ndarray, None],
        away_idx: Union[int, np.ndarray, None],
        home_ratings: Union[np.ndarray, None] = None,
        away_ratings: Union[np.ndarray, None] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract team ratings from parameters.

        Parameters
        ----------
        home_idx : Union[int, np.ndarray, None]
            Index(es) of home team(s)
        away_idx : Union[int, np.ndarray, None]
            Index(es) of away team(s)
        home_ratings : Union[np.ndarray, None], default=None
            Optional home ratings to use
        away_ratings : Union[np.ndarray, None] = None
            Optional away ratings to use

        Returns:
        -------
        Dict[str, np.ndarray]
            Dictionary with team ratings
        """
        if home_ratings is None and away_ratings is None:
            home_ratings, away_ratings = np.split(self.params[: 2 * self.n_teams], 2)
        if home_idx is None:
            home_idx, away_idx = self.home_idx, self.away_idx

        assert home_ratings is not None and away_ratings is not None, (
            "home_ratings and away_ratings must be provided"
        )

        return {
            "home_rating": home_ratings[home_idx],
            "home_away_rating": away_ratings[home_idx],
            "away_rating": home_ratings[away_idx],
            "away_home_rating": away_ratings[away_idx],
        }

    def _parameter_estimate(
        self, adj_factor: np.float64, home_rating: np.ndarray, away_rating: np.ndarray
    ) -> pd.DataFrame:
        """Calculate parameter estimate for score prediction.

        Parameters
        ----------
        adj_factor : float
            Adjustment factor
        home_rating : np.ndarray
            Home team rating
        away_rating : np.ndarray
            Away team rating

        Returns:
        -------
        np.ndarray
            Parameter estimate
        """
        return adj_factor + home_rating - away_rating

    def _transform_to_score(
        self, param: np.ndarray, mean: np.float64, std: np.float64
    ) -> pd.DataFrame:
        """Transform parameter to actual score prediction.

        Parameters
        ----------
        param : np.ndarray
            Parameter value
        mean : float
            Mean score
        std : float
            Standard deviation of scores

        Returns:
        -------
        np.ndarray
            Predicted score
        """
        exp_prob = self._logit_transform(param)
        z_score = stats.norm.ppf(exp_prob)
        return np.asarray(mean + std * z_score)

    def _calculate_scoring_statistics(self) -> None:
        """Calculate and store scoring statistics for home and away teams."""
        # Calculate all statistics in one pass using numpy
        home_stats: np.ndarray = np.array(
            [np.mean(self.home_goals), np.std(self.home_goals, ddof=1)]
        )
        away_stats: np.ndarray = np.array(
            [np.mean(self.away_goals), np.std(self.away_goals, ddof=1)]
        )

        # Unpack results
        self.mean_home_score: np.float64 = home_stats[0]
        self.std_home_score: np.float64 = home_stats[1]
        self.mean_away_score: np.float64 = away_stats[0]
        self.std_away_score: np.float64 = away_stats[1]

        # Validate statistics
        if not (self.std_home_score > 0 and self.std_away_score > 0):
            raise ValueError(
                "Invalid scoring statistics: zero or negative standard deviation"
            )

    def get_params(self) -> dict:
        """Get the current parameters of the model.

        Returns:
        -------
        dict
            Dictionary containing model parameters
        """
        return {
            "teams": self.teams,
            "team_map": self.team_map,
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
        self.teams = params["teams"]
        self.team_map = params["team_map"]
        self.params = params["params"]
        self.is_fitted = params["is_fitted"]

    def get_team_ratings(self) -> pd.DataFrame:
        """Get team ratings as a DataFrame.

        Returns:
        -------
        pd.DataFrame
            Team ratings with columns ['team', 'home', 'away']
        """
        self._check_is_fitted()

        home_ratings = self.params[: self.n_teams]
        away_ratings = self.params[self.n_teams : 2 * self.n_teams]

        return pd.DataFrame(
            {"team": self.teams, "home": home_ratings, "away": away_ratings}
        ).set_index("team")
