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
        self.is_fitted_ = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        Z: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None,
    ) -> "ZSD":
        """Fit the ZSD model.

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
        self : ZSD
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

            self.home_goals_ = Z.iloc[:, 0].to_numpy()
            self.away_goals_ = Z.iloc[:, 1].to_numpy()

            if len(self.home_goals_) == 0 or len(self.away_goals_) == 0:
                raise ValueError("Empty input data")

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

            self._calculate_scoring_statistics()

            # Optimize
            self.params_ = self._optimize_parameters()

            # Fit spread model
            pred_scores = self._predict_scores()
            predictions = pred_scores["home"] - pred_scores["away"]
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
        Z : pd.DataFrame
            Additional data for prediction. Not used in this method but included for API consistency.
        point_spread : float, default=0.0
            Point spread adjustment
        """
        self._check_is_fitted()
        self._validate_X(X, fit=False)

        home_teams = X.iloc[:, 0].to_numpy()
        away_teams = X.iloc[:, 1].to_numpy()

        predicted_spreads = np.zeros(len(X))

        for i, (home_team, away_team) in enumerate(zip(home_teams, away_teams)):
            # Validate teams
            self._validate_teams([home_team, away_team])

            # Get predicted scores using team indices
            pred_scores = self._predict_scores(
                home_idx=self.team_map_[home_team], away_idx=self.team_map_[away_team]
            )

            # Calculate spread
            predicted_spreads[i] = pred_scores["home"] - pred_scores["away"]

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

            # Get predicted scores using team indices
            pred_scores = self._predict_scores(
                home_idx=self.team_map_[home_team], away_idx=self.team_map_[away_team]
            )

            # Calculate spread
            predicted_spread = pred_scores["home"] - pred_scores["away"]

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

    def _optimize_parameters(self) -> np.ndarray:
        """Optimize model parameters using SLSQP optimization.

        Returns:
        -------
        np.ndarray
            Optimized parameters

        Raises:
        ------
        RuntimeError
            If optimization fails
        """
        constraints = [
            {"type": "eq", "fun": lambda p: np.mean(p[: self.n_teams_])},
            {
                "type": "eq",
                "fun": lambda p: np.mean(p[self.n_teams_ : 2 * self.n_teams_]),
            },
        ]

        bounds = [(-50, 50)] * (2 * self.n_teams_) + [(-np.inf, np.inf)] * 2
        x0 = self._get_initial_params()

        result = minimize(
            fun=self._sse_function,
            x0=x0,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": 100000, "ftol": 1e-8},
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        return result.x

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
            self.home_idx_,
            self.away_idx_,
            *np.split(params, [self.n_teams_, 2 * self.n_teams_]),
        )
        squared_errors = (self.home_goals_ - pred_scores["home"]) ** 2 + (
            self.away_goals_ - pred_scores["away"]
        ) ** 2
        return np.sum(squared_errors * self.weights_, axis=0)

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
            factors = self.params_[-2:]

        ratings = self._get_team_ratings(home_idx, away_idx, home_ratings, away_ratings)

        return {
            "home": self._transform_to_score(
                self._parameter_estimate(
                    factors[0], ratings["home_rating"], ratings["away_rating"]
                ),
                self.mean_home_score_,
                self.std_home_score_,
            ),
            "away": self._transform_to_score(
                self._parameter_estimate(
                    factors[1], ratings["home_away_rating"], ratings["away_home_rating"]
                ),
                self.mean_away_score_,
                self.std_away_score_,
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
            home_ratings, away_ratings = np.split(self.params_[: 2 * self.n_teams_], 2)
        if home_idx is None:
            home_idx, away_idx = self.home_idx_, self.away_idx_

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
    ) -> np.ndarray:
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
    ) -> np.ndarray:
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
        z_score = self._z_inverse(exp_prob)
        return np.asarray(mean + std * z_score)

    def _z_inverse(self, prob: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate inverse of standard normal CDF.

        Parameters
        ----------
        prob : Union[float, np.ndarray]
            Probability value(s)

        Returns:
        -------
        Union[float, np.ndarray]
            Z-score(s)
        """
        return stats.norm.ppf(prob)

    def _logit_transform(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Apply logistic transformation with numerical stability.

        Parameters
        ----------
        x : Union[float, np.ndarray]
            Input value(s)

        Returns:
        -------
        Union[float, np.ndarray]
            Transformed value(s)
        """
        # Clip values to avoid overflow
        x_clipped = np.clip(x, -700, 700)  # exp(700) is close to float max
        return 1 / (1 + np.exp(-x_clipped))

    def _calculate_scoring_statistics(self) -> None:
        """Calculate and store scoring statistics for home and away teams."""
        # Calculate all statistics in one pass using numpy
        home_stats: np.ndarray = np.array(
            [np.mean(self.home_goals_), np.std(self.home_goals_, ddof=1)]
        )
        away_stats: np.ndarray = np.array(
            [np.mean(self.away_goals_), np.std(self.away_goals_, ddof=1)]
        )

        # Unpack results
        self.mean_home_score_: np.float64 = home_stats[0]
        self.std_home_score_: np.float64 = home_stats[1]
        self.mean_away_score_: np.float64 = away_stats[0]
        self.std_away_score_: np.float64 = away_stats[1]

        # Validate statistics
        if not (self.std_home_score_ > 0 and self.std_away_score_ > 0):
            raise ValueError(
                "Invalid scoring statistics: zero or negative standard deviation"
            )

    def _get_initial_params(self) -> np.ndarray:
        """Generate initial parameters, incorporating any provided values.

        Returns:
        -------
        np.ndarray
            Complete parameter vector

        Raises:
        ------
        ValueError
            If parameters are invalid or don't match teams
        """
        return np.random.normal(0, 0.1, 2 * self.n_teams_ + 2)

    def get_team_ratings(self) -> pd.DataFrame:
        """Get team ratings as a DataFrame.

        Returns:
        -------
        pd.DataFrame
            Team ratings with columns ['team', 'home', 'away']
        """
        self._check_is_fitted()

        home_ratings = self.params_[: self.n_teams_]
        away_ratings = self.params_[self.n_teams_ : 2 * self.n_teams_]

        return pd.DataFrame(
            {"team": self.teams_, "home": home_ratings, "away": away_ratings}
        ).set_index("team")

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            "teams": self.teams_,
            "team_map": self.team_map_,
            "params": self.params_,
            "is_fitted": self.is_fitted_,
        }

    def set_params(self, params: dict) -> None:
        """Set model parameters."""
        self.teams_ = params["teams"]
        self.team_map_ = params["team_map"]
        self.params_ = params["params"]
        self.is_fitted_ = params["is_fitted"]
