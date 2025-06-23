"""Bayesian Poisson models for sports match prediction.

This module implements Poisson-based models for predicting match outcomes in sports.
Poisson models are particularly suitable for modeling goal-scoring events, treating
home and away goals as independent Poisson processes with team-specific attack and
defense parameters.

Classes
-------
Poisson : PredictiveModel
    Standard Poisson model for match prediction
PoissonDecay : Poisson
    Poisson model with temporal decay weighting for recent matches

The Poisson models predict both individual goal counts (home_goals, away_goals) and
goal differences, making them suitable for detailed match simulation and various
betting market predictions. They use hierarchical Bayesian estimation to learn
team-specific attack and defense strengths from historical match data.

Features:
- Individual goal prediction for home and away teams
- Goal difference prediction for match outcomes
- Probability estimation for win/draw/loss scenarios
- Support for temporal decay weighting (PoissonDecay)
- Monte Carlo simulation for match outcomes
"""

from typing import Optional, Union

import numpy as np
import pandas as pd

from ssat.bayesian.base_predictive_model import PredictiveModel


class Poisson(PredictiveModel):
    """Bayesian Poisson model for predicting match scores.

    This model treats home and away goals as independent Poisson processes,
    with team-specific attack and defense parameters. It's particularly suitable
    for sports where scoring events are relatively rare and independent.

    Model Structure
    ---------------
    - Home goals ~ Poisson(λ_home)
    - Away goals ~ Poisson(λ_away)
    - λ_home = exp(home_advantage + attack_home - defense_away)
    - λ_away = exp(attack_away - defense_home)

    Parameters
    ----------
    stem : str, default="poisson"
        Name of the Stan model file (without .stan extension)

    Attributes:
    ----------
    predictions : Optional[np.ndarray]
        Stored predictions from the last predict() call
        Shape: [3, n_simulations, n_matches] for [goal_diff, home_goals, away_goals]

    Examples:
    --------
    >>> model = Poisson()
    >>> model.fit(X_train, y_train)  # X: team pairs, y: [home_goals, away_goals]
    >>> predictions = model.predict(X_test)  # Returns goal differences
    >>> probabilities = model.predict_proba(X_test)  # Returns [home, draw, away] probs
    >>> simulations = model.simulate_matches(X_test)  # Returns [home_goals, away_goals]
    """

    def __init__(
        self,
        stem: str = "poisson",
    ):
        """Initialize the Poisson model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "poisson".
        """
        super().__init__(stan_file=stem)

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        Z: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        point_spread: int = 0,
    ) -> pd.DataFrame:
        """Generate predictions for new data.

        For Poisson model, this returns predicted goal differences (home - away).

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

        # Convert X to DataFrame format for _data_dict
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=["home_team", "away_team"])
        else:
            X_df = X.copy()

        # For prediction, create dummy y data (will be ignored by Stan model)
        dummy_y = np.zeros((len(X_df), 2))  # Dummy home_goals, away_goals

        # Use _data_dict_replacement for standardized data preparation
        data_dict = self._data_dict(X_df, dummy_y, Z, None, fit=False)

        if self.model is None or self.fit_result is None:
            raise ValueError("Model not properly initialized")

        # Generate predictions using Stan model
        preds = self.model.generate_quantities(
            data=data_dict, previous_fit=self.fit_result
        )

        # Get all three prediction variables for Poisson model
        pred_home_goals = preds.stan_variable("pred_home_goals_match")
        pred_away_goals = preds.stan_variable("pred_away_goals_match")
        pred_goal_diff = preds.stan_variable("pred_goal_diff_match")

        # Store all predictions for predict_proba (shape: [3, n_sims, n_matches])
        self.predictions = np.array([pred_goal_diff, pred_home_goals, pred_away_goals])

        # Return median goal differences with point_spread adjustment
        result = np.median(pred_goal_diff, axis=0) + point_spread

        return self._format_predictions(
            X,
            result,
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

        # Always generate fresh predictions to ensure Z parameter changes are respected
        _ = self.predict(X, Z, point_spread=0)  # Don't apply point_spread twice
        predictions = self.predictions

        # For Poisson model: predictions[0] = goal_diff, predictions[1] = home_goals, predictions[2] = away_goals
        if predictions.shape[0] == 3:
            # Use the goal differences directly from Stan model
            goal_differences = predictions[0] + point_spread
        else:
            raise ValueError(
                f"Invalid predictions shape for Poisson model: {predictions.shape}"
            )

        # Calculate probabilities using Monte Carlo integration over posterior samples
        home_probs = (goal_differences > 0).mean(axis=0)
        draw_probs = (goal_differences == 0).mean(axis=0)
        away_probs = (goal_differences < 0).mean(axis=0)

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

    def simulate_matches(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        Z: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Generate separate home and away goal simulations.

        This method is specific to Poisson model which predicts individual goals.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Team pairs data with exactly 2 columns: [home_team, away_team]
        Z : Optional[Union[pd.DataFrame, np.ndarray]], default=None
            Additional match-level features

        Returns:
        -------
        np.ndarray
            Simulated goals with shape (n_matches, 2) for [home_goals, away_goals]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before making predictions")

        # Always generate fresh predictions to ensure Z parameter changes are respected
        _ = self.predict(X, Z)
        predictions = self.predictions

        # For Poisson model: predictions[0] = goal_diff, predictions[1] = home_goals, predictions[2] = away_goals
        if predictions.shape[0] == 3:
            home_goals = np.median(predictions[1], axis=0)  # Median home goals
            away_goals = np.median(predictions[2], axis=0)  # Median away goals
            result = np.column_stack([home_goals, away_goals])
        else:
            raise ValueError(
                f"Invalid predictions shape for Poisson model: {predictions.shape}"
            )

        return self._format_predictions(
            X,
            result,
            col_names=["home_goals", "away_goals"],
        )


class PoissonDecay(Poisson):
    """Bayesian Poisson model with temporal decay weighting for recent matches.

    Extends the basic Poisson model by incorporating temporal decay weights,
    giving more importance to recent matches when estimating team parameters.
    This is particularly useful for sports where team strength changes significantly
    over time due to transfers, form, or other factors.

    Model Structure
    ---------------
    Same as Poisson model but with temporal weights applied to the likelihood:
    - Each match has weight = exp(-decay_rate * days_since_match)
    - Recent matches have higher influence on parameter estimation

    Parameters
    ----------
    stem : str, default="poisson_decay"
        Name of the Stan model file (without .stan extension)

    Additional Requirements
    ----------------------
    Z : array-like
        Temporal decay weights or days since each match for training data
        Required for fitting, optional for prediction (defaults to 0)

    Examples:
    --------
    >>> model = PoissonDecay()
    >>> days_since = (max_date - match_dates).dt.days
    >>> model.fit(X_train, y_train, Z=days_since)  # Z provides temporal weights
    >>> predictions = model.predict(X_test)  # Z defaults to 0 for new matches
    """

    def __init__(
        self,
        stem: str = "poisson_decay",
    ):
        """Initialize the Poisson Weighted model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "poisson_decay".
        """
        super().__init__(stem=stem)
