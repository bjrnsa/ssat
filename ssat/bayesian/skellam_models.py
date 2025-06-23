"""Bayesian Skellam models for sports match prediction.

This module implements Skellam-based models for predicting match outcomes in sports.
Skellam models directly model the goal difference (home_goals - away_goals) using
the Skellam distribution, which is the distribution of the difference between two
Poisson random variables. This approach is particularly efficient for prediction
tasks focused on match outcomes rather than individual goal counts.

Classes
-------
Skellam : Poisson
    Standard Skellam model for goal difference prediction
SkellamDecay : Skellam
    Skellam model with temporal decay weighting
SkellamZero : Skellam
    Zero-inflated Skellam model for low-scoring sports
SkellamZeroDecay : SkellamZero
    Zero-inflated Skellam model with temporal decay weighting

Skellam models are computationally efficient for outcome prediction since they
directly model goal differences rather than individual goals. They inherit the
Poisson class prediction interface but override key methods to work with goal
differences. The zero-inflated variants are particularly useful for low-scoring
sports with frequent draws.

Model Features:
- Direct goal difference modeling using Skellam distribution
- Efficient outcome probability calculation
- Zero-inflation option for sports with frequent draws
- Temporal decay weighting support
- Custom quantile function implementation for accurate predictions

Note: Skellam models cannot simulate individual goal counts (simulate_matches
will raise NotImplementedError), only goal differences.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd

from ssat.bayesian.poisson_models import Poisson
from ssat.utils.skellam_optim import qskellam


class Skellam(Poisson):
    """Bayesian Skellam model for predicting match goal differences.

    Uses the Skellam distribution to directly model goal differences (home - away)
    rather than individual goal counts. The Skellam distribution is the difference
    of two Poisson distributions, making it mathematically elegant for modeling
    match outcomes while being computationally efficient.

    Model Structure
    ---------------
    - Goal difference ~ Skellam(λ_home, λ_away)
    - λ_home = exp(home_advantage + attack_home - defense_away)
    - λ_away = exp(attack_away - defense_home)
    - Skellam(k; λ₁, λ₂) represents difference of Poisson(λ₁) - Poisson(λ₂)

    Parameters
    ----------
    stem : str, default="skellam"
        Name of the Stan model file (without .stan extension)

    Attributes:
    ----------
    predictions : Optional[np.ndarray]
        Stored predictions from the last predict() call
        Shape: [1, n_simulations, n_matches] for goal differences only

    Limitations
    -----------
    - Cannot predict individual goal counts
    - simulate_matches() raises NotImplementedError
    - Only suitable for outcome-focused prediction tasks

    Examples:
    --------
    >>> model = Skellam()
    >>> model.fit(X_train, y_goal_diff)  # y is goal differences, not individual goals
    >>> predictions = model.predict(X_test)  # Returns goal differences
    >>> probabilities = model.predict_proba(X_test)  # Returns outcome probabilities
    """

    def __init__(
        self,
        stem: str = "skellam",
    ):
        """Initialize the Skellam model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "skellam".
        """
        super().__init__(stem=stem)

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        Z: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        point_spread: int = 0,
    ) -> pd.DataFrame:
        """Generate predictions for new data.

        For Skellam model, this returns predicted goal differences (home - away).
        Note: Skellam models only predict goal differences, not individual goals.

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
        dummy_y = np.zeros(len(X_df))  # Dummy goal differences

        # Use _data_dict_replacement for standardized data preparation
        data_dict = self._data_dict(X_df, dummy_y, Z, None, fit=False)

        if self.model is None or self.fit_result is None:
            raise ValueError("Model not properly initialized")

        # Generate predictions using Stan model
        preds = self.model.generate_quantities(
            data=data_dict, previous_fit=self.fit_result
        )
        stan_predictions = np.array(
            [preds.stan_variable(pred_var) for pred_var in self.pred_vars]
        )

        _, n_sims, n_matches = stan_predictions.shape

        # Use qskellam sampling by default for Skellam model
        # Set seed for reproducible predictions
        rng = np.random.Generator(np.random.PCG64(42))
        predictions = qskellam(
            rng.uniform(0, 1, size=(n_sims, n_matches)),
            stan_predictions[1],
            stan_predictions[2],
        )

        # Store predictions for predict_proba (shape: [1, n_sims, n_matches])
        self.predictions = predictions.reshape(1, n_sims, n_matches)

        # Return median goal differences with point_spread adjustment
        result = np.median(predictions, axis=0) + point_spread

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

        # For Skellam model: predictions[0] = goal_diff only (no individual goals)
        if predictions.shape[0] == 1:
            # Use the goal differences directly from Stan model
            goal_differences = predictions[0] + point_spread
        else:
            raise ValueError(
                f"Invalid predictions shape for Skellam model: {predictions.shape}"
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

    def simulate_matches(self, *args, **kwargs):
        """Skellam models do not support detailed match simulation.

        Raises:
        ------
        NotImplementedError
            Skellam models only predict goal differences, not individual home/away goals.
            Use predict() method instead to get goal differences.
        """
        raise NotImplementedError(
            "Skellam models only predict goal differences, not individual home/away goals. "
            "Use predict() method instead to get goal differences."
        )


class SkellamDecay(Skellam):
    """Bayesian Skellam model with temporal decay weighting.

    Combines the efficiency of Skellam goal difference modeling with temporal
    decay weighting to emphasize recent matches. This model gives more weight
    to recent matches when estimating team parameters, making it suitable for
    sports where team strength changes significantly over time.

    Model Structure
    ---------------
    Same as Skellam model but with temporal weights applied:
    - Each match weighted by exp(-decay_rate * days_since_match)
    - Goal differences modeled via Skellam distribution
    - Recent matches have higher influence on parameter estimation

    Parameters
    ----------
    stem : str, default="skellam_decay"
        Name of the Stan model file (without .stan extension)

    Additional Requirements
    ----------------------
    Z : array-like
        Temporal decay weights or days since each match for training data
        Required for fitting, optional for prediction (defaults to 0)

    Examples:
    --------
    >>> model = SkellamDecay()
    >>> days_since = (max_date - match_dates).dt.days
    >>> model.fit(X_train, y_goal_diff, Z=days_since)
    >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        stem: str = "skellam_decay",
    ):
        """Initialize the Skellam model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "skellam_decay   ".
        """
        super().__init__(stem=stem)


class SkellamZero(Skellam):
    """Bayesian Zero-inflated Skellam model for low-scoring sports.

    Extends the Skellam model with zero-inflation to explicitly model the
    probability of draws (goal difference = 0). This is particularly useful
    for sports with frequent draws or low-scoring matches where the standard
    Skellam distribution may underestimate draw probabilities.

    Model Structure
    ---------------
    - With probability π: goal difference = 0 (draw)
    - With probability (1-π): goal difference ~ Skellam(λ_home, λ_away)
    - π ~ prior distribution (zero-inflation probability)
    - Additional flexibility for modeling draw-heavy competitions

    Parameters
    ----------
    stem : str, default="skellam_zero"
        Name of the Stan model file (without .stan extension)

    Use Cases
    ---------
    - Soccer/football leagues with many draws
    - Low-scoring sports (hockey, field hockey)
    - Defensive competitions with frequent ties
    - Any sport where draws are more common than predicted by standard models

    Examples:
    --------
    >>> model = SkellamZero()
    >>> model.fit(X_train, y_goal_diff)  # Will learn zero-inflation parameter
    >>> probs = model.predict_proba(X_test)  # May show higher draw probabilities
    """

    def __init__(
        self,
        stem: str = "skellam_zero",
    ):
        """Initialize the Zero-inflated Skellam model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "skellam_zero".
        """
        super().__init__(stem=stem)


class SkellamZeroDecay(SkellamZero):
    """Bayesian Zero-inflated Skellam model with temporal decay weighting.

    Combines zero-inflation for frequent draws with temporal decay weighting
    for time-varying team strengths. This model is particularly suitable for
    low-scoring sports where both draw frequency and team form changes over
    time are important factors.

    Model Structure
    ---------------
    - Zero-inflated Skellam with temporal weights
    - Each match weighted by exp(-decay_rate * days_since_match)
    - Explicit modeling of draw probability with zero-inflation
    - Recent matches have higher influence on all parameters

    Parameters
    ----------
    stem : str, default="skellam_zero_decay"
        Name of the Stan model file (without .stan extension)

    Additional Requirements
    ----------------------
    Z : array-like
        Temporal decay weights or days since each match for training data
        Required for fitting, optional for prediction (defaults to 0)

    Use Cases
    ---------
    - Long-term league analysis with seasonal changes
    - Sports with both frequent draws and form variations
    - Tournament prediction with recent form emphasis

    Examples:
    --------
    >>> model = SkellamZeroDecay()
    >>> days_since = (max_date - match_dates).dt.days
    >>> model.fit(X_train, y_goal_diff, Z=days_since)
    >>> predictions = model.predict(X_test)  # Emphasizes recent form and draws
    """

    def __init__(
        self,
        stem: str = "skellam_zero_decay",
    ):
        """Initialize the Zero-inflated Skellam Weighted model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "skellam_zero_decay".
        """
        super().__init__(stem=stem)


# %%
if __name__ == "__main__":
    import pandas as pd

    from ssat.data import get_sample_handball_match_data

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load sample data
    df = get_sample_handball_match_data()
    league = "Starligue"
    season = 2024
    match_df = df.loc[(df["league"] == league) & (df["season"] == season)]
    goal_diff = match_df["home_goals"] - match_df["away_goals"]
    dt = match_df["datetime"]

    # Prepare data
    X = match_df[["home_team", "away_team"]]
    y = match_df[["home_goals", "away_goals"]].assign(
        goal_diff=lambda x: x["home_goals"] - x["away_goals"]
    )
    weights = np.random.normal(1, 0.1, len(match_df))

    # Train-test split
    train_size = int(len(match_df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    weights_train, weights_test = weights[:train_size], weights[train_size:]
    dt_train, dt_test = dt[:train_size], dt[train_size:]

    # Days since last match
    Z_train = (dt_train.max() - dt_train).dt.days.astype(int)
    Z_test = (dt_test.max() - dt_test).dt.days.astype(int)

    # instantiate all models
    models = [
        Poisson(),
        PoissonDecay(),
        # NegBinom(),
        # NegBinomDecay(),
        # Skellam(),
        # SkellamDecay(),
        # SkellamZero(),
        # SkellamZeroDecay(),
    ]
    # Fit model
    for model in models:
        print(model)
        name = model.__class__.__name__
        if "Skellam" in name:
            y_train_temp = y_train["goal_diff"]
        else:
            y_train_temp = y_train[["home_goals", "away_goals"]]

        if "Decay" in model.__class__.__name__:
            model.fit(X_train, y_train_temp, Z_train)
        else:
            model.fit(X_train, y_train_temp, weights=weights_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)

        model.plot_trace()
        model.plot_team_stats()
        # Days since last match

# %%
