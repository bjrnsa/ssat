# %%
"""Bayesian Poisson Model for sports prediction."""

from typing import Optional, Union

import numpy as np
import pandas as pd

from ssat.bayesian.poisson_models import Poisson
from ssat.stats.skellam_optim import qskellam


class Skellam(Poisson):
    """Bayesian Skellam Model for predicting match scores.

    This model uses a Skellam distribution (difference of two Poisson distributions)
    to directly model the goal difference between teams, accounting for both team
    attack and defense capabilities.
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
    ) -> np.ndarray:
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
    ) -> np.ndarray:
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
    """Bayesian Skellam Model for predicting match scores.

    This model uses a Skellam distribution (difference of two Poisson distributions)
    to directly model the goal difference between teams, accounting for both team
    attack and defense capabilities.
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
    """Bayesian Zero-inflated Skellam Model for predicting match scores.

    This model uses a zero-inflated Skellam distribution to model goal differences,
    particularly suitable for low-scoring matches or competitions with frequent draws.
    The zero-inflation component explicitly models the probability of a draw.
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
    """Bayesian Zero-inflated Skellam Model for predicting match scores.

    This model uses a zero-inflated Skellam distribution to model goal differences,
    particularly suitable for low-scoring matches or competitions with frequent draws.
    The zero-inflation component explicitly models the probability of a draw.
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
