"""Bayesian Negative Binomial Hierarchical Model."""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ssat.bayesian.base_model import BaseModel, FitError
from ssat.bayesian.hierarchical_base import HierarchicalBaseModel


class SkellamHierarchical(HierarchicalBaseModel):
    """Bayesian Negative Binomial Hierarchical Model."""

    def __init__(
        self,
        stem: str = "skellam_hierachichal",
    ):
        """Initialize the Negative Binomial Hierarchical model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "skellam_hierachichal".
        """
        super().__init__(stan_file=stem)

    def fit(
        self,
        X: pd.DataFrame,
        Z: pd.DataFrame,
        y: Optional[Union[NDArray[np.float64], pd.Series]] = None,
        weights: Optional[NDArray[np.float64]] = None,
        draws: int = 2000,
        warmup: int = 1000,
        chains: int = 4,
        validate: bool = False,
        plot_diagnostics: bool = False,
    ) -> "SkellamHierarchical":
        """Fit the Bayesian Hierarchical Negative Binomial model.

        Parameters
        ----------
        X : pd.DataFrame
            Training data with team names.
        Z : pd.DataFrame
            Additional data, must include home and away goals.
        y : Optional[Union[NDArray[np.float64], pd.Series]], optional
            Target variable (not used in this model). Defaults to None.
        weights : Optional[NDArray[np.float64]], optional
            Match weights. Defaults to None.
        draws : int, optional
            Number of MCMC draws. Defaults to 2000.
        warmup : int, optional
            Warmup iterations for MCMC. Defaults to 1000.
        chains : int, optional
            Number of MCMC chains. Defaults to 4.

        Returns:
        -------
        NegBinomHierarchical
            Fitted model instance.
        """
        self._validate_X(X)
        self._validate_Z(X, Z)

        # Prepare data for Stan model
        data = self._data_dict(X, Z, weights, fit=True)

        # Compile and fit Stan model using base class method
        self.fit_result = self._compile_and_fit_stan_model(data, draws, warmup, chains)

        if validate:
            # Validate the MCMC diagnostics
            self._validate_mcmc_diagnostics()

        if plot_diagnostics:
            # Plot Trace plots
            self._plot_diagnostics()

        # Generate inference data for analysis
        self._generate_inference_data(data)

        return self

    def predict(
        self,
        X: pd.DataFrame,
        return_matches: bool = False,
        return_uncertainty: bool = False,
        credible_interval: float = 0.95,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Predict match outcomes with uncertainty quantification.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with home and away teams for prediction.
        return_matches : bool, optional
            Whether to return individual match predictions. Defaults to False.
        return_uncertainty : bool, optional
            Whether to return uncertainty metrics. Defaults to False.
        credible_interval : float, optional
            Credible interval level (0 to 1). Defaults to 0.95.

        Returns:
        -------
        Union[np.ndarray, Dict[str, np.ndarray]]
            If return_uncertainty=False:
                Predicted score differences (home - away)
            If return_uncertainty=True:
                Dictionary containing:
                - 'mean': Mean predictions
                - 'std': Standard deviation of predictions
                - 'lower': Lower bound of credible interval
                - 'upper': Upper bound of credible interval

        Raises:
        ------
        ValueError
            If model is not fitted, teams are unknown, or invalid parameters
        FitError
            If prediction fails
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fit before making predictions")

        if not 0 < credible_interval < 1:
            raise ValueError("credible_interval must be between 0 and 1")

        self._validate_X(X, fit=False)
        self._validate_teams(X.values[:, 0:2].flatten().tolist())

        data = self._data_dict(X, X, fit=False)

        if self.model is None or self.fit_result is None:
            raise ValueError("Model not properly initialized")

        try:
            # Generate predictions using Stan model
            preds = self.model.generate_quantities(
                data=data, previous_fit=self.fit_result
            )

            # Calculate score differences for each posterior sample
            predicted_spreads = preds.stan_variable("pred_goal_diff_match")

            if return_matches:
                return predicted_spreads

            # Calculate mean predictions
            mean_preds = predicted_spreads.mean(axis=0)

            if not return_uncertainty:
                return mean_preds

            # Calculate uncertainty metrics
            std_preds = predicted_spreads.std(axis=0)
            alpha = (1 - credible_interval) / 2
            lower = np.percentile(predicted_spreads, 100 * alpha, axis=0)
            upper = np.percentile(predicted_spreads, 100 * (1 - alpha), axis=0)

            return {
                "mean": mean_preds,
                "std": std_preds,
                "lower": lower,
                "upper": upper,
            }

        except Exception as e:
            raise FitError(f"Failed to generate predictions: {str(e)}") from e

    def predict_proba(
        self,
        X: pd.DataFrame,
        point_spread: float = 0.0,
        include_draw: bool = True,
        outcome: Optional[str] = None,
        threshold: float = 0.0,
        return_uncertainty: bool = False,
        credible_interval: float = 0.95,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Predict match outcome probabilities with uncertainty quantification.

        Parameters
        ----------
        X : pd.DataFrame
            Data to predict on.
        point_spread : float, default=0.0
            Point spread adjustment (positive favors home team).
        include_draw : bool, default=True
            Whether to include draw probability.
        outcome : Optional[str], default=None
            Specific outcome to predict ('home', 'away', or 'draw').
        threshold : float, default=0.0
            Threshold for predicting draw outcome.
        return_uncertainty : bool, default=False
            Whether to return uncertainty metrics.
        credible_interval : float, default=0.95
            Credible interval level (0 to 1).

        Returns:
        -------
        Union[np.ndarray, Dict[str, np.ndarray]]
            If return_uncertainty=False:
                Predicted probabilities for each outcome.
            If return_uncertainty=True:
                Dictionary containing:
                - 'probabilities': Mean probabilities
                - 'std': Standard deviation of probabilities
                - 'lower': Lower bound of credible interval
                - 'upper': Upper bound of credible interval

        Raises:
        ------
        ValueError
            If model is not fitted, teams are unknown, or invalid parameters
        FitError
            If prediction fails
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fit before making predictions")

        if not 0 < credible_interval < 1:
            raise ValueError("credible_interval must be between 0 and 1")

        if threshold < 0:
            raise ValueError("threshold must be non-negative")

        if outcome not in [None, "home", "away", "draw"]:
            raise ValueError("outcome must be None, 'home', 'away', or 'draw'")

        self._validate_X(X, fit=False)
        self._validate_teams(X.values[:, 0:2].flatten().tolist())

        data = self._data_dict(X, X, fit=False)

        if self.model is None or self.fit_result is None:
            raise ValueError("Model not properly initialized")

        try:
            # Generate predictions
            new_quantities = self.model.generate_quantities(
                data=data, previous_fit=self.fit_result
            )

            spread = new_quantities.stan_variable("pred_goal_diff_match")

            # Initialize arrays for storing probabilities
            n_samples = spread.shape[0]
            n_matches = spread.shape[1]
            n_classes = 3 if include_draw else 2

            # Calculate probabilities for each posterior sample
            sample_probs = np.zeros(
                (n_samples, n_matches, n_classes if outcome is None else 1)
            )

            # Calculate outcome masks with threshold
            draw_mask = (spread >= -threshold) & (spread <= threshold)
            home_win_mask = spread > threshold
            away_win_mask = spread < -threshold

            # Handle numerical stability for probability calculation
            eps = np.finfo(float).eps
            total_outcomes = (
                draw_mask.astype(float)
                + home_win_mask.astype(float)
                + away_win_mask.astype(float)
            )
            total_outcomes = np.maximum(total_outcomes, eps)  # Avoid division by zero

            if outcome is None:
                sample_probs[..., 0] = home_win_mask.astype(float) / total_outcomes
                if include_draw:
                    sample_probs[..., 1] = draw_mask.astype(float) / total_outcomes
                sample_probs[..., -1] = away_win_mask.astype(float) / total_outcomes
            else:
                if outcome == "home":
                    sample_probs = home_win_mask.astype(float) / total_outcomes
                elif outcome == "away":
                    sample_probs = away_win_mask.astype(float) / total_outcomes
                elif outcome == "draw":
                    sample_probs = draw_mask.astype(float) / total_outcomes

            # Calculate mean probabilities
            mean_probs = sample_probs.mean(axis=0)

            if not return_uncertainty:
                return mean_probs

            # Calculate uncertainty metrics
            std_probs = sample_probs.std(axis=0)
            alpha = (1 - credible_interval) / 2
            lower = np.percentile(sample_probs, 100 * alpha, axis=0)
            upper = np.percentile(sample_probs, 100 * (1 - alpha), axis=0)

            return {
                "probabilities": mean_probs,
                "std": std_probs,
                "lower": lower,
                "upper": upper,
            }

        except Exception as e:
            raise FitError(
                f"Failed to generate probability predictions: {str(e)}"
            ) from e
