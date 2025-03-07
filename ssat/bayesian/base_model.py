"""Base Bayesian Model for Sports Match Prediction with Abstract Interface.

This module provides a base class for Bayesian models used in sports match prediction.
It handles common functionality such as:
- MCMC sampling with Stan
- Result caching
- Data validation
- Diagnostic checks
- Parameter management
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import arviz as az
import cmdstanpy
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Type variables for generic typing
T = TypeVar("T", bound="BaseModel")
DataType = TypeVar("DataType", pd.DataFrame, np.ndarray)

# Configure cmdstanpy logging
logger = logging.getLogger("cmdstanpy")
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


@runtime_checkable
class ModelData(Protocol):
    """Protocol defining the required structure for model data."""

    def validate(self) -> bool:
        """Validate the data structure."""
        ...

    def to_stan_input(self) -> Dict[str, Any]:
        """Convert to Stan input format."""
        ...


@dataclass
class MCMCConfig:
    """Configuration for MCMC sampling."""

    draws: int = 2000
    warmup: int = 1000
    chains: int = 4
    seed: int = 1


class ModelError(Exception):
    """Base exception for model-related errors."""

    pass


class ValidationError(ModelError):
    """Raised when data validation fails."""

    pass


class FitError(ModelError):
    """Raised when model fitting fails."""

    pass


class ConvergenceError(ModelError):
    """Raised when MCMC convergence diagnostics fail."""

    pass


@dataclass
class ModelState:
    """Container for model state variables."""

    team_map: Dict[str, int] = field(default_factory=dict)
    is_fitted: bool = False
    fit_result: Optional[cmdstanpy.CmdStanMCMC] = None
    inference_data: Optional[az.InferenceData] = None
    n_teams: Optional[int] = None
    n_samples: Optional[int] = None
    n_chains: int = 0
    n_draws: Optional[int] = None
    model: Optional[cmdstanpy.CmdStanModel] = None
    src_info: Dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """Abstract base class for Bayesian predictive models.

    This class provides core functionality for:
    - Model initialization and configuration
    - Data validation and preprocessing
    - MCMC sampling with Stan
    - Result caching and retrieval
    - Diagnostic checks
    """

    def __init__(
        self,
        stan_file: str = "base",
        mcmc_config: Optional[MCMCConfig] = None,
    ) -> None:
        """Initialize the Bayesian base model.

        Parameters
        ----------
        stan_file : str
            Name of the Stan model file (without .stan extension)
        mcmc_config : Optional[MCMCConfig]
            MCMC sampling configuration

        Raises:
        ------
        ValueError
            If Stan file does not exist
        """
        # Initialize model state
        self._state = ModelState()

        # Configuration
        self._stan_file = Path("ssat/bayesian/stan_files") / f"{stan_file}.stan"
        if not self._stan_file.exists():
            raise ValueError(f"Stan file not found: {self._stan_file}")

        self._mcmc_config = mcmc_config or MCMCConfig()

        # Setup caching
        self._cache_dir = Path("./.model_cache")
        self._cache_dir.mkdir(exist_ok=True)

    def _get_model_hash(
        self, data: Dict[str, Any], draws: int, warmup: int, chains: int
    ) -> str:
        """Generate a unique hash for the model configuration and data.

        Parameters
        ----------
        data : Dict[str, Any]
            Data for Stan model
        draws : int
            Number of posterior samples per chain
        warmup : int
            Number of warmup iterations
        chains : int
            Number of MCMC chains

        Returns:
        -------
        str
            Hash string representing the model configuration
        """
        # Create a string representation of the model configuration
        # Use data keys and shapes instead of full data for efficiency
        data_repr = str(
            {
                k: (np.array(v).shape if isinstance(v, (list, np.ndarray)) else v)
                for k, v in data.items()
            }
        )

        config_str = (
            f"stan_file={self.STAN_FILE},"
            f"data={data_repr},"
            f"draws={draws},"
            f"warmup={warmup},"
            f"chains={chains},"
        )

        # Generate a hash
        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_cache_path(self, model_hash: str) -> Path:
        """Get the path to the cached model results.

        Parameters
        ----------
        model_hash : str
            Hash string representing the model configuration

        Returns:
        -------
        Path
            Path to the cached model file
        """
        return self.cache_dir / f"model_results_{model_hash}.pkl"

    def _compile_and_fit_stan_model(
        self,
        data: Dict[str, Any],
        draws: Optional[int] = None,
        warmup: Optional[int] = None,
        chains: Optional[int] = None,
    ) -> cmdstanpy.CmdStanMCMC:
        """Compile and fit Stan model with MCMC sampling with caching.

        Parameters
        ----------
        data : Dict[str, Any]
            Data for Stan model
        draws : Optional[int]
            Number of posterior samples per chain
        warmup : Optional[int]
            Number of warmup iterations
        chains : Optional[int]
            Number of MCMC chains

        Returns:
        -------
        cmdstanpy.CmdStanMCMC
            Fitted Stan model

        Raises:
        ------
        FitError
            If model fitting fails
        """
        try:
            # Use MCMCConfig defaults if not specified
            mcmc_config = self._mcmc_config
            draws = draws or mcmc_config.draws
            warmup = warmup or mcmc_config.warmup
            chains = chains or mcmc_config.chains

            # Compile model
            model = cmdstanpy.CmdStanModel(stan_file=self._stan_file)

            # Run sampling
            fit_result = model.sample(
                data=data,
                iter_sampling=draws,
                iter_warmup=warmup,
                chains=chains,
                seed=mcmc_config.seed,
            )

            # Update model state
            self._state.is_fitted = True
            self._state.fit_result = fit_result
            self._state.n_chains = chains
            self._state.n_draws = draws
            self._state.n_samples = draws * chains
            self._state.model = model
            self._state.src_info = model.src_info()

            return fit_result

        except Exception as e:
            raise FitError(f"Failed to fit model: {str(e)}") from e

    @abstractmethod
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
    ) -> "BaseModel":
        """Fit the Bayesian model using MCMC sampling.

        Parameters
        ----------
        X : pd.DataFrame
            Training data with team information
        Z : pd.DataFrame
            Additional features, likely home/away goals
        y : Optional[Union[NDArray[np.float64], pd.Series]], default=None
            Target variable (if applicable)
        weights : Optional[NDArray[np.float64]], default=None
            Sample weights
        draws : int, default=2000
            Number of posterior samples per chain
        warmup : int, default=1000
            Number of warmup iterations per chain
        chains : int, default=4
            Number of MCMC chains

        Returns:
        -------
        BaseModel
            Fitted model instance
        """
        pass

    @abstractmethod
    def predict(
        self, X: pd.DataFrame, return_posterior: bool = False
    ) -> Union[NDArray[np.float64], az.InferenceData]:
        """Generate predictions from the posterior distribution.

        Parameters
        ----------
        X : pd.DataFrame
            Data to predict on
        return_posterior : bool, default=False
            If True, return full posterior predictions
            If False, return mean predictions

        Returns:
        -------
        Union[NDArray[np.float64], az.InferenceData]
            Predictions or posterior samples
        """
        pass

    @abstractmethod
    def predict_proba(
        self,
        X: pd.DataFrame,
        Z: Optional[pd.DataFrame] = None,
        point_spread: float = 0.0,
        include_draw: bool = True,
        outcome: Optional[str] = None,
        threshold: Optional[float] = None,
        return_posterior: bool = False,
    ) -> Union[NDArray[np.float64], az.InferenceData]:
        """Predict match outcome probabilities using posterior samples.

        Parameters
        ----------
        X : pd.DataFrame
            Data to predict on
        Z : Optional[pd.DataFrame], default=None
            Additional features
        point_spread : float, default=0.0
            Point spread adjustment
        include_draw : bool, default=True
            Whether to include draw probability
        outcome : Optional[str], default=None
            Specific outcome to predict
        threshold : Optional[float], default=None
            Threshold for draw outcomes
        return_posterior : bool, default=False
            If True, return full posterior probabilities
            If False, return mean probabilities

        Returns:
        -------
        Union[NDArray[np.float64], az.InferenceData]
            Predicted probabilities or posterior samples
        """
        pass

    @abstractmethod
    def get_team_ratings(self) -> pd.DataFrame:
        """Get team ratings with uncertainty quantification.

        Returns:
        -------
        pd.DataFrame
            Team ratings including:
            - Posterior means
            - Credible intervals
            - Additional team-specific parameters
        """
        pass

    @abstractmethod
    def _generate_inference_data(self) -> None:
        """Generate ArviZ inference data for analysis.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with home and away teams for prediction.
        Z : pd.DataFrame
            Additional data for prediction
            (e.g., home and away goals, match weights).

        Returns:
        -------
        az.InferenceData
            Inference data object containing:
            - Posterior samples
            - Observed data
            - Prior samples (if available)
            - MCMC diagnostics
        """
        pass

    def _validate_X(self, X: pd.DataFrame, fit: bool = True) -> None:
        """Validate input data structure.

        Parameters
        ----------
        X : pd.DataFrame
            Input data
        fit : bool, default=True
            Whether validation is for fitting or prediction

        Raises:
        ------
        ValueError
            If validation fails
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        min_cols = 2
        if X.shape[1] < min_cols:
            raise ValueError(f"X must have at least {min_cols} columns")

        if not fit and X.shape[1] != 2:
            raise ValueError("X must have exactly 2 columns for prediction")

        for i in range(2):
            if not pd.api.types.is_string_dtype(X.iloc[:, i]):
                raise ValueError(f"Column {i} must contain string values (team names)")

    def _validate_Z(
        self, X: pd.DataFrame, Z: pd.DataFrame, require_goals: bool = False
    ) -> None:
        """Validate Z DataFrame dimensions and content.

        Parameters
        ----------
        X : pd.DataFrame
            Input data
        Z : pd.DataFrame
            Additional data
        require_goals : bool, default=False
            Whether to require home_goals and away_goals columns

        Raises:
        ------
        ValueError
            If validation fails
        """
        if Z is None and require_goals:
            raise ValueError("Z must be provided with home_goals and away_goals data")
        if Z is not None and len(Z) != len(X):
            raise ValueError("Z must have the same number of rows as X")

    def _validate_teams(self, teams: List[str]) -> None:
        """Validate team existence in the model.

        Parameters
        ----------
        teams : List[str]
            List of team names

        Raises:
        ------
        ValueError
            If any team is unknown
        """
        for team in teams:
            if team not in self.team_map_:
                raise ValueError(f"Unknown team: {team}")

    def _check_is_fitted(self) -> None:
        """Check if model has been fitted.

        Raises:
        ------
        ValueError
            If model is not fitted
        """
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet")

    def _validate_mcmc_diagnostics(self) -> None:
        """Validate MCMC convergence diagnostics.

        This method checks three key MCMC diagnostics:
        1. R-hat (potential scale reduction factor)
        2. Bulk effective sample size (ESS)
        3. Tail effective sample size (ESS)

        Raises:
        ------
        ValueError
            If fit_result is not available
        ConvergenceError
            If any diagnostic check fails
        """
        if self.fit_result is None:
            raise ValueError("No fit results available")

        try:
            summary_df = self.fit_result.summary()

            # Check R-hat values
            rhat_threshold = 1.01
            rhat_values = np.asarray(summary_df["R_hat"].values, dtype=np.float64)
            bad_rhat = rhat_values > rhat_threshold
            if np.any(bad_rhat):
                params = summary_df.index[bad_rhat].tolist()
                raise ConvergenceError(
                    f"Poor MCMC convergence detected. R-hat > {rhat_threshold} "
                    f"for parameters: {params}"
                )

            # Check Bulk-ESS
            bulk_ess_threshold = 100 * self.n_chains_
            bulk_ess_values = np.asarray(
                summary_df["ESS_bulk"].values, dtype=np.float64
            )
            low_bulk_ess = bulk_ess_values < bulk_ess_threshold
            if np.any(low_bulk_ess):
                params = summary_df.index[low_bulk_ess].tolist()
                raise ConvergenceError(
                    f"Low bulk effective sample size. ESS_bulk < {bulk_ess_threshold} "
                    f"for parameters: {params}"
                )

            # Check Tail-ESS
            tail_ess_threshold = 100 * self.n_chains_
            tail_ess_values = np.asarray(
                summary_df["ESS_tail"].values, dtype=np.float64
            )
            low_tail_ess = tail_ess_values < tail_ess_threshold
            if np.any(low_tail_ess):
                params = summary_df.index[low_tail_ess].tolist()
                raise ConvergenceError(
                    f"Low tail effective sample size. ESS_tail < {tail_ess_threshold} "
                    f"for parameters: {params}"
                )

        except Exception as e:
            if isinstance(e, ConvergenceError):
                raise
            raise ConvergenceError(
                f"Failed to validate MCMC diagnostics: {str(e)}"
            ) from e

    @abstractmethod
    def _plot_diagnostics(self) -> None:
        """Plot MCMC diagnostics."""
        pass

    @property
    def team_map_(self) -> Dict[str, int]:
        """Maintain backward compatibility for team_map_ attribute."""
        return self._state.team_map

    @team_map_.setter
    def team_map_(self, value: Dict[str, int]) -> None:
        """Set team map."""
        self._state.team_map = value

    @property
    def is_fitted_(self) -> bool:
        """Maintain backward compatibility for is_fitted_ attribute."""
        return self._state.is_fitted

    @is_fitted_.setter
    def is_fitted_(self, value: bool) -> None:
        """Set fitted status."""
        self._state.is_fitted = value

    @property
    def fit_result(self) -> Optional[cmdstanpy.CmdStanMCMC]:
        """Access to fit result."""
        return self._state.fit_result

    @fit_result.setter
    def fit_result(self, value: Optional[cmdstanpy.CmdStanMCMC]) -> None:
        """Set fit result."""
        self._state.fit_result = value

    @property
    def inference_data(self) -> Optional[az.InferenceData]:
        """Access to inference data."""
        return self._state.inference_data

    @inference_data.setter
    def inference_data(self, value: Optional[az.InferenceData]) -> None:
        """Set inference data."""
        self._state.inference_data = value

    @property
    def n_teams_(self) -> Optional[int]:
        """Access to number of teams."""
        return self._state.n_teams

    @n_teams_.setter
    def n_teams_(self, value: Optional[int]) -> None:
        """Set number of teams."""
        self._state.n_teams = value

    @property
    def n_samples_(self) -> Optional[int]:
        """Access to number of samples."""
        return self._state.n_samples

    @n_samples_.setter
    def n_samples_(self, value: Optional[int]) -> None:
        """Set number of samples."""
        self._state.n_samples = value

    @property
    def n_chains_(self) -> int:
        """Access to number of chains."""
        return self._state.n_chains

    @n_chains_.setter
    def n_chains_(self, value: int) -> None:
        """Set number of chains."""
        self._state.n_chains = value

    @property
    def n_draws_(self) -> Optional[int]:
        """Access to number of draws."""
        return self._state.n_draws

    @n_draws_.setter
    def n_draws_(self, value: Optional[int]) -> None:
        """Set number of draws."""
        self._state.n_draws = value

    @property
    def STAN_FILE(self) -> Path:
        """Access to Stan file path."""
        return self._stan_file

    @property
    def cache_dir(self) -> Path:
        """Access to cache directory."""
        return self._cache_dir

    @property
    def model(self) -> Optional[cmdstanpy.CmdStanModel]:
        """Access to Stan model."""
        return self._state.model

    @model.setter
    def model(self, value: Optional[cmdstanpy.CmdStanModel]) -> None:
        """Set Stan model."""
        self._state.model = value

    @property
    def src_info(self) -> Dict[str, Any]:
        """Access to model source information."""
        return self._state.src_info

    @src_info.setter
    def src_info(self, value: Dict[str, Any]) -> None:
        """Set model source information."""
        self._state.src_info = value
