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
import re
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

        # Parse Stan file and print data requirements
        self._parse_stan_file()
        self._print_data_requirements()

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

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        draws: Optional[int] = None,
        warmup: Optional[int] = None,
        chains: Optional[int] = None,
    ) -> "BaseModel":
        """Fit the model using MCMC sampling.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            Training data containing all required columns as specified by _print_data_requirements
        draws : Optional[int], optional
            Number of posterior samples per chain, by default None
        warmup : Optional[int], optional
            Number of warmup iterations, by default None
        chains : Optional[int], optional
            Number of MCMC chains, by default None

        Returns:
        -------
        BaseModel
            The fitted model instance

        Raises:
        ------
        FitError
            If model fitting fails
        ValueError
            If data validation fails
        """
        # Prepare data dictionary
        data_dict = self._data_dict(data, fit=True)

        # Run MCMC sampling
        try:
            self._compile_and_fit_stan_model(
                data=data_dict,
                draws=draws,
                warmup=warmup,
                chains=chains,
            )

            # Generate inference data
            self._generate_inference_data(data_dict)

        except Exception as e:
            raise FitError(f"Failed to fit model: {str(e)}") from e

        return self

    def predict(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        return_matches: bool = False,
    ) -> np.ndarray:
        """Generate predictions for new data.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            Data array or DataFrame. If array, columns must be in order shown by _print_data_requirements.
            If DataFrame, column names don't matter but order of columns must match requirements.
        return_matches : bool, optional
            Whether to return individual match predictions, by default False

        Returns:
        -------
        np.ndarray
            Predicted values

        Raises:
        ------
        ValueError
            If model is not fitted
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fit before making predictions")

        # Prepare data dictionary for prediction
        data_dict = self._data_dict(data, fit=False)

        if self.model is None or self.fit_result is None:
            raise ValueError("Model not properly initialized")

        try:
            # Generate predictions using Stan model
            preds = self.model.generate_quantities(
                data=data_dict, previous_fit=self.fit_result
            )

            # Find prediction variables in Stan output
            pred_vars = [var for var in preds.keys() if var.startswith("pred_")]
            if not pred_vars:
                raise ValueError("No prediction variables found in Stan output")

            # Get main prediction variable (usually ends with _match)
            match_vars = [var for var in pred_vars if var.endswith("_match")]
            if not match_vars:
                # If no _match variables, use first prediction variable
                pred_var = pred_vars[0]
            else:
                pred_var = match_vars[0]

            # Extract predictions
            predictions = preds.stan_variable(pred_var)

            # Average over samples if not returning matches
            if not return_matches:
                predictions = predictions.mean(axis=0)

            return predictions

        except Exception as e:
            raise FitError(f"Failed to generate predictions: {str(e)}") from e

    def predict_proba(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        point_spread: float = 0.0,
        include_draw: bool = True,
        outcome: Optional[str] = None,
        threshold: float = 0.0,
    ) -> np.ndarray:
        """Generate probability predictions for new data.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            Data array or DataFrame. If array, columns must be in order shown by _print_data_requirements.
            If DataFrame, column names don't matter but order of columns must match requirements.
        point_spread : float, optional
            Point spread adjustment (positive favors home team), by default 0.0
        include_draw : bool, optional
            Whether to include draw probability, by default True
        outcome : Optional[str], optional
            Specific outcome to predict ('home', 'away', or 'draw'), by default None
        threshold : float, optional
            Threshold for predicting draw outcome, by default 0.0

        Returns:
        -------
        np.ndarray
            Predicted probabilities for each outcome

        Raises:
        ------
        ValueError
            If model is not fitted or if threshold is negative or if outcome is invalid
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fit before making predictions")

        if threshold < 0:
            raise ValueError("threshold must be non-negative")

        if outcome not in [None, "home", "away", "draw"]:
            raise ValueError("outcome must be None, 'home', 'away', or 'draw'")

        # Get raw predictions first
        predictions = self.predict(data, return_matches=True)

        # Apply point spread
        if point_spread != 0:
            predictions = predictions + point_spread

        # Initialize arrays for storing probabilities
        n_samples = predictions.shape[0]
        n_matches = predictions.shape[1]
        n_classes = 3 if include_draw else 2

        # Calculate probabilities for each posterior sample
        sample_probs = np.zeros(
            (n_samples, n_matches, n_classes if outcome is None else 1)
        )

        # Calculate outcome masks with threshold
        draw_mask = (predictions >= -threshold) & (predictions <= threshold)
        home_win_mask = predictions > threshold
        away_win_mask = predictions < -threshold

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

        # Average over samples
        return sample_probs.mean(axis=0)

    # @abstractmethod
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

    def _generate_inference_data(self, data: Dict[str, Any]) -> None:
        """Generate inference data from Stan fit result.

        This method dynamically creates an ArviZ InferenceData object based on the Stan model's
        variables, organizing them into appropriate groups (observed data, parameters, etc).

        Parameters:
        ----------
        data : Dict[str, Any]
            Data dictionary used for fitting

        Raises:
        ------
        ValueError
            If model is not fitted
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fit before generating inference data")

        if self.model is None or self.fit_result is None:
            raise ValueError("Model not properly initialized")

        # Get all variables from Stan output
        param_names = self.fit_result.param_names
        if not param_names:
            raise ValueError("No parameters found in Stan output")

        # Group variables by their role
        observed_data = {}
        coords = {}
        dims = {}

        # Add data variables
        data_vars = {}
        for var in self._data_vars:
            name = var["name"]
            if name in data and name not in ["N", "T"]:  # Skip dimensions
                data_vars[name] = data[name]

        # Add coordinates for team indices if present
        if hasattr(self, "_entities"):
            coords["team"] = list(self._entities)
            for var in self._data_vars:
                if var["name"].endswith("_idx_match"):
                    base_name = var["name"].replace("_idx_match", "")
                    dims[base_name] = ["match"]
                    data_vars[base_name] = self._entities[data[var["name"]] - 1]

        # Create inference data
        self.inference_data = az.from_cmdstanpy(
            posterior=self.fit_result,
            posterior_predictive="pred_",
            observed_data=data_vars,
            coords=coords,
            dims=dims,
        )

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

    def _parse_stan_file(self) -> None:
        """Parse the Stan file to extract data block requirements."""
        with open(self._stan_file, "r") as f:
            content = f.read()

        # Find data block
        data_match = re.search(r"data\s*{([^}]*)}", content, re.DOTALL)
        if not data_match:
            raise ValueError(f"No data block found in {self._stan_file}")

        data_block = data_match.group(1)

        # Parse variable declarations
        self._data_vars = []
        for line in data_block.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("//"):  # Skip empty lines and comments
                # Extract type, name, and comment if exists
                parts = line.split(";")[0].split("//")
                declaration = parts[0].strip()
                comment = parts[1].strip() if len(parts) > 1 else ""

                # Parse array and constraints
                array_match = re.match(r"array\[([^\]]+)\]", declaration)
                if array_match:
                    array_dims = array_match.group(1)
                    declaration = re.sub(r"array\[[^\]]+\]\s*", "", declaration)
                else:
                    array_dims = None

                # Extract constraints
                constraints = re.findall(r"<[^>]+>", declaration)
                constraints = constraints[0] if constraints else None

                # Clean up type and name
                clean_decl = re.sub(r"<[^>]+>", "", declaration)
                parts = clean_decl.split()
                var_type = parts[0]
                var_name = parts[-1]

                self._data_vars.append(
                    {
                        "name": var_name,
                        "type": var_type,
                        "array_dims": array_dims,
                        "constraints": constraints,
                        "description": comment,
                    }
                )

    def _print_data_requirements(self) -> None:
        """Print the data requirements for this model."""
        print(f"\nData requirements for {self._stan_file.name}:")
        print("-" * 50)

        # Group variables by their role
        index_vars = []
        dimension_vars = []
        data_vars = []
        weight_vars = []

        for var in self._data_vars:
            if var["name"].endswith("_idx_match"):
                index_vars.append(var)
            elif var["name"] in ["N", "T"]:
                dimension_vars.append(var)
            elif var["name"].endswith("_match"):
                if "weights" in var["name"]:
                    weight_vars.append(var)
                else:
                    data_vars.append(var)

        print("Required columns (in order):")
        col_idx = 0

        print("  Index columns (first columns):")
        for var in index_vars:
            name = var["name"].replace("_idx_match", "")
            constraints = var["constraints"] or ""
            desc = var["description"] or f"{name.replace('_', ' ').title()} index"
            print(f"    {col_idx}. {desc} {constraints}")
            col_idx += 1

        print("\n  Data columns:")
        for var in data_vars:
            name = var["name"].replace("_match", "")
            type_str = "int" if var["type"] == "int" else "float"
            desc = var["description"] or f"{name.replace('_', ' ').title()}"
            print(f"    {col_idx}. {desc} ({type_str})")
            col_idx += 1

        if weight_vars:
            print("\n  Optional columns:")
            for var in weight_vars:
                name = var["name"].replace("_match", "")
                desc = var["description"] or "Sample weights"
                print(f"    {col_idx}. {desc} (float, optional)")
                col_idx += 1

        print("\nExample usage:")
        print("  # Using a DataFrame:")
        print(
            "  data = pd.DataFrame(your_data)  # columns must be in order shown above"
        )
        print("  model.fit(data)")
        print("\n  # Using a numpy array:")
        print("  data = np.array(your_data)  # columns must be in order shown above")
        print("  model.fit(data)")

        print("\nRaw Stan data requirements:")
        print("-" * 30)
        for var in self._data_vars:
            type_str = var["type"]
            if var["array_dims"]:
                type_str = f"array[{var['array_dims']}] {type_str}"
            if var["constraints"]:
                type_str = f"{type_str} {var['constraints']}"
            desc = f" // {var['description']}" if var["description"] else ""
            print(f"{type_str} {var['name']};{desc}")
        print("-" * 50)

    def _data_dict(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        fit: bool = True,
    ) -> Dict[str, Any]:
        """Prepare data dictionary for Stan model dynamically based on Stan file requirements.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            Data array or DataFrame. If array, columns must be in order shown by _print_data_requirements.
            If DataFrame, column names don't matter but order of columns must match requirements.
        fit : bool, optional
            Whether the data is for fitting or prediction. Defaults to True.

        Returns:
        -------
        Dict[str, Any]
            Data dictionary for Stan model.

        Raises:
        ------
        ValueError:
            If data doesn't match requirements or has incorrect types
        """
        # Convert data to numpy array if DataFrame
        if isinstance(data, pd.DataFrame):
            data_array = data.to_numpy()
        else:
            data_array = np.asarray(data)

        # Initialize data dictionary with dimensions
        data_dict = {
            "N": len(data_array),
        }

        # Group variables by their role
        index_vars = []
        dimension_vars = []
        data_vars = []
        weight_vars = []

        for var in self._data_vars:
            if var["name"].endswith("_idx_match"):
                index_vars.append(var)
            elif var["name"] in ["N", "T"]:
                dimension_vars.append(var)
            elif var["name"].endswith("_match"):
                if "weights" in var["name"]:
                    weight_vars.append(var)
                else:
                    data_vars.append(var)

        # Track current column index
        col_idx = 0

        # Handle index columns (e.g., team indices)
        if index_vars:
            # Get unique entities and create mapping
            index_cols = []
            for _ in index_vars:
                if col_idx >= data_array.shape[1]:
                    raise ValueError(
                        f"Not enough columns in data. Expected index column at position {col_idx}"
                    )
                index_cols.append(data_array[:, col_idx])
                col_idx += 1

            teams = np.unique(np.concatenate(index_cols))
            n_teams = len(teams)
            team_map = {entity: idx + 1 for idx, entity in enumerate(teams)}

            # Store dimensions and mapping for future use
            data_dict["T"] = n_teams
            if fit:
                self._team_map = team_map
                self._n_teams = n_teams
                self._entities = teams

            # Create index arrays
            for var in index_vars:
                if not fit:
                    # Validate entities exist in mapping
                    unknown = set(data_array[:, col_idx - len(index_vars)]) - set(
                        self._team_map.keys()
                    )
                    if unknown:
                        raise ValueError(
                            f"Unknown entities in column {col_idx - len(index_vars)}: {unknown}"
                        )
                    team_map = self._team_map

                data_dict[var["name"]] = np.array(
                    [
                        team_map[entity]
                        for entity in data_array[:, col_idx - len(index_vars)]
                    ]
                )

        # Handle data columns
        for var in data_vars:
            if col_idx >= data_array.shape[1]:
                if not fit:
                    # For prediction, use zeros if column not provided
                    data_dict[var["name"]] = np.zeros(
                        len(data_array),
                        dtype=np.int32 if var["type"] == "int" else np.float64,
                    )
                    continue
                else:
                    raise ValueError(
                        f"Not enough columns in data. Expected data column at position {col_idx}"
                    )

            # Convert to correct type
            if var["type"] == "int":
                data_dict[var["name"]] = np.array(
                    data_array[:, col_idx], dtype=np.int32
                )
            else:
                data_dict[var["name"]] = np.array(
                    data_array[:, col_idx], dtype=np.float64
                )
            col_idx += 1

        # Handle weights
        for var in weight_vars:
            if col_idx < data_array.shape[1]:
                data_dict[var["name"]] = np.array(
                    data_array[:, col_idx], dtype=np.float64
                )
                col_idx += 1
            else:
                data_dict[var["name"]] = np.ones(len(data_array), dtype=np.float64)

        return data_dict
