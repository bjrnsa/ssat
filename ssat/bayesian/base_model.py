"""Abstract base interface for Bayesian sports match prediction models.

This module defines the BaseModel abstract base class that provides a standardized
interface for all Bayesian predictive models in the SSAT framework. It handles
Stan file parsing, data requirements extraction, and defines the contract that
all concrete model implementations must follow.

Classes
-------
BaseModel : ABC
    Abstract base class defining the interface for Bayesian predictive models

The BaseModel provides:
- Stan file parsing and validation
- Automatic data requirements extraction from Stan model specifications
- Abstract methods for fitting, prediction, and model validation
- Standardized API for team validation and data preprocessing
- Built-in help system showing model-specific data requirements

All concrete model implementations should inherit from this class and implement
the required abstract methods for their specific distribution and modeling approach.
"""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Configure cmdstanpy logging
logger = logging.getLogger("cmdstanpy")
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

STANFILES_ROOT = Path(__file__).parent / "stan_files"


class BaseModel(ABC):
    """Abstract base class for Bayesian predictive models."""

    def __init__(
        self,
        stan_file: str = "base",
    ) -> None:
        """Initialize the Bayesian base model.

        Parameters
        ----------
        stan_file : str
            Name of the Stan model file (without .stan extension)

        Raises:
        ------
        ValueError
            If Stan file does not exist
        """
        # Configuration
        self._stan_file = STANFILES_ROOT / f"{stan_file}.stan"
        if not self._stan_file.exists():
            raise ValueError(f"Stan file not found: {self._stan_file}")
        self.name = self._stan_file.stem
        # Parse Stan file to extract data requirements
        self._parse_stan_file()

    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        Z: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        weights: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs,
    ) -> "BaseModel":
        """Fit the model using MCMC sampling.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Team pairs data with exactly 2 columns: [home_team, away_team]
        y : Union[np.ndarray, pd.Series, pd.DataFrame]
            Target variable (REQUIRED):
            - For Poisson/NegBinom: Home goals, Away goals (2 columns)
            - For Skellam: Goal differences (1 column)
        Z : Optional[Union[pd.DataFrame, np.ndarray]], default=None
            Additional match-level features (e.g., match importance, referee data)
        weights : Optional[Union[np.ndarray, pd.Series]], default=None
            Sample weights for each match. If None, uses equal weights.
        **kwargs : dict
            Additional keyword arguments for sampling

        Returns:
        -------
        BaseModel
            The fitted model instance
        """
        pass

    @abstractmethod
    def _data_dict(
        self,
        base_data: Union[np.ndarray, pd.DataFrame],
        model_data: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        optional_data: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        fit: bool = True,
    ) -> Dict[str, Any]:
        """Prepare data dictionary for Stan model.

        Parameters
        ----------
        base_data : Union[np.ndarray, pd.DataFrame]
            Base data required by all models
        model_data : Optional[Union[np.ndarray, pd.DataFrame, pd.Series]], optional
            Additional model-specific data
        optional_data : Optional[Union[np.ndarray, pd.DataFrame, pd.Series]], optional
            Optional model-specific data
        fit : bool, optional
            Whether this is for fitting (True) or prediction (False)

        Returns:
        -------
        Dict[str, Any]
            Dictionary of data for Stan model
        """
        pass

    @abstractmethod
    def _generate_inference_data(self, data: Dict[str, Any]) -> None:
        """Generate inference data from Stan fit result."""
        pass

    @abstractmethod
    def _validate_teams(self, teams: List[str]) -> None:
        """Validate team existence in the model."""
        pass

    @abstractmethod
    def _check_is_fitted(self) -> None:
        """Check if the model has been fitted."""
        pass

    @abstractmethod
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        Z: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        point_spread: int = 0,
    ) -> np.ndarray:
        """Generate predictions for new data.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Team pairs data with exactly 2 columns: [home_team, away_team]
        Z : Optional[Union[pd.DataFrame, np.ndarray]], default=None
            Additional match-level features
        point_spread : int, default=0
            Point spread adjustment applied to predictions

        Returns:
        -------
        np.ndarray
            Predicted values (goal differences or similar)
        """
        pass

    @abstractmethod
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
            Threshold for outcome prediction

        Returns:
        -------
        np.ndarray
            Predicted probabilities
        """
        pass

    @abstractmethod
    def simulate_matches(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        Z: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> np.ndarray:
        """Generate individual match simulations.

        This method simulates detailed match outcomes, such as individual
        home and away goals for models that support this level of detail.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Team pairs data with exactly 2 columns: [home_team, away_team]
        Z : Optional[Union[pd.DataFrame, np.ndarray]], default=None
            Additional match-level features

        Returns:
        -------
        np.ndarray
            Simulated match details. Format depends on model type:
            - Poisson/NegBinom models: (n_matches, 2) for [home_goals, away_goals]
            - Skellam models: May raise NotImplementedError if only goal differences are available

        Raises:
        ------
        NotImplementedError
            If the model type does not support detailed match simulation
            (e.g., models that only predict goal differences)
        """
        pass

    def _parse_stan_file(self) -> None:
        with open(self._stan_file, "r") as f:
            content = f.read()

        # Find model block
        model_match = re.search(r"model\s*{([^}]*)}", content, re.DOTALL)
        if not model_match:
            raise ValueError(f"No model block found in {self._stan_file}")

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
                if "vector" in parts[0]:
                    parts[0] = parts[0].replace("vector", "float").replace("[N]", " N")
                declaration = parts[0].strip()

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
                comment = (
                    line.strip().split("//")[-1].strip()
                    if len(parts) > 1
                    else parts[1].strip()
                )

                self._data_vars.append(
                    {
                        "name": var_name,
                        "type": var_type,
                        "array_dims": array_dims,
                        "constraints": constraints,
                        "description": comment,
                    }
                )

        # Parse model block
        model_block = model_match.group(1)
        self._model_vars = []
        for line in model_block.strip().split("\n"):
            line = line.strip()
            # Line must have ~ in it
            if "~" in line:
                # Extract type, name, and comment if exists
                parts = line.split("~")
                self._model_vars.append(parts[0].strip())

    def _get_data_requirements_string(self) -> str:
        """Get the data requirements for this model as a formatted string."""
        lines = []
        lines.append(f"Data requirements for {self._stan_file.name}:")
        lines.append("-" * 50)

        # Group variables by their role
        index_vars = []
        dimension_vars = []
        data_vars = []
        data_vars_prefix = [
            "home_goals",
            "away_goals",
            "goal_diff",
            "home_team",
            "away_team",
        ]
        model_vars = []
        optional_vars = []

        for var in self._data_vars:
            if var["name"].endswith("_idx_match"):
                index_vars.append(var)
            elif var["name"] in ["N", "T"]:
                dimension_vars.append(var)
            elif var["name"].endswith("_match"):
                if any(prefix in var["name"] for prefix in data_vars_prefix):
                    data_vars.append(var)
                else:
                    model_vars.append(var)
            elif var["name"].endswith("sample_weights"):
                optional_vars.append(var)

        # Add usage
        lines.append("")
        lines.append("Usage:")
        lines.append("  model.fit(X, y, Z, weights)")
        lines.append("  model.predict(X, Z, point_spread)")
        lines.append("  model.predict_proba(X, Z, point_spread, include_draw, outcome)")
        lines.append("")

        # Add X requirements (team pairs)
        lines.append("X - Fixtures:")
        lines.append("  DataFrame with exactly 2 columns: [home_team, away_team]")
        for i, var in enumerate(index_vars):
            name = var["name"].replace("_idx_match", "").replace("_", " ").title()
            lines.append(f"  - Column {i}: {name} names (strings)")
        lines.append("")

        # Add y requirements (target variable)
        lines.append("y - Target:")
        target_descriptions = []
        for var in data_vars:
            name = var["name"].replace("_match", "")
            if "goal_diff" in name:
                target_descriptions.append("Goal differences (home - away)")
            elif "home_goals" in name or "away_goals" in name:
                target_descriptions.append("Home goals, Away goals (2 columns)")

        if target_descriptions:
            for desc in set(target_descriptions):  # Remove duplicates
                lines.append(f"  - {desc}")
        else:
            lines.append("  - Target variable as specified by model")
        lines.append("")

        # Determine model-specific requirements
        model_name = self._stan_file.stem.lower()
        is_decay = "decay" in model_name

        # Add Z requirements (additional match data)
        if model_vars:
            if is_decay:
                lines.append("Z - Additional Data:")
                lines.append("  DataFrame or array with temporal weighting data")
            else:
                lines.append("Z - Additional Data:")
                lines.append(
                    "  DataFrame or array with additional match-level features"
                )

            for i, var in enumerate(model_vars):
                name = var["name"].replace("_match", "")
                desc = var["description"] or f"{name.replace('_', ' ').title()}"
                type_str = "int" if var["type"] == "int" else "float"
                lines.append(f"  - Column {i}: {desc} ({type_str})")
            lines.append("")

        # Add weights requirements (sample weights)
        if optional_vars and not is_decay:
            lines.append("weights - sample_weights (optional):")
            lines.append("  Array of per-match weights for model fitting")
            for var in optional_vars:
                name = var["name"].replace("sample_weights", "")
                desc = var["description"] or f"{name.replace('_', ' ').title()}"
                lines.append(f"  - {desc} (float array, length = len(X))")
            lines.append("")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return detailed model information including data requirements."""
        return self._get_data_requirements_string()

    def help(self) -> None:
        """Display detailed usage information for this model."""
        print(self._get_data_requirements_string())

    @classmethod
    def create(cls, stan_file: str = "base") -> "BaseModel":
        """Create model instance and display data requirements.

        This is the recommended way to create models as it shows the data
        requirements immediately, helping users prepare the correct data format.

        Parameters
        ----------
        stan_file : str, default="base"
            Name of the Stan model file (without .stan extension)

        Returns:
        -------
        BaseModel
            Model instance with requirements displayed

        Examples:
        --------
        >>> model = PoissonDecay.create()
        Data requirements for poisson_decay.stan:
        ...
        >>> model.fit(X, y, Z)
        """
        # Create instance using normal constructor (silent)
        instance = cls(stan_file)

        # Display requirements after creation
        print(instance._get_data_requirements_string())

        return instance
