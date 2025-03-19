"""Base Bayesian Model for Sports Match Prediction with Abstract Interface."""

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
        self._stan_file = Path("ssat/bayesian/stan_files") / f"{stan_file}.stan"
        if not self._stan_file.exists():
            raise ValueError(f"Stan file not found: {self._stan_file}")
        self.name = self._stan_file.stem
        # Parse Stan file and print data requirements
        self._parse_stan_file()
        self._print_data_requirements()

    @abstractmethod
    def fit(
        self,
        base_data: Union[np.ndarray, pd.DataFrame],
        model_data: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        optional_data: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        **kwargs,
    ) -> "BaseModel":
        """Fit the model using MCMC sampling.

        Parameters
        ----------
        base_data : Union[np.ndarray, pd.DataFrame]
            Base data required by all models (e.g., team indices, scores)
        model_data : Optional[Union[np.ndarray, pd.DataFrame, pd.Series]], optional
            Additional model-specific data (e.g., weights, covariates)
        optional_data : Optional[Union[np.ndarray, pd.DataFrame, pd.Series]], optional
            Optional model-specific data (e.g., weights, covariates)
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
        data: Union[np.ndarray, pd.DataFrame],
        return_matches: bool = False,
    ) -> np.ndarray:
        """Generate predictions for new data."""
        pass

    @abstractmethod
    def predict_proba(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        point_spread: float = 0.0,
        outcome: Optional[str] = None,
    ) -> np.ndarray:
        """Generate probability predictions for new data."""
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

    def _print_data_requirements(self) -> None:
        """Print the data requirements for this model."""
        print(f"\nData requirements for {self._stan_file.name}:")
        print("-" * 50)

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
            elif var["name"].endswith("_optional"):
                optional_vars.append(var)

        # Print base data requirements
        print("Base Data Requirements (required):")
        print("  These columns must be provided in base_data")
        base_col_idx = 0

        print("\n  Index columns (first columns):")
        for var in index_vars:
            name = var["name"].replace("_idx_match", "")
            constraints = var["constraints"] or ""
            desc = var["description"] or f"{name.replace('_', ' ').title()} index"
            print(f"    {base_col_idx}. {desc} {constraints}")
            base_col_idx += 1

        print("\n  Data columns:")
        for var in data_vars:
            name = var["name"].replace("_match", "")
            type_str = "int" if var["type"] == "int" else "float"
            desc = var["description"] or f"{name.replace('_', ' ').title()}"
            print(f"    {base_col_idx}. {desc} ({type_str})")
            base_col_idx += 1

        # Print model-specific data requirements if any
        if model_vars:
            print("\nModel-Specific Data Requirements:")
            print("  These columns can be provided in model_data")
            model_col_idx = 0

            for var in model_vars:
                name = var["name"].replace("_match", "")
                desc = var["description"] or "Sample weights"
                print(f"    {model_col_idx}. {desc} (float)")
                model_col_idx += 1

        # Print optional data requirements if any
        if optional_vars:
            print("\nOptional Data Requirements:")
            print("  These columns can be provided in model_data")
            optional_col_idx = 0

            for var in optional_vars:
                name = var["name"].replace("_optional", "")
                desc = var["description"] or f"{name.replace('_', ' ').title()}"
                print(f"    {optional_col_idx}. {desc} (float)")
                optional_col_idx += 1
