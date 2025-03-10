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

        # Parse Stan file and print data requirements
        self._parse_stan_file()
        self._print_data_requirements()

    @abstractmethod
    def fit(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> "BaseModel":
        """Fit the model using MCMC sampling."""
        pass

    @abstractmethod
    def _data_dict(
        self, data: Union[np.ndarray, pd.DataFrame], fit: bool = True
    ) -> Dict[str, Any]:
        """Prepare data dictionary for Stan model."""
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
