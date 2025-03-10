"""Base Bayesian Model for Sports Match Prediction with Abstract Interface.

This module provides a base class for Bayesian models used in sports match prediction.
It handles common functionality such as:
- MCMC sampling with Stan
- Result caching
- Data validation
- Diagnostic checks
- Parameter management
"""

import logging
import re
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import arviz as az
import cmdstanpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure cmdstanpy logging
logger = logging.getLogger("cmdstanpy")
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


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
        # Configuration
        self._stan_file = Path("ssat/bayesian/stan_files") / f"{stan_file}.stan"
        if not self._stan_file.exists():
            raise ValueError(f"Stan file not found: {self._stan_file}")

        # Parse Stan file and print data requirements
        self._parse_stan_file()
        self._print_data_requirements()

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        draws: int = 8000,
        warmup: int = 2000,
        chains: int = 4,
        seed: Optional[int] = 1,
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

        # Compile model
        model = cmdstanpy.CmdStanModel(stan_file=self._stan_file)

        # Run sampling
        fit_result = model.sample(
            data=data_dict,
            iter_sampling=draws,
            iter_warmup=warmup,
            chains=chains,
            seed=seed,
        )

        # Update model state
        self.is_fitted = True
        self.fit_result = fit_result
        self.n_chains = chains
        self.n_draws = draws
        self.n_samples = draws * chains
        self.model = model
        self.src_info = model.src_info()

        # Generate inference data
        self._generate_inference_data(data_dict)

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
        if not self.is_fitted:
            raise ValueError("Model must be fit before making predictions")

        data_dict = self._data_dict(data, fit=False)

        if self.model is None or self.fit_result is None:
            raise ValueError("Model not properly initialized")

        # Generate predictions using Stan model
        preds = self.model.generate_quantities(
            data=data_dict, previous_fit=self.fit_result
        )
        predictions = np.array(
            [preds.stan_variable(pred_var) for pred_var in self.pred_vars]
        )

        if return_matches:
            return predictions

        else:
            return self._format_predictions(
                data,
                np.median(predictions, axis=1).T,
                col_names=self.pred_vars,
            )

    def predict_proba(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        point_spread: float = 0.0,
        outcome: Optional[str] = None,
    ) -> np.ndarray:
        """Generate probability predictions for new data.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            Data array or DataFrame. If array, columns must be in order shown by _print_data_requirements.
            If DataFrame, column names don't matter but order of columns must match requirements.
        point_spread : float, optional
            Point spread adjustment (positive favors home team), by default 0.0
        outcome : Optional[str], optional
            Specific outcome to predict ('home', 'away', or 'draw'), by default None

        Returns:
        -------
        np.ndarray
            Predicted probabilities for each outcome

        Raises:
        ------
        ValueError
            If model is not fitted or if outcome is invalid
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before making predictions")

        if outcome not in [None, "home", "away", "draw"]:
            raise ValueError("outcome must be None, 'home', 'away', or 'draw'")

        # Get raw predictions and calculate goal differences
        predictions = self.predict(data, return_matches=True)

        # If predictions dimension n x 1, assume predictions are already goal differences
        if predictions.shape[0] == 1:
            goal_differences = predictions[0] + point_spread
        elif predictions.shape[0] == 2:
            goal_differences = predictions[0] - predictions[1] + point_spread
        else:
            raise ValueError("Invalid predictions shape")

        # Calculate home win probabilities directly
        home_probs = (goal_differences > 0).mean(axis=0)
        draw_probs = (goal_differences == 0).mean(axis=0)
        away_probs = (goal_differences < 0).mean(axis=0)

        # Handle specific outcome requests
        if outcome == "home":
            return self._format_predictions(data, home_probs)
        elif outcome == "away":
            return self._format_predictions(data, away_probs)
        elif outcome == "draw":
            return self._format_predictions(data, draw_probs)

        # Return both probabilities
        return self._format_predictions(
            data,
            np.stack([home_probs, draw_probs, away_probs]).T,
            col_names=["home", "draw", "away"],
        )

    def plot_trace(
        self,
        var_names: Optional[list[str]] = [
            "attack_team",
            "defence_team",
            "home_advantage",
        ],
    ) -> None:
        """Plot trace of the model.

        Parameters
        ----------
        var_names : Optional[list[str]], optional
            List of variable names to plot, by default None
            Keyword arguments passed to arviz.plot_trace
        """
        az.plot_trace(
            self.inference_data,
            var_names=var_names,
            compact=True,
            combined=True,
        )

        plt.tight_layout()
        plt.show()

    def plot_team_stats(self) -> None:
        """Plot team strength statistics."""
        ax = az.plot_forest(
            self.inference_data.posterior.attack_team
            - self.inference_data.posterior.defence_team,
            labeller=TeamLabeller(),
        )
        ax[0].set_title("Overall Team Strength")
        plt.tight_layout()
        plt.show()

    def _format_predictions(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        predictions: np.ndarray,
        col_names: list[str],
    ) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(predictions, index=self._match_ids, columns=col_names)
        else:
            return predictions

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
        if not self.is_fitted:
            raise ValueError("Model must be fit before generating inference data")

        if self.model is None or self.fit_result is None:
            raise ValueError("Model not properly initialized")

        # Get model structure information
        model_info = self.src_info

        # Extract variables by naming conventions
        self.pred_vars = [
            var
            for var in model_info.get("generated quantities", {}).keys()
            if var.startswith("pred_")
        ]

        log_likelihood = [
            var
            for var in model_info.get("generated quantities", {}).keys()
            if var.startswith("ll_")
        ]

        # Extract observed data (variables ending with _obs)
        observed_data = {}
        for var_name in data.keys():
            if var_name.endswith("_obs_match"):
                # Strip _obs_match suffix to get the base name
                base_name = var_name.replace("_obs_match", "")
                observed_data[base_name] = data[var_name]

        # All other data goes into constant_data
        constant_data = {k: v for k, v in data.items() if k not in observed_data}

        # Set up coordinates
        coords = {
            "match": self._match_ids,
            "team": self._entities,
        }

        # Automatically generate dimensions mapping
        dims = {}

        # Process all variables in the model
        for section in [
            "parameters",
            "transformed parameters",
            "generated quantities",
            "inputs",
        ]:
            for var_name, var_info in model_info.get(section, {}).items():
                if var_info["dimensions"] > 0:
                    # Assign dimensions based on suffix
                    if var_name.endswith("_team"):
                        dims[var_name] = ["team"]
                    elif var_name.endswith("_match"):
                        dims[var_name] = ["match"]
                    elif var_name.endswith("_idx_match"):
                        dims[var_name] = ["match"]

        # Create inference data
        self.inference_data = az.from_cmdstanpy(
            posterior=self.fit_result,
            observed_data=observed_data,
            constant_data=constant_data,
            coords=coords,
            dims=dims,
            posterior_predictive=self.pred_vars,
            log_likelihood=log_likelihood,
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
            self._match_ids = data.index.to_numpy()
        else:
            data_array = np.asarray(data)
            self._match_ids = np.arange(len(data_array))

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
            if fit:
                self._team_map = team_map
                self._n_teams = n_teams
                self._entities = teams
                data_dict["T"] = n_teams
            else:
                data_dict["T"] = self._n_teams

        # Create index arrays
        for i, var in enumerate(index_vars):
            if not fit:
                # Validate entities exist in mapping
                unknown = set(data_array[:, i]) - set(self._team_map.keys())
                if unknown:
                    raise ValueError(f"Unknown entities in column {i}: {unknown}")
                team_map = self._team_map

            data_dict[var["name"]] = np.array(
                [team_map[entity] for entity in data_array[:, i]]
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


class TeamLabeller(az.labels.BaseLabeller):
    """Custom labeler for team indices."""

    def make_label_flat(self, var_name, sel, isel):
        """Generate flat label for team indices."""
        sel_str = self.sel_to_str(sel, isel)
        return sel_str
