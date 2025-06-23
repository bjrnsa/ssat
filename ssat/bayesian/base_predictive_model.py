"""Concrete Bayesian predictive model implementation for sports match prediction.

This module provides the PredictiveModel class, which implements the abstract BaseModel
interface with concrete Stan-based Bayesian functionality. It handles MCMC sampling,
inference data generation, and prediction methods for sports match modeling.

Classes
-------
TeamLabeller : az.labels.BaseLabeller
    Custom ArviZ labeller for team indices in plots
PredictiveModel : BaseModel
    Concrete implementation of Bayesian predictive models using Stan/cmdstanpy

The PredictiveModel provides a complete implementation for:
- MCMC fitting with configurable sampling parameters
- Data preprocessing for Stan models
- Inference data generation with ArviZ integration
- Standardized prediction and probability estimation methods
- Visualization utilities for model diagnostics and team statistics
"""

from typing import Any, Dict, List, Optional, Union

import arviz as az
import cmdstanpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ssat.bayesian.base_model import BaseModel


class TeamLabeller(az.labels.BaseLabeller):
    """Custom labeler for team indices."""

    def make_label_flat(self, var_name, sel, isel):
        """Generate flat label for team indices."""
        sel_str = self.sel_to_str(sel, isel)
        return sel_str


class PredictiveModel(BaseModel):
    """Concrete implementation of Bayesian predictive models using Stan/cmdstanpy.

    This class provides a complete implementation of the BaseModel interface,
    handling MCMC sampling, data preprocessing, inference data generation, and
    prediction methods. It serves as the foundation for all concrete model
    implementations in the SSAT framework.

    Key Features
    ------------
    - MCMC sampling with configurable parameters
    - Automatic Stan model compilation and fitting
    - ArviZ integration for Bayesian inference analysis
    - Standardized data preprocessing for X/y/Z/weights format
    - Prediction and probability estimation methods
    - Model diagnostics and visualization utilities

    Attributes:
    ----------
    kwargs : dict
        Default MCMC sampling parameters for Stan
    predictions : Optional[np.ndarray]
        Cached predictions from the last predict() call
    _fitted_with_Z : bool
        Flag indicating if model was fitted with additional features Z

    The class handles the complete workflow from data preparation through
    model fitting to prediction generation, providing a consistent interface
    for all Bayesian sports prediction models.
    """

    kwargs = {
        "iter_sampling": 4000,
        "iter_warmup": 1000,
        "chains": 2,
        "seed": 1,
        "adapt_delta": 0.95,
        "max_treedepth": 12,
        "step_size": 0.5,
        "show_console": False,
        "parallel_chains": 10,
    }
    predictions: Optional[np.ndarray] = None
    _fitted_with_Z: bool = False

    def _get_model_inits(self) -> Optional[Dict[str, Any]]:
        """Get model inits for Stan model."""
        return None

    def _format_predictions(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        predictions: np.ndarray,
        col_names: list[str],
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Format predictions with appropriate index and column names.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            Original input data used for prediction
        predictions : np.ndarray
            Raw predictions from the model
        col_names : list[str]
            Names for the prediction columns

        Returns:
        -------
        Union[np.ndarray, pd.DataFrame]
            Formatted predictions with meaningful index if input was DataFrame
        """
        if isinstance(data, pd.DataFrame):
            # Create meaningful fixture index if we have team names
            if data.shape[1] >= 2 and hasattr(data, "columns"):
                try:
                    # Try to create fixture names: "Home-Away"
                    fixture_index = data.apply(
                        lambda x: f"{x.iloc[0]}-{x.iloc[1]}", axis=1
                    )
                    return pd.DataFrame(
                        predictions, index=fixture_index, columns=col_names
                    )
                except Exception:
                    # Fallback to original match IDs if fixture creation fails
                    return pd.DataFrame(
                        predictions, index=self._match_ids, columns=col_names
                    )
            else:
                return pd.DataFrame(
                    predictions, index=self._match_ids, columns=col_names
                )
        else:
            return predictions

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        Z: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        weights: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs,
    ) -> "PredictiveModel":
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
        PredictiveModel
            The fitted model instance
        """
        # Track if model was fitted with Z data
        self._fitted_with_Z = Z is not None

        # Prepare data dictionary using standardized API
        data_dict = self._data_dict(X, y, Z, weights, fit=True)

        # Compile model
        model = cmdstanpy.CmdStanModel(stan_file=self._stan_file)

        inits = self._get_model_inits()

        # If update default kwargs
        for key, value in self.kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        # Run sampling
        fit_result = model.sample(data=data_dict, inits=inits, **kwargs)

        # Update model state
        self.is_fitted = True
        self.fit_result = fit_result
        self.model = model
        self.src_info = model.src_info()

        # Generate inference data
        self._generate_inference_data(data_dict)

        return self

    def _generate_inference_data(self, data: Dict[str, Any]) -> None:
        """Generate inference data from Stan fit result."""
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
        """Validate team existence in the model."""
        for team in teams:
            if team not in self._team_map:
                raise ValueError(f"Unknown team: {team}")

    def _check_is_fitted(self) -> None:
        """Check if model has been fitted."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")

    def _data_dict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        Z: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        weights: Optional[Union[np.ndarray, pd.Series]] = None,
        fit: bool = True,
    ) -> Dict[str, Any]:
        """Complete replacement for _data_dict using standardized X/y/Z/weights format.

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
        fit : bool, default=True
            Whether this is for fitting (True) or prediction (False)

        Returns:
        -------
        Dict[str, Any]
            Dictionary of data for Stan model
        """
        # Convert X to pandas DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=["home_team", "away_team"])
        else:
            X_df = X.copy()

        # Validate X has exactly 2 columns (team pairs)
        if X_df.shape[1] != 2:
            raise ValueError(
                "X must have exactly 2 columns for team pairs [home_team, away_team]"
            )

        # Handle target variable (y) - ALWAYS provided
        if isinstance(y, pd.Series):
            y_df = y.to_frame()
        elif isinstance(y, pd.DataFrame):
            y_df = y.copy()
        elif isinstance(y, np.ndarray):
            if y.ndim == 1:
                y_df = pd.DataFrame(y, columns=["target"], index=X_df.index)
            else:
                # Multiple columns (e.g., home_goals, away_goals)
                y_df = pd.DataFrame(y, index=X_df.index)
        else:
            y_df = pd.DataFrame(y, index=X_df.index)

        # Combine team pairs with target variable
        base_data = pd.concat([X_df, y_df], axis=1)

        # Convert base_data to numpy array
        base_array = base_data.to_numpy()
        self._match_ids = base_data.index.to_numpy()

        # Convert Z (additional match data) to numpy array if provided
        # Auto-fill Z=zeros for decay models in prediction mode
        model_array = None
        if Z is not None:
            if isinstance(Z, pd.DataFrame):
                model_array = Z.to_numpy()
            elif isinstance(Z, pd.Series):
                model_array = Z.to_numpy().reshape(-1, 1)
            elif isinstance(Z, np.ndarray):
                model_array = Z.copy()
                if model_array.ndim == 1:
                    model_array = model_array.reshape(-1, 1)
            else:
                model_array = np.asarray(Z)
                if model_array.ndim == 1:
                    model_array = model_array.reshape(-1, 1)

            # Validate shapes
            if len(model_array) != len(base_array):
                raise ValueError(
                    f"Z length ({len(model_array)}) must match X length ({len(base_array)})"
                )
        elif not fit and self._fitted_with_Z:
            # Auto-fill zeros for decay models during prediction
            model_array = np.zeros((len(base_array), 1), dtype=int)

        # Convert weights to numpy array if provided
        weights_array = None
        if weights is not None:
            if isinstance(weights, pd.Series):
                weights_array = weights.to_numpy().reshape(-1, 1)
            elif isinstance(weights, np.ndarray):
                weights_array = weights.copy().reshape(-1, 1)
            else:
                weights_array = np.asarray(weights).reshape(-1, 1)
        else:
            weights_array = np.ones(len(base_array)).reshape(-1, 1)

        # Initialize data dictionary with dimensions
        data_dict = {
            "N": len(base_array),
        }

        # Group variables by their role (same logic as original)
        index_vars = []
        dimension_vars = []
        data_vars = []
        data_vars_prefix = [
            "home_goals",
            "away_goals",
            "home_team",
            "away_team",
            "goal_diff",
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

        # Track current column index for base_data, Z, and weights
        base_col_idx = 0
        model_col_idx = 0
        optional_col_idx = 0

        # Handle index columns (e.g., team indices)
        if index_vars:
            # Get unique entities and create mapping
            index_cols = []
            for _ in index_vars:
                if base_col_idx >= base_array.shape[1]:
                    raise ValueError(
                        f"Not enough columns in base_data. Expected index column at position {base_col_idx}"
                    )
                index_cols.append(base_array[:, base_col_idx])
                base_col_idx += 1

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
                unknown = set(base_array[:, i]) - set(self._team_map.keys())
                if unknown:
                    raise ValueError(f"Unknown entities in column {i}: {unknown}")
                team_map = self._team_map

            data_dict[var["name"]] = np.array(
                [team_map[entity] for entity in base_array[:, i]]
            )

        # Handle data columns from base_data (target variables)
        for var in data_vars:
            data_dtype = var["type"]
            data_var_name = var["name"]
            if base_col_idx >= base_array.shape[1]:
                if not fit:
                    # For prediction, use zeros if column not provided
                    data_dict[data_var_name] = np.zeros(
                        len(base_array),
                        dtype=data_dtype,
                    )
                    continue
                else:
                    raise ValueError(
                        f"Not enough columns in base_data. Expected data column at position {base_col_idx}"
                    )

            # Convert to correct type
            data_dict[data_var_name] = np.array(
                base_array[:, base_col_idx], dtype=data_dtype
            )
            base_col_idx += 1

        # Handle Z (additional model-specific data)
        for var in model_vars:
            model_var_name = var["name"]
            model_dtype = var["type"]
            if model_array is not None and model_col_idx < model_array.shape[1]:
                data_dict[model_var_name] = np.array(
                    model_array[:, model_col_idx], dtype=model_dtype
                )
                model_col_idx += 1

        # Handle weights (optional data columns)
        for var in optional_vars:
            optional_var_name = var["name"]
            optional_dtype = var["type"]
            if weights_array is not None and optional_col_idx < weights_array.shape[1]:
                data_dict[optional_var_name] = np.array(
                    weights_array[:, optional_col_idx], dtype=optional_dtype
                )
                optional_col_idx += 1

        return data_dict

    @classmethod
    def create(cls) -> "PredictiveModel":
        """Create model instance and display data requirements.

        This is the recommended way to create models as it shows the data
        requirements immediately, helping users prepare the correct data format.

        Returns:
        -------
        PredictiveModel
            Model instance with requirements displayed

        Examples:
        --------
        >>> model = PoissonWeighted.create()
        Data requirements for poisson_decay.stan:
        ...
        >>> model.fit(X, y, Z)
        """
        # Create instance using normal constructor (silent)
        instance = cls()

        # Display requirements after creation
        print(instance._get_data_requirements_string())

        return instance

    def plot_trace(
        self,
        var_names: Optional[list[str]] = None,
    ) -> None:
        """Plot trace of the model.

        Parameters
        ----------
        var_names : Optional[list[str]], optional
            List of variable names to plot, by default None
            Keyword arguments passed to arviz.plot_trace
        """
        if var_names is None:
            var_names = self._model_vars

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
