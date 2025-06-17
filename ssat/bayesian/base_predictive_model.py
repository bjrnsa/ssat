"""Bayesian Model for Sports Match Prediction."""

from typing import Any, Dict, List, Optional, Union

import arviz as az
import cmdstanpy
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
    """Abstract base class for Bayesian predictive models that can predict matches."""

    kwargs = {
        "iter_sampling": 8000,
        "iter_warmup": 2000,
        "chains": 4,
        "seed": 1,
        "adapt_delta": 0.95,
        "max_treedepth": 12,
        "step_size": 0.5,
        "show_console": False,
        "parallel_chains": 10,
    }
    predictions: Optional[np.ndarray] = None

    def _get_model_inits(self) -> Optional[Dict[str, Any]]:
        """Get model inits for Stan model."""
        return None

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

    def fit(
        self,
        base_data: Union[np.ndarray, pd.DataFrame],
        model_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        optional_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs,
    ) -> "PredictiveModel":
        """Fit the model using MCMC sampling.

        Parameters
        ----------
        base_data : Union[np.ndarray, pd.DataFrame]
            Base data required by all models (e.g., team indices, scores)
        model_data : Optional[Union[np.ndarray, pd.DataFrame]], optional
            Additional model-specific data (e.g., weights, covariates)
        optional_data : Optional[Union[np.ndarray, pd.DataFrame]], optional
            Optional model-specific data (e.g., weights, covariates)
        **kwargs : dict
            Additional keyword arguments for sampling

        Returns:
        -------
        PredictiveModel
            The fitted model instance
        """
        # Prepare data dictionary
        data_dict = self._data_dict(base_data, model_data, optional_data, fit=True)

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

    def _data_dict(
        self,
        base_data: Union[np.ndarray, pd.DataFrame],
        model_data: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        optional_data: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        fit: bool = True,
    ) -> Dict[str, Any]:
        """Prepare data dictionary for Stan model dynamically based on Stan file requirements.

        Parameters
        ----------
        base_data : Union[np.ndarray, pd.DataFrame]
            Base data required by all models (e.g., team indices, scores)
        model_data : Optional[Union[np.ndarray, pd.DataFrame, pd.Series]], optional
            Additional model-specific data (e.g., weights, covariates)
        optional_data : Optional[Union[np.ndarray, pd.DataFrame, pd.Series]], optional
            Optional model-specific data (e.g., weights, covariates)
        fit : bool, optional
            Whether this is for fitting (True) or prediction (False)

        Returns:
        -------
        Dict[str, Any]
            Dictionary of data for Stan model
        """
        # Convert base_data to numpy array if DataFrame
        if isinstance(base_data, pd.DataFrame):
            base_array = base_data.to_numpy()
            self._match_ids = base_data.index.to_numpy()
        else:
            base_array = np.asarray(base_data)
            self._match_ids = np.arange(len(base_array))

        # Convert model_data to numpy array if provided
        model_array = None
        if model_data is not None:
            if isinstance(model_data, pd.Series):
                model_array = model_data.to_numpy().reshape(-1, 1)
            elif isinstance(model_data, pd.DataFrame):
                model_array = model_data.to_numpy()
            else:
                model_array = np.asarray(model_data)
                if model_array.ndim == 1:
                    model_array = model_array.reshape(-1, 1)

            # Validate shapes
            if len(model_array) != len(base_array):
                raise ValueError(
                    f"model_data length ({len(model_array)}) must match base_data length ({len(base_array)})"
                )

        # Convert optional_data to numpy array if provided
        optional_array = None
        if optional_data is not None:
            if isinstance(optional_data, pd.Series):
                optional_array = optional_data.to_numpy().reshape(-1, 1)
            elif isinstance(optional_data, pd.DataFrame):
                optional_array = optional_data.to_numpy()
        else:
            optional_array = np.ones(len(base_array)).reshape(-1, 1)

        # Initialize data dictionary with dimensions
        data_dict = {
            "N": len(base_array),
        }

        # Group variables by their role
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
            elif var["name"].endswith("_optional"):
                optional_vars.append(var)

        # Track current column index for base_data and model_data
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

        # Handle data columns from base_data
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

        # Handle weights and additional model-specific data
        for var in model_vars:
            model_var_name = var["name"]
            model_dtype = var["type"]
            if model_array is not None and model_col_idx < model_array.shape[1]:
                data_dict[model_var_name] = np.array(
                    model_array[:, model_col_idx], dtype=model_dtype
                )
                model_col_idx += 1

        # Handle optional data columns
        for var in optional_vars:
            optional_var_name = var["name"]
            optional_dtype = var["type"]
            if (
                optional_array is not None
                and optional_col_idx < optional_array.shape[1]
            ):
                data_dict[optional_var_name] = np.array(
                    optional_array[:, optional_col_idx], dtype=optional_dtype
                )
                optional_col_idx += 1

        return data_dict

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

    def predict(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        return_matches: bool = False,
        func: str = "median",
        sampling_method: Optional[str] = None,
    ) -> np.ndarray:
        """Generate predictions for new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fit before making predictions")

        data_dict = self._data_dict(data, fit=False)

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
        predictions = stan_predictions[-1]

        if return_matches:
            self.predictions = predictions.reshape(1, n_sims, n_matches)
            return self.predictions

        else:
            self.predictions = predictions.reshape(1, n_sims, n_matches)
            return self._format_predictions(
                data,
                getattr(np, func)(predictions, axis=0).T,
                col_names=[self.pred_vars[-1]],
            )

    def predict_proba(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        point_spread: float = 0.0,
        outcome: Optional[str] = None,
        func: str = "median",
        sampling_method: Optional[str] = None,
    ) -> np.ndarray:
        """Generate probability predictions for new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fit before making predictions")

        if outcome not in [None, "home", "away", "draw"]:
            raise ValueError("outcome must be None, 'home', 'away', or 'draw'")

        if self.predictions is None:
            # Get raw predictions and calculate goal differences
            predictions = self.predict(
                data, return_matches=True, sampling_method=sampling_method, func=func
            )
        else:
            predictions = self.predictions

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
            return self._format_predictions(data, home_probs, col_names=["home"])
        elif outcome == "away":
            return self._format_predictions(data, away_probs, col_names=["away"])
        elif outcome == "draw":
            return self._format_predictions(data, draw_probs, col_names=["draw"])

        # Return both probabilities
        return self._format_predictions(
            data,
            np.stack([home_probs, draw_probs, away_probs]).T,
            col_names=["home", "draw", "away"],
        )
