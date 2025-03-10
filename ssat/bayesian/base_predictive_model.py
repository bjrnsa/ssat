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

    def fit(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> "PredictiveModel":
        """Fit the model using MCMC sampling."""
        # Prepare data dictionary
        data_dict = self._data_dict(data, fit=True)

        # Compile model
        model = cmdstanpy.CmdStanModel(stan_file=self._stan_file)

        # Run sampling
        fit_result = model.sample(
            data=data_dict,
            iter_sampling=kwargs.get("draws", 8000),
            iter_warmup=kwargs.get("warmup", 2000),
            chains=kwargs.get("chains", 4),
            seed=kwargs.get("seed", 1),
        )

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
        data: Union[np.ndarray, pd.DataFrame],
        fit: bool = True,
    ) -> Dict[str, Any]:
        """Prepare data dictionary for Stan model dynamically based on Stan file requirements."""
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
        """Generate probability predictions for new data."""
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
