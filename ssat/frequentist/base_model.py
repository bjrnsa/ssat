"""This module contains the abstract base class for predictive models used in sports betting analysis."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for predictive models."""

    def __init__(self):
        """Initialize the BaseModel with default attributes."""
        super().__init__()
        self.team_map = {}
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float64]],
        y: Union[npt.NDArray[np.float64], pd.Series, pd.DataFrame],
        Z: Optional[Union[pd.DataFrame, npt.NDArray[np.float64]]] = None,
        weights: Optional[Union[npt.NDArray[np.float64], pd.Series]] = None,
        **kwargs,
    ) -> "BaseModel":
        """Fit the model to the training data.

        Parameters
        ----------
        X : Union[pd.DataFrame, npt.NDArray[np.float64]]
            Team pairs data with exactly 2 columns: [home_team, away_team]
        y : Union[npt.NDArray[np.float64], pd.Series, pd.DataFrame]
            Target variable (REQUIRED)
        Z : Optional[Union[pd.DataFrame, npt.NDArray[np.float64]]], default=None
            Additional match-level features
        weights : Optional[Union[npt.NDArray[np.float64], pd.Series]], default=None
            Sample weights for each match. If None, uses equal weights.
        **kwargs : dict
            Additional keyword arguments

        Returns:
        -------
        BaseModel
            Fitted model instance.
        """
        pass

    @abstractmethod
    def predict(
        self, X: Union[pd.DataFrame, npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        """Predict outcomes based on the fitted model.

        Parameters
        ----------
        X : Union[pd.DataFrame, npt.NDArray[np.float64]]
            Team pairs data with exactly 2 columns: [home_team, away_team]

        Returns:
        -------
        npt.NDArray[np.float64]
            Predicted values.
        """
        pass

    @abstractmethod
    def predict_proba(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float64]],
        Z: Optional[Union[pd.DataFrame, npt.NDArray[np.float64]]] = None,
        point_spread: int = 0,
        include_draw: bool = True,
        outcome: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> npt.NDArray[np.float64]:
        """Predict match outcome probabilities.

        Parameters
        ----------
        X : Union[pd.DataFrame, npt.NDArray[np.float64]]
            Team pairs data with exactly 2 columns: [home_team, away_team]
        Z : Optional[Union[pd.DataFrame, npt.NDArray[np.float64]]], default=None
            Additional match-level features
        point_spread : float = 0.0
            Point spread adjustment.
        include_draw : bool, default=True
            Whether to include draw probability.
        outcome : Optional[str], default=None
            Specific outcome to predict.
        threshold : Optional[float], default=None
            Threshold for predicting draw outcome.

        Returns:
        -------
        npt.NDArray[np.float64]
            Predicted probabilities.
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get the current parameters of the model.

        Returns:
        -------
        Dict[str, Any]
            Model parameters.
        """
        pass

    @abstractmethod
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set parameters for the model.

        Parameters
        ----------
        params : Dict[str, Any]
            Model parameters.
        """
        pass

    @abstractmethod
    def get_team_ratings(self) -> pd.DataFrame:
        """Get team ratings as a DataFrame.

        Returns:
        -------
        pd.DataFrame
            Team ratings.
        """
        pass

    def _validate_X(self, X: pd.DataFrame, fit: bool = True) -> None:
        """Validate input DataFrame dimensions and types.

        Parameters
        ----------
        X : pd.DataFrame
            Input data with required columns:
            - First column: Home team names (string)
            - Second column: Away team names (string)
            If fit=True and y is None, must have third column with goal differences
        fit : bool, default=True
            Whether this is being called during fit (requires at least 2 columns)
            or during predict (requires exactly 2 columns)

        Raises:
        ------
        ValueError
            If input validation fails
        """
        # Check if X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        # Check minimum number of columns
        min_cols = 2
        if X.shape[1] < min_cols:
            raise ValueError(f"X must have at least {min_cols} columns")

        # For predict methods, exactly 2 columns are required
        if not fit and X.shape[1] != 2:
            raise ValueError("X must have exactly 2 columns for prediction")

        # Check that first two columns contain strings (team names)
        for i in range(2):
            if not pd.api.types.is_string_dtype(X.iloc[:, i]):
                raise ValueError(f"Column {i} must contain string values (team names)")

    def _validate_teams(self, teams: List[str]) -> None:
        """Validate teams exist in the model."""
        for team in teams:
            if team not in self.team_map:
                raise ValueError(f"Unknown team: {team}")

    def _check_is_fitted(self) -> None:
        """Check if the model is fitted.

        Raises:
        ------
        ValueError
            If model has not been fitted
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

    @staticmethod
    def _logit_transform(
        x: Union[float, npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        """Apply logistic transformation with numerical stability.

        Parameters
        ----------
        x : Union[float, npt.NDArray[np.float64]]
            Input value(s)

        Returns:
        -------
        npt.NDArray[np.float64]
            Transformed value(s)
        """
        x_array = np.asarray(x, dtype=np.float64)
        x_clipped = np.clip(x_array, -700, 700)  # exp(700) is close to float max
        return 1 / (1 + np.exp(-x_clipped))

    def _validate_Z(
        self, X: pd.DataFrame, Z: Optional[pd.DataFrame], require_goals: bool = False
    ) -> None:
        """Validate Z DataFrame dimensions and content.

        Parameters
        ----------
        X : pd.DataFrame
            Input data
        Z : Optional[pd.DataFrame]
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

    def _format_predictions(
        self,
        data: Union[npt.NDArray[np.float64], pd.DataFrame],
        predictions: npt.NDArray[np.float64],
        col_names: List[str],
    ) -> Union[npt.NDArray[np.float64], pd.DataFrame]:
        """Format predictions with appropriate index and column names.

        Parameters
        ----------
        data : Union[npt.NDArray[np.float64], pd.DataFrame]
            Original input data used for prediction
        predictions : npt.NDArray[np.float64]
            Raw predictions from the model
        col_names : List[str]
            Names for the prediction columns

        Returns:
        -------
        Union[npt.NDArray[np.float64], pd.DataFrame]
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
                    if predictions.ndim == 1:
                        return pd.Series(
                            predictions, index=fixture_index, name=col_names[0]
                        )
                    else:
                        return pd.DataFrame(
                            predictions, index=fixture_index, columns=col_names
                        )
                except Exception:
                    # Fallback to original index if fixture creation fails
                    if predictions.ndim == 1:
                        return pd.Series(
                            predictions, index=data.index, name=col_names[0]
                        )
                    else:
                        return pd.DataFrame(
                            predictions, index=data.index, columns=col_names
                        )
            else:
                if predictions.ndim == 1:
                    return pd.Series(predictions, index=data.index, name=col_names[0])
                else:
                    return pd.DataFrame(
                        predictions, index=data.index, columns=col_names
                    )
        else:
            return predictions

    def _extract_teams(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract home and away teams from input data.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Team pairs data with exactly 2 columns: [home_team, away_team]

        Returns:
        -------
        tuple[np.ndarray, np.ndarray]
            home_teams, away_teams as numpy arrays
        """
        if isinstance(X, np.ndarray):
            return X[:, 0], X[:, 1]
        else:
            return X.iloc[:, 0].to_numpy(), X.iloc[:, 1].to_numpy()

    def _get_team_indices(
        self, home_teams: np.ndarray, away_teams: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get team indices from team names with validation.

        Parameters
        ----------
        home_teams : np.ndarray
            Array of home team names
        away_teams : np.ndarray
            Array of away team names

        Returns:
        -------
        tuple[np.ndarray, np.ndarray]
            home_idx, away_idx as arrays of team indices

        Raises:
        ------
        ValueError
            If any team is not found in team_map
        """
        # Validate all teams exist
        all_teams = np.concatenate([home_teams, away_teams])
        self._validate_teams(np.unique(all_teams))

        # Map teams to indices efficiently
        home_idx_map = np.vectorize(self.team_map.get)
        away_idx_map = np.vectorize(self.team_map.get)

        return home_idx_map(home_teams), away_idx_map(away_teams)
