"""This module contains the abstract base class for predictive models used in sports betting analysis."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for predictive models."""

    def __init__(self):
        """Initialize the BaseModel with default attributes."""
        self.team_map_ = {}
        self.is_fitted_ = False

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[npt.NDArray[np.float64], pd.Series]] = None,
        Z: Optional[pd.DataFrame] = None,
        weights: Optional[npt.NDArray[np.float64]] = None,
    ) -> "BaseModel":
        """Fit the model to the training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data features.
        y : Optional[Union[npt.NDArray[np.float64], pd.Series]], default=None
            Training data target variable.
        Z : Optional[pd.DataFrame], default=None
            Additional training data.
        weights : Optional[npt.NDArray[np.float64]], default=None
            Sample weights.

        Returns:
        -------
        BaseModel
            Fitted model instance.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Predict outcomes based on the fitted model.

        Parameters
        ----------
        X : pd.DataFrame
            Data to predict on.

        Returns:
        -------
        npt.NDArray[np.float64]
            Predicted values.
        """
        pass

    @abstractmethod
    def predict_proba(
        self,
        X: pd.DataFrame,
        Z: Optional[pd.DataFrame] = None,
        point_spread: int = 0,
        include_draw: bool = True,
        outcome: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> npt.NDArray[np.float64]:
        """Predict match outcome probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            Data to predict on.
        Z : Optional[pd.DataFrame], default=None
            Additional data for prediction.
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
    def get_params(self) -> Dict:
        """Get the current parameters of the model.

        Returns:
        -------
        Dict
            Model parameters.
        """
        pass

    @abstractmethod
    def set_params(self, params: Dict) -> None:
        """Set parameters for the model.

        Parameters
        ----------
        params : Dict
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
            if team not in self.team_map_:
                raise ValueError(f"Unknown team: {team}")

    def _check_is_fitted(self) -> None:
        """Check if the model is fitted.

        Raises:
        ------
        ValueError
            If model has not been fitted
        """
        if not self.is_fitted_:
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
