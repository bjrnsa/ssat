"""Bayesian Negative Binomial Hierarchical Model."""

from typing import Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ssat.bayesian.hierarchical_base import HierarchicalBaseModel


class NegBinomHierarchical(HierarchicalBaseModel):
    """Bayesian Negative Binomial Hierarchical Model."""

    def __init__(
        self,
        stem: str = "nbinom_hierachical",
    ):
        """Initialize the Negative Binomial Hierarchical model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "nbinom_hierachical".
        """
        super().__init__(stan_file=stem)

    def fit(
        self,
        X: pd.DataFrame,
        Z: pd.DataFrame,
        y: Optional[Union[NDArray[np.float64], pd.Series]] = None,
        weights: Optional[NDArray[np.float64]] = None,
        draws: int = 2000,
        warmup: int = 1000,
        chains: int = 4,
        validate: bool = False,
        plot_diagnostics: bool = False,
    ) -> "NegBinomHierarchical":
        """Fit the Bayesian Hierarchical Negative Binomial model.

        Parameters
        ----------
        X : pd.DataFrame
            Training data with team names.
        Z : pd.DataFrame
            Additional data, must include home and away goals.
        y : Optional[Union[NDArray[np.float64], pd.Series]], optional
            Target variable (not used in this model). Defaults to None.
        weights : Optional[NDArray[np.float64]], optional
            Match weights. Defaults to None.
        draws : int, optional
            Number of MCMC draws. Defaults to 2000.
        warmup : int, optional
            Warmup iterations for MCMC. Defaults to 1000.
        chains : int, optional
            Number of MCMC chains. Defaults to 4.

        Returns:
        -------
        NegBinomHierarchical
            Fitted model instance.
        """
        self._validate_X(X)
        self._validate_Z(X, Z)

        # Prepare data for Stan model
        data = self._data_dict(X, Z, weights, fit=True)

        # Compile and fit Stan model using base class method
        self.fit_result = self._compile_and_fit_stan_model(data, draws, warmup, chains)

        if validate:
            # Validate the MCMC diagnostics
            self._validate_mcmc_diagnostics()

        if plot_diagnostics:
            # Plot Trace plots
            self._plot_diagnostics()

        # Generate inference data for analysis
        self._generate_inference_data(data)

        return self
