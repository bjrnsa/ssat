"""Bayesian Poisson Model for sports prediction."""

from typing import Optional, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bernoulli

from ssat.bayesian.base_predictive_model import PredictiveModel, TeamLabeller
from ssat.stats.skellam_optim import qskellam


class Poisson(PredictiveModel):
    """Bayesian Poisson Model for predicting match scores.

    This model uses a Poisson distribution to model goal scoring,
    accounting for both team attack and defense capabilities.
    """

    def __init__(
        self,
        stem: str = "poisson",
    ):
        """Initialize the Poisson model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "poisson".
        """
        super().__init__(stan_file=stem)

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


class PoissonWeighted(Poisson):
    """Bayesian Poisson Model for predicting match scores.

    This model uses a Poisson distribution to model goal scoring,
    accounting for both team attack and defense capabilities.
    """

    def __init__(
        self,
        stem: str = "poisson_weighted",
    ):
        """Initialize the Poisson Weighted model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "poisson_weighted".
        """
        super().__init__(stem=stem)


class NegBinom(Poisson):
    """Bayesian Negative Binomial Model for predicting match scores.

    This model uses a negative binomial distribution to model goal scoring,
    accounting for both team attack and defense capabilities.
    """

    def __init__(
        self,
        stem: str = "nbinom",
    ):
        """Initialize the Negative Binomial model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "nbinom".
        """
        super().__init__(stem=stem)


class NegBinomWeighted(Poisson):
    """Bayesian Negative Binomial Model for predicting match scores.

    This model uses a negative binomial distribution to model goal scoring,
    accounting for both team attack and defense capabilities.
    """

    def __init__(
        self,
        stem: str = "nbinom_weighted",
    ):
        """Initialize the Negative Binomial Weighted model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "nbinom_weighted".
        """
        super().__init__(stem=stem)


class Skellam(Poisson):
    """Bayesian Skellam Model for predicting match scores.

    This model uses a Skellam distribution (difference of two Poisson distributions)
    to directly model the goal difference between teams, accounting for both team
    attack and defense capabilities.
    """

    def __init__(
        self,
        stem: str = "skellam",
    ):
        """Initialize the Skellam model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "skellam".
        """
        super().__init__(stem=stem)

    def predict(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        return_matches: bool = False,
        func: str = "mean",
        sampling_method: str = "qskellam",
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

        if sampling_method == "qskellam":
            predictions = qskellam(
                np.random.uniform(0, 1, size=(n_sims, n_matches)),
                stan_predictions[1],
                stan_predictions[2],
            )

        elif sampling_method == "rskellam":
            predictions = stan_predictions[0]
        else:
            raise ValueError("Only qskellam and rskellam are supported")

        if return_matches:
            self.predictions = predictions.reshape(1, n_sims, n_matches)
            return self.predictions

        else:
            self.predictions = predictions.reshape(1, n_sims, n_matches)
            return self._format_predictions(
                data,
                getattr(np, func)(predictions, axis=0).T,
                col_names=[self.pred_vars[0]],
            )


class SkellamDweibull(Skellam):
    """Bayesian Skellam Model for predicting match scores.

    This model uses a Skellam distribution (difference of two Poisson distributions)
    to directly model the goal difference between teams, accounting for both team
    attack and defense capabilities.
    """

    def __init__(
        self,
        stem: str = "skellam_dweibull",
    ):
        """Initialize the Skellam model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "skellam_dweibull".
        """
        super().__init__(stem=stem)


class SkellamWeighted(Skellam):
    """Bayesian Skellam Model for predicting match scores.

    This model uses a Skellam distribution (difference of two Poisson distributions)
    to directly model the goal difference between teams, accounting for both team
    attack and defense capabilities.
    """

    def __init__(
        self,
        stem: str = "skellam_weighted",
    ):
        """Initialize the Skellam model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "skellam_weighted   ".
        """
        super().__init__(stem=stem)


class SkellamZero(Poisson):
    """Bayesian Zero-inflated Skellam Model for predicting match scores.

    This model uses a zero-inflated Skellam distribution to model goal differences,
    particularly suitable for low-scoring matches or competitions with frequent draws.
    The zero-inflation component explicitly models the probability of a draw.
    """

    def __init__(
        self,
        stem: str = "skellam_zero",
    ):
        """Initialize the Zero-inflated Skellam model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "skellam_zero".
        """
        super().__init__(stem=stem)

    def predict(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        return_matches: bool = False,
        func: str = "mean",
        sampling_method: str = "qskellam",
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
        p = 1 - preds.stan_variable("zi")
        rvs = bernoulli.rvs(p).reshape(-1, 1)

        if sampling_method == "qskellam":
            predictions = qskellam(
                np.random.uniform(0, 1, size=(n_sims, n_matches)),
                stan_predictions[1],
                stan_predictions[2],
            )
            predictions = predictions * rvs
        elif sampling_method == "rskellam":
            predictions = stan_predictions[0]
            predictions = predictions * rvs
        else:
            raise ValueError("Only qskellam and rskellam are supported")

        if return_matches:
            self.predictions = predictions.reshape(1, n_sims, n_matches)
            return self.predictions

        else:
            self.predictions = predictions.reshape(1, n_sims, n_matches)
            return self._format_predictions(
                data,
                getattr(np, func)(predictions, axis=0).T,
                col_names=[self.pred_vars[0]],
            )


class SkellamZeroWeighted(SkellamZero):
    """Bayesian Zero-inflated Skellam Model for predicting match scores.

    This model uses a zero-inflated Skellam distribution to model goal differences,
    particularly suitable for low-scoring matches or competitions with frequent draws.
    The zero-inflation component explicitly models the probability of a draw.
    """

    def __init__(
        self,
        stem: str = "skellam_zero_weighted",
    ):
        """Initialize the Zero-inflated Skellam Weighted model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "skellam_zero_weighted".
        """
        super().__init__(stem=stem)
