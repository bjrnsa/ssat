"""Bayesian Poisson Model for sports prediction."""

from typing import Optional

import arviz as az
import matplotlib.pyplot as plt

from ssat.bayesian.base_predictive_model import PredictiveModel, TeamLabeller


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


class NegBinomZero(Poisson):
    """Bayesian Negative Binomial Zero-inflated Model for predicting match scores.

    This model uses a negative binomial distribution to model goal scoring,
    accounting for both team attack and defense capabilities.
    """

    def __init__(
        self,
        stem: str = "nbinom_zero",
    ):
        """Initialize the Negative Binomial Zero-inflated model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "nbinom_zero".
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
