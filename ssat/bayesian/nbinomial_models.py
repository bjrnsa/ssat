# %%
"""Bayesian Poisson Model for sports prediction."""

from ssat.bayesian.poisson_models import Poisson


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


class NegBinomDecay(Poisson):
    """Bayesian Negative Binomial Model for predicting match scores.

    This model uses a negative binomial distribution to model goal scoring,
    accounting for both team attack and defense capabilities.
    """

    def __init__(
        self,
        stem: str = "nbinom_decay",
    ):
        """Initialize the Negative Binomial Weighted model.

        Parameters
        ----------
        stem : str, optional
            Stem name for the Stan model file.
            Defaults to "nbinom_decay".
        """
        super().__init__(stem=stem)
