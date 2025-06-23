"""Bayesian Negative Binomial models for sports match prediction.

This module implements Negative Binomial-based models for predicting match outcomes
in sports. Negative Binomial models extend Poisson models by adding a dispersion
parameter to handle overdispersion in goal-scoring data, making them more robust
for datasets where goals show more variability than expected under a Poisson model.

Classes
-------
NegBinom : Poisson
    Standard Negative Binomial model inheriting Poisson prediction methods
NegBinomDecay : Poisson
    Negative Binomial model with temporal decay weighting

The Negative Binomial models use the same prediction interface as Poisson models
but with an additional dispersion parameter that allows for greater flexibility
in modeling goal distributions. They're particularly useful when the data shows
evidence of overdispersion (variance > mean) in goal counts.

Model Features:
- Handles overdispersion in goal-scoring data
- Same prediction capabilities as Poisson models
- Individual goal and goal difference prediction
- Support for temporal decay weighting (NegBinomDecay)
- Robust parameter estimation via MCMC sampling
"""

from ssat.bayesian.poisson_models import Poisson


class NegBinom(Poisson):
    """Bayesian Negative Binomial model for predicting match scores.

    Extends the Poisson model by adding a dispersion parameter to handle
    overdispersion in goal-scoring data. Uses negative binomial distributions
    for home and away goals, which can better capture the variability often
    observed in real sports data.

    Model Structure
    ---------------
    - Home goals ~ NegBinom(μ_home, φ)
    - Away goals ~ NegBinom(μ_away, φ)
    - μ_home = exp(home_advantage + attack_home - defense_away)
    - μ_away = exp(attack_away - defense_home)
    - φ ~ prior distribution (dispersion parameter)

    The negative binomial parameterization allows for variance > mean,
    making it more flexible than the Poisson model for overdispersed data.

    Parameters
    ----------
    stem : str, default="nbinom"
        Name of the Stan model file (without .stan extension)

    Inheritance
    -----------
    Inherits all prediction methods from the Poisson class, providing
    the same interface for predict(), predict_proba(), and simulate_matches().

    Examples:
    --------
    >>> model = NegBinom()
    >>> model.fit(X_train, y_train)  # Same interface as Poisson
    >>> predictions = model.predict(X_test)
    >>> probabilities = model.predict_proba(X_test)
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
    """Bayesian Negative Binomial model with temporal decay weighting.

    Combines the overdispersion handling of negative binomial distributions
    with temporal decay weighting to emphasize recent matches. This model
    is particularly useful for sports data with both overdispersion and
    time-varying team strengths.

    Model Structure
    ---------------
    Same as NegBinom model but with temporal weights applied:
    - Each match weighted by exp(-decay_rate * days_since_match)
    - Negative binomial distributions for goal counts
    - Recent matches have higher influence on parameter estimation

    Parameters
    ----------
    stem : str, default="nbinom_decay"
        Name of the Stan model file (without .stan extension)

    Additional Requirements
    ----------------------
    Z : array-like
        Temporal decay weights or days since each match for training data
        Required for fitting, optional for prediction (defaults to 0)

    Inheritance
    -----------
    Inherits all prediction methods from the Poisson class.

    Examples:
    --------
    >>> model = NegBinomDecay()
    >>> days_since = (max_date - match_dates).dt.days
    >>> model.fit(X_train, y_train, Z=days_since)
    >>> predictions = model.predict(X_test)
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
