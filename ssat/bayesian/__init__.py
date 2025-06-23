"""This module contains Bayesian models for sports prediction and betting.

Available models:
- Poisson: Basic Poisson model for goal scoring
- NegBinom: Negative Binomial model for overdispersed scoring
- Skellam: Direct modeling of goal differences
- SkellamZero: Zero-inflated Skellam for matches with frequent draws
"""

from .nbinomial_models import NegBinom, NegBinomDecay
from .poisson_models import Poisson, PoissonDecay
from .skellam_models import Skellam, SkellamDecay, SkellamZero, SkellamZeroDecay

__all__ = [
    "Poisson",
    "PoissonDecay",
    "NegBinom",
    "NegBinomDecay",
    "Skellam",
    "SkellamDecay",
    "SkellamZero",
    "SkellamZeroDecay",
]
