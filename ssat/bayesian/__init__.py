"""This module contains Bayesian models for sports prediction and betting.

Available models:
- Poisson: Basic Poisson model for goal scoring
- NegBinom: Negative Binomial model for overdispersed scoring
- Skellam: Direct modeling of goal differences
- SkellamZero: Zero-inflated Skellam for matches with frequent draws
"""

from .nbinom import NegBinom
from .poisson import Poisson
from .skellam import Skellam
from .skellam_zero import SkellamZero

__all__ = [
    "Poisson",
    "NegBinom",
    "Skellam",
    "SkellamZero",
]
