"""This module provides functions for the Skellam distribution."""

from .skellam_optim import dskellam, pskellam, qskellam, rskellam
from .skellam_rng import (
    skellam_rng_values,
    zero_inflated_skellam_rng_values,
    zero_inflated_skellam_rng_vec,
)

__all__ = [
    "rskellam",
    "qskellam",
    "pskellam",
    "dskellam",
    "skellam_rng_values",
    "zero_inflated_skellam_rng_values",
    "zero_inflated_skellam_rng_vec",
]
