"""This module provides functions for the Skellam distribution."""

import numpy as np

from ssat.stats.skellam_optim import qskellam


def skellam_rng_values(n, lambda1, lambda2):
    """Random generator for standard Skellam distribution using the quantile function.

    This function generates random deviates from the Skellam distribution by using
    the inverse transform sampling method with the quantile function.

    Parameters:
    -----------
    n : int
        Number of random values to generate
    lambda1 : float
        First Poisson parameter
    lambda2 : float
        Second Poisson parameter

    Returns:
    --------
    array-like
        Random deviates from the Skellam distribution
    """
    # Generate uniform random numbers between 0 and 1
    q = np.random.uniform(0, 1, size=n)

    # Use qskellam to transform to Skellam distribution
    return qskellam(q, lambda1, lambda2)


def zero_inflated_skellam_rng_values(n, lambda1, lambda2, p):
    """Random generator for zero-inflated Skellam distribution.

    This function generates random deviates from a zero-inflated Skellam distribution,
    which is a mixture of a point mass at zero (with probability p) and a standard
    Skellam distribution (with probability 1-p).

    Parameters:
    -----------
    n : int
        Number of random values to generate
    lambda1 : float
        First Poisson parameter
    lambda2 : float
        Second Poisson parameter
    p : float
        Probability of generating a zero (inflation parameter)

    Returns:
    --------
    array-like
        Random deviates from the zero-inflated Skellam distribution
    """
    # Initialize array with zeros
    samples = np.zeros(n)

    # Generate uniform random numbers to determine which values to replace
    mask = np.random.uniform(0, 1, size=n) > p

    # For values where mask is True, generate from standard Skellam
    if np.any(mask):
        samples[mask] = skellam_rng_values(np.sum(mask), lambda1, lambda2)

    return samples


def zero_inflated_skellam_rng_vec(n, lambda1, lambda2, p):
    """Vectorized random generator for zero-inflated Skellam distribution.

    This is a more efficient implementation of my_rzeroinflatedskellam that
    avoids generating separate random numbers for each sample.

    Parameters:
    -----------
    n : int
        Number of random values to generate
    lambda1 : float
        First Poisson parameter
    lambda2 : float
        Second Poisson parameter
    p : float
        Probability of generating a zero (inflation parameter)

    Returns:
    --------
    array-like
        Random deviates from the zero-inflated Skellam distribution
    """
    # Generate a mask for which values will be zeros
    mask = np.random.uniform(0, 1, size=n) > p

    # Count how many non-zero values we need
    count = np.sum(mask)

    # Initialize array with zeros
    samples = np.zeros(n)

    # Generate Skellam random values for non-zero positions
    if count > 0:
        # Generate uniform random numbers
        q = np.random.uniform(0, 1, size=count)

        # Transform to Skellam distribution
        skellam_values = qskellam(q, lambda1, lambda2)

        # Assign to non-zero positions
        samples[mask] = skellam_values

    return samples


if __name__ == "__main__":
    # Test cases
    print("\nTesting my_rskellam:")
    print(skellam_rng_values(5, 2, 1))

    print("\nTesting my_rzeroinflatedskellam:")
    print(zero_inflated_skellam_rng_values(10, 2, 1, 0.3))

    print("\nTesting my_rzeroinflatedskellam_vectorized:")
    print(zero_inflated_skellam_rng_vec(10, 2, 1, 0.3))

    # Compare distributions
    import matplotlib.pyplot as plt

    # Generate samples
    n_samples = 10000
    lambda1, lambda2 = 5, 3
    p_zero = 0.3

    # Standard Skellam
    std_samples = skellam_rng_values(n_samples, lambda1, lambda2)

    # Zero-inflated Skellam
    zi_samples = zero_inflated_skellam_rng_vec(n_samples, lambda1, lambda2, p_zero)

    # Plot histograms
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(
        std_samples,
        bins=range(min(std_samples.astype(int)) - 1, max(std_samples.astype(int)) + 2),
        alpha=0.7,
    )
    plt.title(f"Standard Skellam Distribution\n(λ1={lambda1}, λ2={lambda2})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(
        zi_samples,
        bins=range(min(zi_samples.astype(int)) - 1, max(zi_samples.astype(int)) + 2),
        alpha=0.7,
    )
    plt.title(
        f"Zero-Inflated Skellam Distribution\n(λ1={lambda1}, λ2={lambda2}, p={p_zero})"
    )
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("skellam_distributions.png")
    print("\nHistogram comparison saved as 'skellam_distributions.png'")
    print("\nHistogram comparison saved as 'skellam_distributions.png'")
    print("\nHistogram comparison saved as 'skellam_distributions.png'")
