"""This module provides optimized functions for the Skellam distribution."""

import warnings

import numpy as np
from scipy import special, stats


def dskellam(x, lambda1, lambda2=None, log=False):
    """Density function for the Skellam distribution (difference of Poissons).

    Parameters:
    -----------
    x : float or array-like
        Quantiles
    lambda1 : float or array-like
        First Poisson parameter
    lambda2 : float or array-like, optional
        Second Poisson parameter, defaults to lambda1 if not provided
    log : bool, default=False
        If True, probabilities are given as log(p)

    Returns:
    --------
    float or array-like
        Probability density at x
    """
    # Check required arguments
    if x is None or lambda1 is None:
        raise ValueError("first 2 arguments are required")

    # Set default for lambda2 if not provided
    if lambda2 is None:
        lambda2 = lambda1

    # Convert inputs to numpy arrays
    x = np.asarray(x)
    lambda1 = np.asarray(lambda1)
    lambda2 = np.asarray(lambda2)

    # Check for invalid lambdas
    invalid_lambda1 = ~(np.isfinite(lambda1) & (lambda1 >= 0))
    invalid_lambda2 = ~(np.isfinite(lambda2) & (lambda2 >= 0))

    if np.any(invalid_lambda1) or np.any(invalid_lambda2):
        warnings.warn("NaNs produced")

    # Broadcast arrays to compatible shapes
    # This automatically handles scalar and array inputs
    x, lambda1, lambda2 = np.broadcast_arrays(x, lambda1, lambda2)

    # Initialize return array
    result = np.zeros_like(x, dtype=float)

    # Create masks for different cases
    invalid = invalid_lambda1 | invalid_lambda2
    zero_lambda1 = (lambda1 == 0) & ~invalid
    zero_lambda2 = (lambda2 == 0) & ~invalid
    general_case = ~invalid & ~zero_lambda1 & ~zero_lambda2

    # Set invalid results to NaN
    result[invalid] = np.nan

    # When lambda1 = 0, it's a Poisson with mean lambda2 (negated)
    if np.any(zero_lambda1):
        neg_x = -x[zero_lambda1]
        valid_x = (neg_x >= 0) & (neg_x == np.floor(neg_x))
        result[zero_lambda1] = np.where(
            valid_x, stats.poisson.pmf(neg_x, lambda2[zero_lambda1]), 0
        )

    # When lambda2 = 0, it's a Poisson with mean lambda1
    if np.any(zero_lambda2):
        valid_x = (x[zero_lambda2] >= 0) & (
            x[zero_lambda2] == np.floor(x[zero_lambda2])
        )
        result[zero_lambda2] = np.where(
            valid_x, stats.poisson.pmf(x[zero_lambda2], lambda1[zero_lambda2]), 0
        )

    # Handle general case
    if np.any(general_case):
        # Using the modified Bessel function of the first kind
        abs_x = np.abs(x[general_case])
        sqrt_prod = 2 * np.sqrt(lambda1[general_case] * lambda2[general_case])
        power_term = np.power(
            lambda1[general_case] / lambda2[general_case], x[general_case] / 2
        )
        exp_term = np.exp(-(lambda1[general_case] + lambda2[general_case]))

        # Calculate using the modified Bessel function with exponential scaling
        bessel_term = special.ive(abs_x, sqrt_prod)
        result[general_case] = bessel_term * power_term * exp_term * np.exp(sqrt_prod)

    # Apply log if requested
    if log:
        # Handle zeros before taking log
        zero_density = result == 0
        result[~zero_density] = np.log(result[~zero_density])
        result[zero_density] = -np.inf

    # Return scalar if input was scalar
    return (
        result.item()
        if (
            np.isscalar(x)
            and np.isscalar(lambda1)
            and (lambda2 is None or np.isscalar(lambda2))
        )
        else result
    )


def pskellam_sp(q, lambda1, lambda2, lower_tail=True, log_p=False):
    """Saddlepoint approximation for the Skellam CDF.

    Parameters:
    -----------
    q : float or array-like
        Quantiles
    lambda1 : float or array-like
        First Poisson parameter
    lambda2 : float or array-like
        Second Poisson parameter
    lower_tail : bool, default=True
        If True, probabilities are P[X <= x], otherwise P[X > x]
    log_p : bool, default=False
        If True, probabilities are given as log(p)

    Returns:
    --------
    float or array-like
        Approximate CDF values
    """
    # Convert inputs to numpy arrays and broadcast to compatible shapes
    q, lambda1, lambda2 = np.broadcast_arrays(
        np.asarray(q), np.asarray(lambda1), np.asarray(lambda2)
    )

    # Calculate parameters for the saddlepoint approximation
    mu = lambda1 - lambda2
    sigma = np.sqrt(lambda1 + lambda2)

    # Standardized deviate
    z = (q - mu) / sigma

    # Skewness correction
    gamma1 = (lambda1 - lambda2) / np.power(lambda1 + lambda2, 1.5)

    # Kurtosis correction
    gamma2 = 1 / (lambda1 + lambda2)

    # Cornish-Fisher expansion for better normal approximation
    w = (
        z
        + gamma1 * (z**2 - 1) / 6
        + gamma2 * (z**3 - 3 * z) / 24
        - gamma1**2 * (2 * z**3 - 5 * z) / 36
    )

    # Calculate CDF using normal approximation with corrections
    result = stats.norm.cdf(w) if lower_tail else stats.norm.sf(w)

    # Apply log if requested
    if log_p:
        # Handle zeros before taking log
        zero_prob = result == 0
        result[~zero_prob] = np.log(result[~zero_prob])
        result[zero_prob] = -np.inf

    # Return scalar if input was scalar
    return (
        result.item()
        if (np.isscalar(q) and np.isscalar(lambda1) and np.isscalar(lambda2))
        else result
    )


def pskellam(q, lambda1, lambda2=None, lower_tail=True, log_p=False):
    """CDF of Skellam distribution (difference of Poissons).

    Parameters:
    -----------
    q : float or array-like
        Quantiles
    lambda1 : float or array-like
        First Poisson parameter
    lambda2 : float or array-like, optional
        Second Poisson parameter, defaults to lambda1 if not provided
    lower_tail : bool, default=True
        If True, probabilities are P[X <= x], otherwise P[X > x]
    log_p : bool, default=False
        If True, probabilities are given as log(p)

    Returns:
    --------
    float or array-like
        Probabilities corresponding to the specified quantiles
    """
    # Check required arguments
    if q is None or lambda1 is None:
        raise ValueError("first 2 arguments are required")

    # Set default for lambda2 if not provided
    if lambda2 is None:
        lambda2 = lambda1

    # Convert inputs to numpy arrays
    q = np.asarray(q)
    lambda1 = np.asarray(lambda1)
    lambda2 = np.asarray(lambda2)

    # Check for invalid lambdas
    invalid_lambda1 = ~(np.isfinite(lambda1) & (lambda1 >= 0))
    invalid_lambda2 = ~(np.isfinite(lambda2) & (lambda2 >= 0))

    if np.any(invalid_lambda1) or np.any(invalid_lambda2):
        warnings.warn("NaNs produced")

        # Set invalid lambdas to NaN
        lambda1 = np.where(invalid_lambda1, np.nan, lambda1)
        lambda2 = np.where(invalid_lambda2, np.nan, lambda2)

    # CDF is a step function, so convert to integer values without warning
    x = np.floor(q)

    # Broadcast arrays to compatible shapes
    x, lambda1, lambda2 = np.broadcast_arrays(x, lambda1, lambda2)

    # Different formulas for negative & nonnegative x (zero lambda is OK)
    neg = (x < 0) & (~np.isnan(lambda1)) & (~np.isnan(lambda2))
    pos = (x >= 0) & (~np.isnan(lambda1)) & (~np.isnan(lambda2))

    # Initialize return array with NaN
    result = np.full_like(x, np.nan, dtype=float)

    # Calculate CDF values
    if np.any(neg):
        if lower_tail:
            # P[X <= x] for x < 0
            result[neg] = stats.ncx2.cdf(
                2 * lambda2[neg], df=2 * lambda1[neg], nc=-2 * x[neg]
            )
        else:
            # P[X > x] for x < 0
            result[neg] = stats.ncx2.sf(
                2 * lambda2[neg], df=2 * lambda1[neg], nc=-2 * x[neg]
            )

    if np.any(pos):
        if lower_tail:
            # P[X <= x] for x >= 0
            result[pos] = stats.ncx2.sf(
                2 * lambda1[pos], df=2 * lambda2[pos], nc=2 * (x[pos] + 1)
            )
        else:
            # P[X > x] for x >= 0
            result[pos] = stats.ncx2.cdf(
                2 * lambda1[pos], df=2 * lambda2[pos], nc=2 * (x[pos] + 1)
            )

    # Use saddlepoint approximation if outside the working range of ncx2
    chk = (neg | pos) & (~np.isfinite(result) | (~log_p & (result < 1e-308)))
    if np.any(chk):
        result[chk] = pskellam_sp(x[chk], lambda1[chk], lambda2[chk], lower_tail, log_p)

    # Apply log if requested
    if log_p:
        # Handle zeros before taking log
        zero_prob = result == 0
        result[~zero_prob] = np.log(result[~zero_prob])
        result[zero_prob] = -np.inf

    # Return scalar if input was scalar
    return (
        result.item()
        if (
            np.isscalar(q)
            and np.isscalar(lambda1)
            and (lambda2 is None or np.isscalar(lambda2))
        )
        else result
    )


def qskellam(p, lambda1, lambda2=None, lower_tail=True, log_p=False):
    """Quantile function for the Skellam distribution (difference of Poissons).

    Parameters:
    -----------
    p : float or array-like
        Probabilities
    lambda1 : float or array-like
        First Poisson parameter
    lambda2 : float or array-like, optional
        Second Poisson parameter, defaults to lambda1 if not provided
    lower_tail : bool, default=True
        If True, probabilities are P[X <= x], otherwise P[X > x]
    log_p : bool, default=False
        If True, probabilities are given as log(p)

    Returns:
    --------
    float or array-like
        Quantiles corresponding to the specified probabilities
    """
    # Check required arguments
    if p is None or lambda1 is None:
        raise ValueError("first 2 arguments are required")

    # Set default for lambda2 if not provided
    if lambda2 is None:
        lambda2 = lambda1

    # Convert inputs to numpy arrays
    p = np.asarray(p)
    lambda1 = np.asarray(lambda1)
    lambda2 = np.asarray(lambda2)

    # Broadcast arrays to compatible shapes
    p, lambda1, lambda2 = np.broadcast_arrays(p, lambda1, lambda2)

    # Initialize return array with NaN
    result = np.full_like(p, np.nan, dtype=float)

    # Handle zero lambda separately (quicker than interpreted search)
    verbose = False  # Equivalent to R's getOption("verbose")

    if verbose:
        nz = np.ones_like(p, dtype=bool)  # verify search by using it for Poisson too
    else:
        # Handle cases where lambda2 is 0
        zero_lambda2 = lambda2 == 0
        if np.any(zero_lambda2):
            if log_p:
                p_adj = np.exp(p[zero_lambda2])
            else:
                p_adj = p[zero_lambda2]

            if not lower_tail:
                p_adj = 1 - p_adj

            result[zero_lambda2] = stats.poisson.ppf(p_adj, lambda1[zero_lambda2])

        # Handle cases where lambda1 is 0
        zero_lambda1 = lambda1 == 0
        if np.any(zero_lambda1):
            if log_p:
                p_adj = np.exp(p[zero_lambda1])
            else:
                p_adj = p[zero_lambda1]

            if lower_tail:
                p_adj = 1 - p_adj

            result[zero_lambda1] = -stats.poisson.ppf(p_adj, lambda2[zero_lambda1])

        nz = (lambda1 != 0) & (lambda2 != 0)

    # Handle boundaries correctly
    eps = np.finfo(float).eps
    bdry = nz & ((p == 0) | (p + 1.01 * eps >= 1))

    if np.any(bdry):
        p_is_zero = p[bdry] == 0
        p_is_one = ~p_is_zero

        if lower_tail:
            # For p=0
            if np.any(p_is_zero):
                idx = bdry.copy()
                idx[bdry] = p_is_zero
                result[idx] = np.where(lambda2[idx] == 0, 0, -np.inf)

            # For p=1
            if np.any(p_is_one):
                idx = bdry.copy()
                idx[bdry] = p_is_one
                result[idx] = np.where(lambda1[idx] == 0, 0, np.inf)
        else:
            # For p=1 (when lower_tail=False, p=1 means p=0 in lower_tail=True)
            if np.any(p_is_zero):
                idx = bdry.copy()
                idx[bdry] = p_is_zero
                result[idx] = np.where(lambda1[idx] == 0, 0, np.inf)

            # For p=0 (when lower_tail=False, p=0 means p=1 in lower_tail=True)
            if np.any(p_is_one):
                idx = bdry.copy()
                idx[bdry] = p_is_one
                result[idx] = np.where(lambda2[idx] == 0, 0, -np.inf)

    # Process non-boundary, non-zero lambda cases
    remaining = nz & ~bdry

    if np.any(remaining):
        # Extract the values we need to process
        p_subset = p[remaining]
        lambda1_subset = lambda1[remaining]
        lambda2_subset = lambda2[remaining]

        # Cornish-Fisher approximations
        if log_p:
            z = stats.norm.ppf(np.exp(p_subset) if lower_tail else 1 - np.exp(p_subset))
        else:
            z = stats.norm.ppf(p_subset if lower_tail else 1 - p_subset)

        mu = lambda1_subset - lambda2_subset
        vr = lambda1_subset + lambda2_subset
        sg = np.sqrt(vr)

        # First-order approximation
        c0 = mu + z * sg

        # Second-order correction (skewness)
        gamma1 = (lambda1_subset - lambda2_subset) / np.power(
            lambda1_subset + lambda2_subset, 1.5
        )
        c1 = (z**2 - 1) * gamma1 * sg / 6

        # Third-order correction (kurtosis)
        gamma2 = 1 / (lambda1_subset + lambda2_subset)
        c2 = (
            -(c1 * mu / sg - 2 * lambda1_subset * lambda2_subset * (z**2 - 3) / vr)
            * z
            / 12
            / vr
            / sg
        ) + (gamma2 * z**2)  # Incorporate gamma2 into the correction

        # Test and linear search (slow if p extreme or lambda1+lambda2 small)
        q0 = np.round(c0 + c1 + c2).astype(int)
        p0 = pskellam(
            q0, lambda1_subset, lambda2_subset, lower_tail=lower_tail, log_p=log_p
        )

        if log_p:
            p_adj = p_subset
        else:
            p_adj = p_subset * (1 - 64 * eps)

        if lower_tail:  # smallest x such that F(x) >= p
            up = up1 = p0 < p_adj
            while np.any(up1):
                q0[up1] += 1
                up1_indices = np.where(up1)[0]
                p_new = pskellam(
                    q0[up1],
                    lambda1_subset[up1],
                    lambda2_subset[up1],
                    lower_tail=lower_tail,
                    log_p=log_p,
                )
                up1[up1_indices] = p_new < p_adj[up1]

            down1 = (~up) & (p0 > p_adj)
            if np.any(down1):
                down1_indices = np.where(down1)[0]
                p_new = pskellam(
                    q0[down1] - 1,
                    lambda1_subset[down1],
                    lambda2_subset[down1],
                    lower_tail=lower_tail,
                    log_p=log_p,
                )
                down1[down1_indices] = p_new > p_adj[down1]

            while np.any(down1):
                q0[down1] -= 1
                down1_indices = np.where(down1)[0]
                p_new = pskellam(
                    q0[down1] - 1,
                    lambda1_subset[down1],
                    lambda2_subset[down1],
                    lower_tail=lower_tail,
                    log_p=log_p,
                )
                down1[down1_indices] = p_new > p_adj[down1]
        else:  # largest x such that F(x,lower_tail=FALSE) <= p
            down = down1 = p0 > p_adj
            while np.any(down1):
                q0[down1] += 1
                down1_indices = np.where(down1)[0]
                p_new = pskellam(
                    q0[down1],
                    lambda1_subset[down1],
                    lambda2_subset[down1],
                    lower_tail=lower_tail,
                    log_p=log_p,
                )
                down1[down1_indices] = p_new > p_adj[down1]

            up1 = (~down) & (p0 < p_adj)
            if np.any(up1):
                up1_indices = np.where(up1)[0]
                p_new = pskellam(
                    q0[up1] - 1,
                    lambda1_subset[up1],
                    lambda2_subset[up1],
                    lower_tail=lower_tail,
                    log_p=log_p,
                )
                up1[up1_indices] = ~(p_new > p_adj[up1])

            while np.any(up1):
                q0[up1] -= 1
                up1_indices = np.where(up1)[0]
                p_new = pskellam(
                    q0[up1] - 1,
                    lambda1_subset[up1],
                    lambda2_subset[up1],
                    lower_tail=lower_tail,
                    log_p=log_p,
                )
                up1[up1_indices] = ~(p_new > p_adj[up1])

        result[remaining] = q0

    # Return scalar if input was scalar
    return (
        result.item()
        if (
            np.isscalar(p)
            and np.isscalar(lambda1)
            and (lambda2 is None or np.isscalar(lambda2))
        )
        else result
    )


def rskellam(n, lambda1, lambda2=None):
    """Generate random deviates from the Skellam distribution (difference of Poissons).

    Parameters:
    -----------
    n : int
        Number of random values to generate
    lambda1 : float or array-like
        First Poisson parameter
    lambda2 : float or array-like, optional
        Second Poisson parameter, defaults to lambda1 if not provided

    Returns:
    --------
    array-like
        Random deviates from the Skellam distribution
    """
    # Set default for lambda2 if not provided
    if lambda2 is None:
        lambda2 = lambda1

    # Convert inputs to numpy arrays
    lambda1 = np.asarray(lambda1)
    lambda2 = np.asarray(lambda2)

    # Generate random Poisson variables and take their difference
    return stats.poisson.rvs(lambda1, size=n) - stats.poisson.rvs(lambda2, size=n)


if __name__ == "__main__":
    # Test cases
    print("Testing dskellam:", dskellam(0, 1, 1))
    print("Testing pskellam:", pskellam(0, 1, 1))
    print("Testing qskellam (scalar):", qskellam(0.5, 1, 1))
    print("Testing qskellam (array):", qskellam([0.5, 0.9], [2, 4], [1, 3]))
    print("Testing rskellam:", rskellam(5, 1, 1))
    print("Testing rskellam:", rskellam(5, 1, 1))
