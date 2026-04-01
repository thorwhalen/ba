"""Prior construction helpers for Beta and Dirichlet distributions."""

from __future__ import annotations

import re

import numpy as np
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import beta as beta_dist


def jeffreys(k: int = 2) -> np.ndarray:
    """Jeffreys prior: Dirichlet(0.5, ..., 0.5).

    For k=2, this is Beta(0.5, 0.5).

    >>> jeffreys()
    array([0.5, 0.5])
    >>> jeffreys(3)
    array([0.5, 0.5, 0.5])
    """
    return np.full(k, 0.5)


def uniform(k: int = 2) -> np.ndarray:
    """Uniform (Bayes-Laplace) prior: Dirichlet(1, ..., 1).

    For k=2, this is Beta(1, 1).

    >>> uniform()
    array([1., 1.])
    """
    return np.ones(k)


def from_mean_kappa(mean: float | np.ndarray, kappa: float) -> np.ndarray:
    """Construct prior from mean and concentration (κ = α + β).

    κ = 2 gives uniform; κ = 50 gives a strongly informative prior.
    The "equivalent sample size" is κ.

    >>> from_mean_kappa(0.3, 10)
    array([3., 7.])
    >>> from_mean_kappa(0.5, 2)
    array([1., 1.])
    """
    mean = np.asarray(mean, dtype=float)
    if mean.ndim == 0:
        # Scalar mean: return Beta(mean*kappa, (1-mean)*kappa)
        return np.array([float(mean * kappa), float((1 - mean) * kappa)])
    return mean * kappa


def from_quantiles(
    q1: float, p1: float, q2: float, p2: float
) -> tuple[float, float]:
    """Solve for Beta(α, β) matching P(θ < q1) = p1 and P(θ < q2) = p2.

    Uses least-squares minimization in log-space for numerical stability.

    >>> a, b = from_quantiles(0.2, 0.05, 0.6, 0.95)
    >>> 1.0 < a < 10.0
    True
    """

    def objective(log_ab):
        a, b = np.exp(log_ab)
        r1 = beta_dist.cdf(q1, a, b) - p1
        r2 = beta_dist.cdf(q2, a, b) - p2
        return r1**2 + r2**2

    res = scipy_minimize(objective, [np.log(2), np.log(2)], method="Nelder-Mead")
    a, b = np.exp(res.x)
    return (float(a), float(b))


def from_counts(successes: int, failures: int) -> tuple[float, float]:
    """'Imaginary data' framing: Beta(successes + 1, failures + 1).

    "Think of the prior as having already seen these imaginary outcomes."

    >>> from_counts(2, 8)
    (3, 9)
    """
    return (successes + 1, failures + 1)


def resolve_prior(
    prior: str | tuple | np.ndarray | None,
    k: int = 2,
) -> np.ndarray:
    """Resolve a prior specification to a Dirichlet parameter array.

    Accepts:
      - 'jeffreys', 'uniform', 'haldane'
      - 'beta(a,b)' for k=2
      - tuple (a, b) for k=2
      - numpy array of length k

    >>> resolve_prior('jeffreys')
    array([0.5, 0.5])
    >>> resolve_prior('beta(2,3)')
    array([2., 3.])
    >>> resolve_prior((1, 1))
    array([1., 1.])
    """
    if prior is None:
        return jeffreys(k)

    if isinstance(prior, np.ndarray):
        if len(prior) != k:
            raise ValueError(f"Prior array length {len(prior)} != k={k}")
        return prior

    if isinstance(prior, (tuple, list)):
        arr = np.asarray(prior, dtype=float)
        if len(arr) != k:
            raise ValueError(f"Prior length {len(arr)} != k={k}")
        return arr

    if isinstance(prior, str):
        prior_lower = prior.lower().strip()
        if prior_lower == "jeffreys":
            return jeffreys(k)
        if prior_lower == "uniform":
            return uniform(k)
        if prior_lower == "haldane":
            return np.full(k, 1e-10)
        # Parse 'beta(a,b)' or 'dirichlet(a1,a2,...)'
        m = re.match(r"beta\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", prior_lower)
        if m:
            if k != 2:
                raise ValueError(f"'beta(a,b)' only valid for k=2, got k={k}")
            return np.array([float(m.group(1)), float(m.group(2))])
        m = re.match(r"dirichlet\((.+)\)", prior_lower)
        if m:
            vals = [float(x.strip()) for x in m.group(1).split(",")]
            if len(vals) != k:
                raise ValueError(f"Dirichlet({len(vals)} params) != k={k}")
            return np.array(vals)
        raise ValueError(
            f"Unknown prior '{prior}'. Use 'jeffreys', 'uniform', 'haldane', "
            f"'beta(a,b)', 'dirichlet(a1,...)', a tuple, or a numpy array."
        )

    raise TypeError(f"Cannot resolve prior of type {type(prior)}")
