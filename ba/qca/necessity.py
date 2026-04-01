"""Necessity and sufficiency analysis for individual conditions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


def necessity(
    data: pd.DataFrame,
    outcome: str,
    conditions: list[str] | None = None,
    *,
    prior_alpha: float = 0.5,
    prior_beta: float = 0.5,
    ci_prob: float = 0.95,
) -> pd.DataFrame:
    """Necessity analysis: is each condition necessary for the outcome?

    A condition X is necessary for outcome Y if Y ⊆ X — i.e., whenever
    Y is present, X is also present. Measured by consistency of necessity =
    P(X|Y) = a / (a + c).

    Returns a DataFrame with one row per condition.

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1,1,0,1,0], 'B': [1,1,1,0,0], 'Y': [1,1,0,0,0]
    ... })
    >>> result = necessity(df, 'Y', ['A', 'B'])
    >>> list(result.columns[:3])
    ['condition', 'consistency', 'coverage']
    """
    if conditions is None:
        conditions = [c for c in data.columns if c != outcome]

    alpha_lo = (1 - ci_prob) / 2
    alpha_hi = 1 - alpha_lo
    rows = []

    y = data[outcome].astype(bool)
    n_y = y.sum()

    for cond in conditions:
        x = data[cond].astype(bool)
        a = int((x & y).sum())
        c = int((~x & y).sum())
        b = int((x & ~y).sum())

        consistency = a / (a + c) if (a + c) > 0 else 0.0  # P(X|Y)
        coverage = a / (a + b) if (a + b) > 0 else 0.0  # P(Y|X)

        # Bayesian CI for consistency
        post_a = prior_alpha + a
        post_b = prior_beta + c
        ci = (
            float(beta_dist.ppf(alpha_lo, post_a, post_b)),
            float(beta_dist.ppf(alpha_hi, post_a, post_b)),
        )

        rows.append({
            "condition": cond,
            "consistency": round(consistency, 4),
            "coverage": round(coverage, 4),
            "n_xy": a,
            "n_y": int(n_y),
            "ci_low": round(ci[0], 4),
            "ci_high": round(ci[1], 4),
        })

    return pd.DataFrame(rows)


def sufficiency(
    data: pd.DataFrame,
    outcome: str,
    conditions: list[str] | None = None,
    *,
    prior_alpha: float = 0.5,
    prior_beta: float = 0.5,
    ci_prob: float = 0.95,
) -> pd.DataFrame:
    """Sufficiency analysis: is each condition sufficient for the outcome?

    A condition X is sufficient for outcome Y if X ⊆ Y — i.e., whenever
    X is present, Y is also present. Measured by consistency of sufficiency =
    P(Y|X) = a / (a + b).

    Returns a DataFrame with one row per condition.

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1,1,0,1,0], 'B': [1,1,1,0,0], 'Y': [1,1,0,0,0]
    ... })
    >>> result = sufficiency(df, 'Y', ['A', 'B'])
    >>> list(result.columns[:3])
    ['condition', 'consistency', 'coverage']
    """
    if conditions is None:
        conditions = [c for c in data.columns if c != outcome]

    alpha_lo = (1 - ci_prob) / 2
    alpha_hi = 1 - alpha_lo
    rows = []

    for cond in conditions:
        x = data[cond].astype(bool)
        y = data[outcome].astype(bool)
        a = int((x & y).sum())
        b = int((x & ~y).sum())
        c = int((~x & y).sum())

        consistency = a / (a + b) if (a + b) > 0 else 0.0  # P(Y|X)
        coverage = a / (a + c) if (a + c) > 0 else 0.0  # P(X|Y)

        # Bayesian CI for consistency
        post_a = prior_alpha + a
        post_b = prior_beta + b
        ci = (
            float(beta_dist.ppf(alpha_lo, post_a, post_b)),
            float(beta_dist.ppf(alpha_hi, post_a, post_b)),
        )

        rows.append({
            "condition": cond,
            "consistency": round(consistency, 4),
            "coverage": round(coverage, 4),
            "n_x": int(x.sum()),
            "n_xy": a,
            "ci_low": round(ci[0], 4),
            "ci_high": round(ci[1], 4),
        })

    return pd.DataFrame(rows)
