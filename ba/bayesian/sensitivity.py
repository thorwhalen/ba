"""Prior sensitivity analysis for Bayesian inference on contingency tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ba.bayesian.posteriors import posterior
from ba.core.contingency import ContingencyTable


def sensitivity(
    ct: ContingencyTable,
    *,
    priors: tuple[str, ...] | list[str] = ("jeffreys", "uniform", "beta(2,2)"),
    n_mc: int = 100_000,
    ci_prob: float = 0.95,
) -> pd.DataFrame:
    """Compute posterior under multiple priors and compare.

    Returns a DataFrame with one row per prior, showing posterior means,
    credible intervals, data weight, and flags for prior influence.

    Flags:
        prior_influenced: data_weight < 0.8
        prior_dominated: data_weight < 0.5

    >>> from ba.core.contingency import ContingencyTable
    >>> ct = ContingencyTable.from_counts(7, 3, 2, 8)
    >>> df = sensitivity(ct)
    >>> len(df) == 3
    True
    >>> 'data_weight' in df.columns
    True
    """
    rows = []
    for prior_spec in priors:
        result = posterior(ct, prior=prior_spec, n_mc=n_mc, ci_prob=ci_prob)

        row = {
            "prior": prior_spec,
            "data_weight": result.data_weight,
            "prior_influenced": result.data_weight < 0.8,
            "prior_dominated": result.data_weight < 0.5,
        }

        # Add posterior means
        for i, mean_val in enumerate(result.posterior_mean.flat):
            row[f"posterior_mean_{i}"] = mean_val

        # Add key CIs
        if ct.is_2x2:
            for key in ("risk_difference", "odds_ratio"):
                if key in result.credible_interval:
                    ci = result.credible_interval[key]
                    row[f"{key}_ci_low"] = ci[0]
                    row[f"{key}_ci_high"] = ci[1]

            # P(RD > 0 | data)
            if result.mc_samples and "risk_difference" in result.mc_samples:
                row["prob_rd_gt_0"] = float(
                    np.mean(result.mc_samples["risk_difference"] > 0)
                )

        rows.append(row)

    return pd.DataFrame(rows)
