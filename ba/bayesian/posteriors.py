"""Bayesian posterior inference for contingency tables."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from ba.bayesian.priors import resolve_prior
from ba.core.contingency import ContingencyTable


@dataclass
class BayesianResult:
    """Container for Bayesian posterior inference results.

    >>> from ba.core.contingency import ContingencyTable
    >>> ct = ContingencyTable.from_counts(7, 3, 2, 8)
    >>> r = posterior(ct)
    >>> 0.5 < float(r.posterior_mean[0]) < 0.9
    True
    >>> float(r.data_weight) > 0.9
    True
    """

    posterior_params: dict
    posterior_mean: np.ndarray
    credible_interval: dict[str, tuple[float, float]]
    bayes_factor: float | None = None
    data_weight: float = 1.0
    prior_params: np.ndarray = field(default_factory=lambda: np.array([]))
    prior_name: str = ""
    mc_samples: dict[str, np.ndarray] | None = None

    def summary(self) -> pd.DataFrame:
        """One-row-per-parameter summary."""
        rows = []
        for key, ci in self.credible_interval.items():
            row = {"parameter": key, "ci_low": ci[0], "ci_high": ci[1]}
            if key in (self.posterior_params or {}):
                params = self.posterior_params[key]
                if isinstance(params, dict) and "alpha" in params:
                    row["posterior_alpha"] = params["alpha"]
                    row["posterior_beta"] = params["beta"]
            rows.append(row)
        df = pd.DataFrame(rows)
        df.attrs["data_weight"] = self.data_weight
        df.attrs["prior"] = self.prior_name
        return df

    def prob_gt(self, threshold: float = 0.0, parameter: str = "risk_difference") -> float:
        """P(parameter > threshold | data) from MC samples."""
        if self.mc_samples is None or parameter not in self.mc_samples:
            raise ValueError(
                f"No MC samples for '{parameter}'. "
                f"Available: {list(self.mc_samples or {})}"
            )
        return float(np.mean(self.mc_samples[parameter] > threshold))


def posterior(
    ct: ContingencyTable,
    *,
    prior: str | tuple | np.ndarray | None = "jeffreys",
    n_mc: int = 100_000,
    ci_prob: float = 0.95,
) -> BayesianResult:
    """Bayesian posterior for row-conditional proportions.

    For 2×2: Beta-Binomial with MC for derived quantities (RD, RR, OR).
    For r×c: Dirichlet-Multinomial per row.

    Args:
        ct: Contingency table.
        prior: Prior specification (see ba.bayesian.priors.resolve_prior).
        n_mc: Number of Monte Carlo draws for derived quantities.
        ci_prob: Probability mass for credible intervals (default 0.95).

    Returns:
        BayesianResult with posterior parameters, CIs, MC samples (2×2 only).
    """
    k = ct.shape[1]  # number of columns (categories for outcome)
    prior_arr = resolve_prior(prior, k=k)
    prior_name = prior if isinstance(prior, str) else repr(prior)

    alpha_low = (1 - ci_prob) / 2
    alpha_high = 1 - alpha_low

    # ESS of the prior is sum of prior params
    ess_prior = float(prior_arr.sum())

    if ct.is_2x2:
        return _beta_binomial(ct.as_2x2(), prior_arr, prior_name, ess_prior, n_mc, alpha_low, alpha_high)
    else:
        return _dirichlet_multinomial(ct, prior_arr, prior_name, ess_prior, alpha_low, alpha_high)


def _beta_binomial(ct, prior_arr, prior_name, ess_prior, n_mc, alpha_low, alpha_high):
    """Beta-Binomial posterior for a 2×2 table."""
    a, b, c, d = ct.a, ct.b, ct.c, ct.d
    n1 = a + b
    n0 = c + d

    # Prior params for each row
    a1_prior, b1_prior = prior_arr[0], prior_arr[1]
    a0_prior, b0_prior = prior_arr[0], prior_arr[1]

    # Posterior params
    a1_post = a1_prior + a
    b1_post = b1_prior + b
    a0_post = a0_prior + c
    b0_post = b0_prior + d

    # Posterior means
    mean1 = a1_post / (a1_post + b1_post)
    mean0 = a0_post / (a0_post + b0_post)

    # Credible intervals for proportions
    ci_p1 = (
        float(beta_dist.ppf(alpha_low, a1_post, b1_post)),
        float(beta_dist.ppf(alpha_high, a1_post, b1_post)),
    )
    ci_p0 = (
        float(beta_dist.ppf(alpha_low, a0_post, b0_post)),
        float(beta_dist.ppf(alpha_high, a0_post, b0_post)),
    )

    # Data weight (average across rows)
    w1 = n1 / (n1 + ess_prior)
    w0 = n0 / (n0 + ess_prior)
    data_weight = (w1 + w0) / 2

    # Monte Carlo for derived quantities
    rng = np.random.default_rng()
    pi1 = rng.beta(a1_post, b1_post, size=n_mc)
    pi0 = rng.beta(a0_post, b0_post, size=n_mc)

    rd = pi1 - pi0
    # Avoid division by zero for RR and OR
    safe_pi0 = np.where(pi0 > 1e-15, pi0, 1e-15)
    safe_1_minus = np.where((1 - pi1) > 1e-15, 1 - pi1, 1e-15)
    safe_1_minus0 = np.where((1 - pi0) > 1e-15, 1 - pi0, 1e-15)
    rr = pi1 / safe_pi0
    or_samples = (pi1 / safe_1_minus) / (safe_pi0 / safe_1_minus0)

    mc_samples = {
        "p1": pi1,
        "p0": pi0,
        "risk_difference": rd,
        "relative_risk": rr,
        "odds_ratio": or_samples,
    }

    ci_rd = (float(np.percentile(rd, alpha_low * 100)), float(np.percentile(rd, alpha_high * 100)))
    ci_rr = (float(np.percentile(rr, alpha_low * 100)), float(np.percentile(rr, alpha_high * 100)))
    ci_or = (float(np.percentile(or_samples, alpha_low * 100)), float(np.percentile(or_samples, alpha_high * 100)))

    return BayesianResult(
        posterior_params={
            "p1": {"alpha": a1_post, "beta": b1_post},
            "p0": {"alpha": a0_post, "beta": b0_post},
        },
        posterior_mean=np.array([mean1, mean0]),
        credible_interval={
            "p1": ci_p1,
            "p0": ci_p0,
            "risk_difference": ci_rd,
            "relative_risk": ci_rr,
            "odds_ratio": ci_or,
        },
        data_weight=data_weight,
        prior_params=prior_arr,
        prior_name=prior_name,
        mc_samples=mc_samples,
    )


def _dirichlet_multinomial(ct, prior_arr, prior_name, ess_prior, alpha_low, alpha_high):
    """Dirichlet-Multinomial posterior for an r×c table."""
    posterior_params = {}
    posterior_means = []
    cis = {}

    for i in range(ct.shape[0]):
        row_counts = ct.counts[i]
        post_alpha = prior_arr + row_counts
        post_mean = post_alpha / post_alpha.sum()
        posterior_means.append(post_mean)

        label = ct.row_labels[i] if i < len(ct.row_labels) else str(i)
        posterior_params[f"row_{label}"] = {"alpha": post_alpha.tolist()}

        # CI for each category proportion in this row (marginal Beta)
        for j in range(ct.shape[1]):
            col_label = ct.col_labels[j] if j < len(ct.col_labels) else str(j)
            key = f"p({col_label}|{label})"
            a_j = post_alpha[j]
            b_j = post_alpha.sum() - a_j
            cis[key] = (
                float(beta_dist.ppf(alpha_low, a_j, b_j)),
                float(beta_dist.ppf(alpha_high, a_j, b_j)),
            )

    # Data weight
    n_total = ct.n
    data_weight = n_total / (n_total + ess_prior * ct.shape[0])

    return BayesianResult(
        posterior_params=posterior_params,
        posterior_mean=np.array(posterior_means),
        credible_interval=cis,
        data_weight=data_weight,
        prior_params=prior_arr,
        prior_name=prior_name,
        mc_samples=None,
    )
