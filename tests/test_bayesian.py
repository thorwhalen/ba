"""Tests for ba.bayesian (posteriors, priors, bayes_factors, sensitivity)."""

import numpy as np
import pytest

from ba.bayesian.priors import (
    jeffreys,
    uniform,
    from_mean_kappa,
    from_quantiles,
    from_counts,
    resolve_prior,
)
from ba.bayesian.posteriors import posterior, BayesianResult
from ba.bayesian.bayes_factors import bayes_factor
from ba.bayesian.sensitivity import sensitivity
from ba.core.contingency import ContingencyTable


# -- Priors ----------------------------------------------------------------


class TestPriors:
    def test_jeffreys_default(self):
        p = jeffreys()
        np.testing.assert_array_equal(p, [0.5, 0.5])

    def test_jeffreys_k3(self):
        p = jeffreys(3)
        assert len(p) == 3
        assert all(v == 0.5 for v in p)

    def test_uniform(self):
        np.testing.assert_array_equal(uniform(), [1.0, 1.0])

    def test_from_mean_kappa(self):
        # mean=0.3, kappa=10 => alpha=3, beta=7
        p = from_mean_kappa(0.3, 10)
        np.testing.assert_allclose(p, [3.0, 7.0])

    def test_from_mean_kappa_uniform(self):
        # mean=0.5, kappa=2 => Beta(1,1) = uniform
        p = from_mean_kappa(0.5, 2)
        np.testing.assert_allclose(p, [1.0, 1.0])

    def test_from_quantiles(self):
        a, b = from_quantiles(0.2, 0.05, 0.6, 0.95)
        assert a > 0 and b > 0
        # Check the solution roughly satisfies the constraints
        from scipy.stats import beta as beta_dist

        assert pytest.approx(beta_dist.cdf(0.2, a, b), abs=0.05) == 0.05
        assert pytest.approx(beta_dist.cdf(0.6, a, b), abs=0.05) == 0.95

    def test_from_counts(self):
        assert from_counts(2, 8) == (3, 9)
        assert from_counts(0, 0) == (1, 1)

    def test_resolve_jeffreys(self):
        np.testing.assert_array_equal(resolve_prior("jeffreys"), [0.5, 0.5])

    def test_resolve_uniform(self):
        np.testing.assert_array_equal(resolve_prior("uniform"), [1.0, 1.0])

    def test_resolve_haldane(self):
        p = resolve_prior("haldane")
        assert all(v < 1e-5 for v in p)

    def test_resolve_beta_string(self):
        np.testing.assert_array_equal(resolve_prior("beta(2,3)"), [2.0, 3.0])

    def test_resolve_tuple(self):
        np.testing.assert_array_equal(resolve_prior((3, 4)), [3.0, 4.0])

    def test_resolve_array(self):
        arr = np.array([1.5, 2.5])
        np.testing.assert_array_equal(resolve_prior(arr), arr)

    def test_resolve_none_defaults_to_jeffreys(self):
        np.testing.assert_array_equal(resolve_prior(None), [0.5, 0.5])

    def test_resolve_unknown_string(self):
        with pytest.raises(ValueError, match="Unknown prior"):
            resolve_prior("bogus_prior")

    def test_resolve_wrong_length(self):
        with pytest.raises(ValueError, match="length"):
            resolve_prior((1, 2, 3), k=2)


# -- Posteriors ------------------------------------------------------------


class TestPosterior2x2:
    @pytest.fixture()
    def ct(self):
        return ContingencyTable.from_counts(7, 3, 2, 8)

    def test_returns_bayesian_result(self, ct):
        r = posterior(ct)
        assert isinstance(r, BayesianResult)

    def test_posterior_mean_in_range(self, ct):
        r = posterior(ct)
        assert all(0 < m < 1 for m in r.posterior_mean)

    def test_posterior_mean_shrinkage(self, ct):
        """Posterior mean should be between prior mean and MLE."""
        r = posterior(ct, prior="uniform")
        mle_p1 = 7 / 10  # a / (a+b)
        prior_mean = 0.5
        # Posterior mean should be between MLE and prior
        assert min(prior_mean, mle_p1) < r.posterior_mean[0] < max(prior_mean, mle_p1)

    def test_credible_intervals_exist(self, ct):
        r = posterior(ct)
        for key in ("p1", "p0", "risk_difference", "relative_risk", "odds_ratio"):
            assert key in r.credible_interval
            lo, hi = r.credible_interval[key]
            assert lo < hi

    def test_data_weight(self, ct):
        # n1=10, n0=10, ESS_jeffreys=1 => w = 10/(10+1) = 0.909 per row
        r = posterior(ct, prior="jeffreys")
        assert r.data_weight > 0.9

    def test_data_weight_strong_prior(self, ct):
        r = posterior(ct, prior=(20, 20))
        # ESS = 40, n1=10 => w = 10/(10+40) = 0.2 per row
        assert r.data_weight < 0.3

    def test_mc_samples_present(self, ct):
        r = posterior(ct)
        assert r.mc_samples is not None
        assert "risk_difference" in r.mc_samples
        assert len(r.mc_samples["risk_difference"]) == 100_000

    def test_prob_gt(self, ct):
        r = posterior(ct)
        p = r.prob_gt(0.0, "risk_difference")
        # With a=7,b=3 vs c=2,d=8 there's strong evidence p1 > p0
        assert p > 0.9

    def test_prob_gt_unknown_param(self, ct):
        r = posterior(ct)
        with pytest.raises(ValueError, match="No MC samples"):
            r.prob_gt(0.0, "nonexistent")


class TestPosteriorRxC:
    @pytest.fixture()
    def ct(self):
        return ContingencyTable(
            counts=np.array([[5, 3, 2], [1, 4, 5]]),
            row_var="X",
            col_var="Y",
            row_labels=("A", "B"),
            col_labels=("good", "fair", "poor"),
        )

    def test_returns_result(self, ct):
        r = posterior(ct)
        assert isinstance(r, BayesianResult)

    def test_credible_intervals_per_cell(self, ct):
        r = posterior(ct)
        # Should have CIs for each p(col|row)
        assert "p(good|A)" in r.credible_interval
        assert "p(poor|B)" in r.credible_interval

    def test_no_mc_samples_for_rxc(self, ct):
        r = posterior(ct)
        assert r.mc_samples is None


# -- Bayes Factors ---------------------------------------------------------


class TestBayesFactor:
    @pytest.fixture()
    def ct_assoc(self):
        """Table with strong association."""
        return ContingencyTable.from_counts(10, 2, 3, 15)

    @pytest.fixture()
    def ct_null(self):
        """Table close to independence."""
        return ContingencyTable.from_counts(5, 5, 5, 5)

    def test_bf_positive_for_association(self, ct_assoc):
        bf = bayes_factor(ct_assoc)
        assert bf > 1  # evidence for association

    def test_bf_near_one_for_independence(self, ct_null):
        bf = bayes_factor(ct_null)
        assert bf < 3  # no strong evidence either way

    def test_sampling_schemes(self, ct_assoc):
        bf_indep = bayes_factor(ct_assoc, sampling="independent")
        bf_joint = bayes_factor(ct_assoc, sampling="joint")
        # Joint should give more evidence than independent
        assert bf_joint > 0
        assert bf_indep > 0

    def test_unknown_sampling_raises(self, ct_assoc):
        with pytest.raises(ValueError, match="Unknown sampling"):
            bayes_factor(ct_assoc, sampling="nonexistent")

    def test_bf_on_rxc(self):
        ct = ContingencyTable(
            counts=np.array([[10, 2], [3, 15], [5, 5]]),
            row_var="X",
            col_var="Y",
        )
        bf = bayes_factor(ct)
        assert bf > 0  # should not crash


# -- Sensitivity -----------------------------------------------------------


class TestSensitivity:
    def test_returns_dataframe(self):
        ct = ContingencyTable.from_counts(7, 3, 2, 8)
        df = sensitivity(ct)
        assert len(df) == 3  # default 3 priors
        assert "data_weight" in df.columns
        assert "prior_influenced" in df.columns
        assert "prior_dominated" in df.columns

    def test_custom_priors(self):
        ct = ContingencyTable.from_counts(7, 3, 2, 8)
        df = sensitivity(ct, priors=["jeffreys", "uniform"])
        assert len(df) == 2

    def test_flags_strong_prior(self):
        ct = ContingencyTable.from_counts(2, 1, 1, 2)
        df = sensitivity(ct, priors=["beta(2,2)", (20, 20)])
        # (20,20) has ESS=40, n=6 => w very low => dominated
        # The second row corresponds to the (20, 20) prior
        assert df["prior_dominated"].iloc[1] == True  # noqa: E712
