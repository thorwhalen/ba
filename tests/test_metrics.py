"""Tests for ba.core.metrics."""

import warnings

import numpy as np
import pytest

from ba.core.contingency import ContingencyTable, ContingencyTable2x2
from ba.core.metrics import MeasureRegistry, registry


# -- Registry behavior -----------------------------------------------------


class TestMeasureRegistry:
    def test_registry_has_measures(self):
        assert len(registry) >= 22

    def test_available_all(self):
        names = registry.available()
        assert "support" in names
        assert "odds_ratio" in names

    def test_available_filtered_for_rxc(self):
        ct = ContingencyTable(
            counts=np.array([[5, 3, 2], [1, 4, 5]]),
            row_var="X",
            col_var="Y",
        )
        names = registry.available(ct)
        assert "support" in names
        assert "odds_ratio" not in names  # binary-only excluded

    def test_describe(self):
        desc = registry.describe("support")
        assert len(desc) > 0

    def test_describe_unknown(self):
        with pytest.raises(KeyError):
            registry.describe("nonexistent_measure")

    def test_unknown_measure_warning(self):
        ct = ContingencyTable.from_counts(5, 5, 5, 5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            registry.compute(ct, ["bogus_measure"])
            assert any("Unknown measure" in str(x.message) for x in w)

    def test_binary_measure_on_rxc_skipped_in_default(self):
        ct = ContingencyTable(
            counts=np.array([[5, 3, 2], [1, 4, 5]]),
            row_var="X",
            col_var="Y",
        )
        result = ct.metrics()
        assert "odds_ratio" not in result
        assert "support" in result

    def test_binary_measure_on_rxc_warns_when_explicit(self):
        ct = ContingencyTable(
            counts=np.array([[5, 3, 2], [1, 4, 5]]),
            row_var="X",
            col_var="Y",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = registry.compute(ct, ["odds_ratio"])
            assert "odds_ratio" not in result
            assert any("2×2" in str(x.message) for x in w)

    def test_user_registration(self):
        r = MeasureRegistry()
        r.register("always_one", lambda ct: 1.0, description="test measure")
        ct = ContingencyTable.from_counts(1, 2, 3, 4)
        assert r.compute(ct, ["always_one"]) == {"always_one": 1.0}
        assert "always_one" in r


# -- Individual measures against known values ------------------------------


class TestMeasureValues:
    """Validate each measure against hand-computed values.

    Table: a=10, b=5, c=3, d=12, n=30.
    P(X) = 15/30 = 0.5, P(Y) = 13/30, P(XY) = 10/30.
    """

    @pytest.fixture()
    def ct(self):
        return ContingencyTable.from_counts(10, 5, 3, 12)

    def test_support(self, ct):
        assert pytest.approx(registry.compute(ct, ["support"])["support"]) == 10 / 30

    def test_confidence(self, ct):
        assert pytest.approx(registry.compute(ct, ["confidence"])["confidence"]) == 10 / 15

    def test_lift(self, ct):
        # confidence / P(Y) = (10/15) / (13/30) = 20/13
        assert pytest.approx(registry.compute(ct, ["lift"])["lift"], rel=1e-6) == 20 / 13

    def test_leverage(self, ct):
        # P(XY) - P(X)*P(Y) = 10/30 - 0.5*(13/30)
        expected = 10 / 30 - 0.5 * (13 / 30)
        assert pytest.approx(registry.compute(ct, ["leverage"])["leverage"], abs=1e-6) == expected

    def test_cosine(self, ct):
        # a / sqrt(15 * 13) = 10 / sqrt(195)
        expected = 10 / np.sqrt(15 * 13)
        assert pytest.approx(registry.compute(ct, ["cosine"])["cosine"], rel=1e-6) == expected

    def test_jaccard(self, ct):
        # a / (row0 + col0 - a) = 10 / (15 + 13 - 10) = 10/18
        assert pytest.approx(registry.compute(ct, ["jaccard"])["jaccard"], rel=1e-6) == 10 / 18

    def test_kulczynski(self, ct):
        # 0.5 * (10/15 + 10/13)
        expected = 0.5 * (10 / 15 + 10 / 13)
        assert pytest.approx(registry.compute(ct, ["kulczynski"])["kulczynski"], rel=1e-6) == expected

    def test_cramers_v_equals_abs_phi_for_2x2(self, ct):
        m = registry.compute(ct, ["cramers_v", "phi"])
        assert pytest.approx(m["cramers_v"], rel=1e-6) == abs(m["phi"])

    def test_phi(self, ct):
        num = 10 * 12 - 5 * 3
        denom = np.sqrt(15 * 15 * 13 * 17)
        assert pytest.approx(registry.compute(ct, ["phi"])["phi"], rel=1e-6) == num / denom

    def test_yules_q(self, ct):
        assert pytest.approx(registry.compute(ct, ["yules_q"])["yules_q"], rel=1e-6) == 105 / 135

    def test_mutual_info_nonnegative(self, ct):
        mi = registry.compute(ct, ["mutual_info"])["mutual_info"]
        assert mi >= 0

    def test_goodman_kruskal_gamma_equals_yules_q_for_2x2(self, ct):
        """For 2×2, Goodman-Kruskal γ = Yule's Q."""
        m = registry.compute(ct, ["goodman_kruskal_gamma", "yules_q"])
        assert pytest.approx(m["goodman_kruskal_gamma"], rel=1e-6) == m["yules_q"]

    def test_fisher_p_range(self, ct):
        p = registry.compute(ct, ["fisher_p"])["fisher_p"]
        assert 0 < p < 1

    def test_odds_ratio(self, ct):
        assert registry.compute(ct, ["odds_ratio"])["odds_ratio"] == 8.0

    def test_risk_difference(self, ct):
        assert pytest.approx(
            registry.compute(ct, ["risk_difference"])["risk_difference"], rel=1e-3
        ) == 7 / 15

    def test_all_default_on_2x2(self, ct):
        """All 21 measures should be computed for a 2×2 table."""
        result = ct.metrics()
        assert len(result) == len(registry)
