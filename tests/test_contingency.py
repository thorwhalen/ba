"""Tests for ba.core.contingency."""

import numpy as np
import pandas as pd
import pytest

from ba.core.contingency import ContingencyTable, ContingencyTable2x2


# -- Construction ----------------------------------------------------------


class TestContingencyTable2x2Construction:
    def test_from_counts(self):
        ct = ContingencyTable.from_counts(10, 5, 3, 12)
        assert isinstance(ct, ContingencyTable2x2)
        assert ct.a == 10
        assert ct.b == 5
        assert ct.c == 3
        assert ct.d == 12
        assert ct.n == 30

    def test_direct_construction(self):
        ct = ContingencyTable2x2(a=4, b=6, c=2, d=8)
        assert ct.n == 20
        assert ct.counts.shape == (2, 2)

    def test_from_dataframe_binary(self):
        df = pd.DataFrame({"X": [1, 1, 1, 0, 0, 0], "Y": [1, 1, 0, 0, 0, 1]})
        ct = ContingencyTable.from_dataframe(df, "X", "Y")
        assert isinstance(ct, ContingencyTable2x2)
        assert ct.n == 6

    def test_from_dataframe_categorical(self):
        df = pd.DataFrame({"X": ["a", "b", "c", "a", "b"], "Y": [1, 0, 1, 0, 1]})
        ct = ContingencyTable.from_dataframe(df, "X", "Y")
        assert isinstance(ct, ContingencyTable)
        assert not isinstance(ct, ContingencyTable2x2)
        assert ct.shape[0] == 3

    def test_counts_must_be_2d(self):
        with pytest.raises(ValueError, match="2D"):
            ContingencyTable(counts=np.array([1, 2, 3]), row_var="X", col_var="Y")


# -- Properties ------------------------------------------------------------


class TestContingencyTable2x2Properties:
    @pytest.fixture()
    def ct(self):
        return ContingencyTable.from_counts(10, 5, 3, 12)

    def test_margins(self, ct):
        np.testing.assert_array_equal(ct.row_margins, [15, 15])
        np.testing.assert_array_equal(ct.col_margins, [13, 17])

    def test_expected(self, ct):
        exp = ct.expected
        # row_margin * col_margin / n for top-left: 15*13/30 = 6.5
        assert pytest.approx(exp[0, 0], rel=1e-6) == 6.5

    def test_is_2x2(self, ct):
        assert ct.is_2x2 is True

    def test_has_zero_cell(self, ct):
        assert ct.has_zero_cell is False
        ct_zero = ContingencyTable.from_counts(5, 0, 3, 7)
        assert ct_zero.has_zero_cell is True


# -- Binary-specific metrics -----------------------------------------------


class TestBinaryMetrics:
    @pytest.fixture()
    def ct(self):
        # a=10, b=5, c=3, d=12 => OR = 10*12/(5*3) = 8.0
        return ContingencyTable.from_counts(10, 5, 3, 12)

    def test_odds_ratio(self, ct):
        assert ct.odds_ratio == 8.0

    def test_odds_ratio_zero_cell(self):
        ct = ContingencyTable.from_counts(5, 0, 3, 7)
        assert ct.odds_ratio is None  # b=0

    def test_relative_risk(self, ct):
        # (10/15) / (3/15) = 10/3 ≈ 3.333
        assert pytest.approx(ct.relative_risk, rel=1e-3) == 10 / 3

    def test_risk_difference(self, ct):
        # 10/15 - 3/15 = 7/15 ≈ 0.4667
        assert pytest.approx(ct.risk_difference, rel=1e-3) == 7 / 15

    def test_phi(self, ct):
        # (10*12 - 5*3) / sqrt(15*15*13*17)
        num = 10 * 12 - 5 * 3
        denom = np.sqrt(15 * 15 * 13 * 17)
        assert pytest.approx(ct.phi, rel=1e-6) == num / denom

    def test_yules_q(self, ct):
        # (ad - bc) / (ad + bc) = (120 - 15) / (120 + 15) = 105/135
        assert pytest.approx(ct.yules_q, rel=1e-6) == 105 / 135

    def test_qca_consistency(self, ct):
        assert pytest.approx(ct.qca_consistency, rel=1e-6) == 10 / 15

    def test_qca_coverage(self, ct):
        assert pytest.approx(ct.qca_coverage, rel=1e-6) == 10 / 13

    def test_fisher_p(self, ct):
        p = ct.fisher_p
        assert 0 < p < 1
        # For this table, Fisher's p ≈ 0.025
        assert p < 0.05


# -- Conversion ------------------------------------------------------------


class TestConversion:
    def test_as_2x2_roundtrip(self):
        ct = ContingencyTable.from_counts(4, 6, 2, 8)
        assert ct.as_2x2() is ct  # already 2x2, returns self

    def test_as_2x2_fails_on_rxc(self):
        ct = ContingencyTable(
            counts=np.array([[1, 2, 3], [4, 5, 6]]),
            row_var="X",
            col_var="Y",
        )
        with pytest.raises(ValueError, match="not 2×2"):
            ct.as_2x2()

    def test_to_dataframe(self):
        ct = ContingencyTable.from_counts(10, 5, 3, 12)
        df = ct.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert df.iloc[0, 0] == 10

    def test_repr(self):
        ct = ContingencyTable.from_counts(1, 2, 3, 4)
        assert "ContingencyTable2x2" in repr(ct)
        assert "n=10" in repr(ct)


# -- r×c table -----------------------------------------------------------


class TestRxCTable:
    @pytest.fixture()
    def ct(self):
        return ContingencyTable(
            counts=np.array([[5, 3, 2], [1, 4, 5]]),
            row_var="treatment",
            col_var="outcome",
            row_labels=("drug", "placebo"),
            col_labels=("good", "fair", "poor"),
        )

    def test_shape(self, ct):
        assert ct.shape == (2, 3)

    def test_is_not_2x2(self, ct):
        assert ct.is_2x2 is False

    def test_n(self, ct):
        assert ct.n == 20

    def test_labels(self, ct):
        assert ct.row_labels == ("drug", "placebo")
        assert ct.col_labels == ("good", "fair", "poor")

    def test_default_labels(self):
        ct = ContingencyTable(
            counts=np.array([[1, 2], [3, 4], [5, 6]]),
            row_var="A",
            col_var="B",
        )
        assert ct.row_labels == ("0", "1", "2")
