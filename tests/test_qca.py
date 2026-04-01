"""Tests for ba.qca (calibrate, truth_table, minimize, necessity)."""

import numpy as np
import pandas as pd
import pytest

from ba.qca.calibrate import calibrate
from ba.qca.truth_table import truth_table
from ba.qca.minimize import minimize, QCASolution
from ba.qca.necessity import necessity, sufficiency


# -- Calibration -----------------------------------------------------------


class TestCalibrate:
    @pytest.fixture()
    def df(self):
        return pd.DataFrame({
            "age": [25, 35, 45, 55],
            "illness": ["no", "yes", "none", "yes"],
            "score": [0.3, 0.7, 0.5, 0.9],
        })

    def test_numeric_threshold(self, df):
        result = calibrate(df, {"age": 40})
        assert result["age"].tolist() == [0, 0, 1, 1]

    def test_any_present(self, df):
        result = calibrate(df, {"illness": "any_present"})
        assert result["illness"].tolist() == [0, 1, 0, 1]

    def test_median(self, df):
        result = calibrate(df, {"score": "median"})
        # median of [0.3, 0.7, 0.5, 0.9] = 0.6
        assert result["score"].tolist() == [0, 1, 0, 1]

    def test_callable(self, df):
        result = calibrate(df, {"age": lambda x: x > 30})
        assert result["age"].tolist() == [0, 1, 1, 1]

    def test_unknown_column_raises(self, df):
        with pytest.raises(KeyError, match="nonexistent"):
            calibrate(df, {"nonexistent": 10})

    def test_unknown_string_rule_raises(self, df):
        with pytest.raises(ValueError, match="Unknown string rule"):
            calibrate(df, {"age": "bogus_rule"})

    def test_passthrough_columns(self, df):
        result = calibrate(df, {"age": 40})
        # illness and score should be unchanged
        assert result["illness"].tolist() == df["illness"].tolist()

    def test_multiple_columns(self, df):
        result = calibrate(df, {"age": 40, "illness": "any_present"})
        assert result["age"].tolist() == [0, 0, 1, 1]
        assert result["illness"].tolist() == [0, 1, 0, 1]


# -- Truth Table -----------------------------------------------------------


class TestTruthTable:
    @pytest.fixture()
    def df(self):
        return pd.DataFrame({
            "A": [1, 1, 1, 0, 0, 0, 1, 0],
            "B": [1, 0, 1, 0, 1, 0, 0, 1],
            "Y": [1, 1, 1, 0, 0, 0, 1, 0],
        })

    def test_basic(self, df):
        tt = truth_table(df, "Y", ["A", "B"])
        assert "consistency" in tt.columns
        assert "n" in tt.columns
        assert "OUT" in tt.columns
        assert len(tt) <= 4  # at most 2^2 rows

    def test_n_cut_filtering(self, df):
        tt = truth_table(df, "Y", ["A", "B"], n_cut=2)
        assert all(tt["n"] >= 2)

    def test_non_binary_raises(self):
        df = pd.DataFrame({"A": [1, 2, 3], "Y": [1, 0, 1]})
        with pytest.raises(ValueError, match="Non-binary"):
            truth_table(df, "Y", ["A"])

    def test_missing_column_raises(self):
        df = pd.DataFrame({"A": [1, 0], "Y": [1, 0]})
        with pytest.raises(ValueError, match="not found"):
            truth_table(df, "Y", ["nonexistent"])

    def test_flag_low_n(self):
        df = pd.DataFrame({
            "A": [1, 0, 0, 0],
            "B": [1, 0, 0, 0],
            "Y": [1, 0, 0, 0],
        })
        tt = truth_table(df, "Y", ["A", "B"])
        # A=1, B=1 has only 1 case
        flagged = tt[tt["flag"] == "low_n"]
        assert len(flagged) > 0


# -- Boolean Minimization --------------------------------------------------


class TestMinimize:
    def test_simple_or(self):
        """A + B => A*B already present as 01, 10, 11 → minimizes to A + B."""
        tt = pd.DataFrame({
            "A": [0, 0, 1, 1],
            "B": [0, 1, 0, 1],
            "OUT": [0, 1, 1, 1],
            "n": [3, 2, 2, 4],
            "consistency": [0.0, 1.0, 1.0, 1.0],
            "flag": ["", "", "", ""],
        })
        sol = minimize(tt)
        assert isinstance(sol, QCASolution)
        # Should produce expression like "A + B" (simplified from A*B + A*~B + ~A*B)
        assert "A" in sol.expression or "B" in sol.expression

    def test_no_positive_rows(self):
        tt = pd.DataFrame({
            "A": [0, 1], "OUT": [0, 0],
            "n": [3, 2], "consistency": [0.0, 0.0], "flag": ["", ""],
        })
        sol = minimize(tt)
        assert "0" in sol.expression

    def test_single_minterm(self):
        tt = pd.DataFrame({
            "A": [0, 1], "B": [0, 1],
            "OUT": [0, 1],
            "n": [3, 2], "consistency": [0.0, 1.0], "flag": ["", ""],
        })
        sol = minimize(tt)
        assert isinstance(sol, QCASolution)
        assert len(sol.prime_implicants) >= 1

    def test_to_dataframe(self):
        tt = pd.DataFrame({
            "A": [0, 0, 1, 1], "B": [0, 1, 0, 1],
            "OUT": [0, 1, 1, 1],
            "n": [3, 2, 2, 4], "consistency": [0.0, 1.0, 1.0, 1.0],
            "flag": ["", "", "", ""],
        })
        sol = minimize(tt)
        df = sol.to_dataframe()
        assert isinstance(df, pd.DataFrame)


# -- Necessity / Sufficiency -----------------------------------------------


class TestNecessitySufficiency:
    @pytest.fixture()
    def df(self):
        return pd.DataFrame({
            "A": [1, 1, 0, 1, 0],
            "B": [1, 1, 1, 0, 0],
            "Y": [1, 1, 0, 0, 0],
        })

    def test_necessity_columns(self, df):
        result = necessity(df, "Y", ["A", "B"])
        assert "condition" in result.columns
        assert "consistency" in result.columns
        assert "ci_low" in result.columns
        assert len(result) == 2

    def test_necessity_values(self, df):
        result = necessity(df, "Y", ["A", "B"])
        # Y=1 for rows 0,1. A=1 for rows 0,1,3. B=1 for rows 0,1,2.
        # necessity(A) = P(A|Y) = 2/2 = 1.0 (both Y=1 cases have A=1)
        a_row = result[result["condition"] == "A"]
        assert a_row["consistency"].iloc[0] == 1.0
        # necessity(B) = P(B|Y) = 2/2 = 1.0 (both Y=1 cases have B=1)
        b_row = result[result["condition"] == "B"]
        assert b_row["consistency"].iloc[0] == 1.0

    def test_sufficiency_columns(self, df):
        result = sufficiency(df, "Y", ["A", "B"])
        assert "condition" in result.columns
        assert "consistency" in result.columns
        assert len(result) == 2

    def test_sufficiency_values(self, df):
        result = sufficiency(df, "Y", ["A", "B"])
        # A=1 for rows 0,1,3. Y=1 for rows 0,1.
        # sufficiency(A) = P(Y|A) = 2/3 ≈ 0.6667
        a_row = result[result["condition"] == "A"]
        assert pytest.approx(a_row["consistency"].iloc[0], rel=0.01) == 2 / 3

    def test_auto_conditions(self, df):
        # If conditions=None, use all non-outcome columns
        result = necessity(df, "Y")
        assert len(result) == 2
