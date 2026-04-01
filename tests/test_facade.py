"""Tests for ba.analyze() façade and AnalysisResult."""

import pandas as pd
import pytest

import ba


class TestAnalyze:
    @pytest.fixture()
    def df(self):
        return pd.DataFrame({
            "A": [1, 1, 0, 0, 1, 0, 1, 0, 1, 1],
            "B": [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
            "Y": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
        })

    def test_basic(self, df):
        result = ba.analyze(df)
        assert isinstance(result, ba.AnalysisResult)
        assert len(result.contingency_tables) == 3  # C(3,2)

    def test_with_outcome(self, df):
        result = ba.analyze(df, outcome="Y")
        assert len(result.contingency_tables) == 2  # A×Y, B×Y

    def test_metrics_dataframe(self, df):
        result = ba.analyze(df, outcome="Y")
        assert "pair" in result.metrics.columns
        assert "n" in result.metrics.columns
        assert len(result.metrics) == 2

    def test_bayesian_posteriors(self, df):
        result = ba.analyze(df, outcome="Y")
        assert result.posterior is not None
        assert len(result.posterior) == 2

    def test_no_bayesian(self, df):
        result = ba.analyze(df, outcome="Y", bayesian=False)
        assert result.posterior is None

    def test_warnings_generated(self, df):
        result = ba.analyze(df, outcome="Y")
        # n=10, should trigger small-sample warnings
        assert len(result.warnings) > 0

    def test_summary(self, df):
        result = ba.analyze(df, outcome="Y")
        s = result.summary()
        assert isinstance(s, pd.DataFrame)
        assert len(s) == 2

    def test_top_pairs(self, df):
        result = ba.analyze(df, outcome="Y")
        top = result.top_pairs(1)
        assert len(top) == 1

    def test_with_rules(self, df):
        result = ba.analyze(df, outcome="Y", rules=True)
        assert result.rules is not None

    def test_repr(self, df):
        result = ba.analyze(df, outcome="Y")
        r = repr(result)
        assert "AnalysisResult" in r
        assert "2 pairs" in r


class TestAnalyzeEdgeCases:
    def test_single_pair(self):
        df = pd.DataFrame({"X": [1, 0, 1, 0], "Y": [1, 1, 0, 0]})
        result = ba.analyze(df)
        assert len(result.contingency_tables) == 1

    def test_categorical_data(self):
        df = pd.DataFrame({
            "color": ["red", "blue", "red", "green", "blue"],
            "size": ["big", "small", "big", "small", "big"],
        })
        result = ba.analyze(df)
        assert len(result.contingency_tables) == 1
        key = list(result.contingency_tables.keys())[0]
        ct = result.contingency_tables[key]
        assert not ct.is_2x2  # 3×2 table
