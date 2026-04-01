"""Tests for ba.rules (encoding, itemsets, mining)."""

import pandas as pd
import pytest

from ba.rules.encoding import to_transactions
from ba.rules.itemsets import mine_itemsets
from ba.rules.mining import mine


# -- Encoding --------------------------------------------------------------


class TestEncoding:
    def test_binary_columns(self):
        df = pd.DataFrame({"A": [1, 0, 1], "B": [0, 1, 1]})
        trans = to_transactions(df)
        assert "A" in trans.columns
        assert "B" in trans.columns
        assert trans["A"].tolist() == [True, False, True]

    def test_categorical_columns(self):
        df = pd.DataFrame({"color": ["red", "blue", "red"]})
        trans = to_transactions(df)
        assert "color=red" in trans.columns
        assert "color=blue" in trans.columns
        assert trans["color=red"].tolist() == [True, False, True]

    def test_mixed(self):
        df = pd.DataFrame({"big": [1, 0, 1], "color": ["r", "b", "r"]})
        trans = to_transactions(df)
        assert "big" in trans.columns
        assert "color=r" in trans.columns

    def test_negation(self):
        df = pd.DataFrame({"A": [1, 0, 1]})
        trans = to_transactions(df, include_negation=True)
        assert "A" in trans.columns
        assert "~A" in trans.columns
        assert trans["~A"].tolist() == [False, True, False]


# -- Itemsets --------------------------------------------------------------


class TestItemsets:
    @pytest.fixture()
    def trans(self):
        return pd.DataFrame({
            "bread": [True, True, False, True, True],
            "milk": [True, False, True, True, True],
            "eggs": [False, True, True, False, True],
        })

    def test_basic(self, trans):
        result = mine_itemsets(trans, min_support=0.4)
        assert "itemsets" in result.columns
        assert "support" in result.columns
        assert len(result) > 0

    def test_min_support_filters(self, trans):
        loose = mine_itemsets(trans, min_support=0.2)
        strict = mine_itemsets(trans, min_support=0.8)
        assert len(loose) >= len(strict)

    def test_max_len(self, trans):
        result = mine_itemsets(trans, min_support=0.2, max_len=1)
        for _, row in result.iterrows():
            assert len(row["itemsets"]) <= 1

    def test_builtin_algorithm(self, trans):
        result = mine_itemsets(trans, min_support=0.4, algorithm="builtin")
        assert len(result) > 0

    def test_unknown_algorithm_raises(self, trans):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            mine_itemsets(trans, algorithm="nonexistent")


# -- Rule Mining -----------------------------------------------------------


class TestMining:
    @pytest.fixture()
    def df(self):
        return pd.DataFrame({
            "A": [1, 1, 0, 0, 1, 1, 0, 1],
            "B": [1, 0, 1, 0, 1, 1, 0, 0],
            "Y": [1, 1, 0, 0, 1, 1, 0, 0],
        })

    def test_basic(self, df):
        rules = mine(df, min_support=0.2, min_confidence=0.5)
        assert "antecedents" in rules.columns
        assert "consequents" in rules.columns
        assert "confidence" in rules.columns

    def test_bayesian_ci(self, df):
        rules = mine(df, min_support=0.2, min_confidence=0.5, bayesian=True)
        if not rules.empty:
            assert "confidence_ci_low" in rules.columns
            assert "confidence_ci_high" in rules.columns

    def test_no_bayesian(self, df):
        rules = mine(df, min_support=0.2, min_confidence=0.5, bayesian=False)
        if not rules.empty:
            assert "confidence_ci_low" not in rules.columns

    def test_outcome_constraint(self, df):
        rules = mine(df, min_support=0.2, min_confidence=0.3, outcome="Y")
        for _, row in rules.iterrows():
            assert any("Y" in str(c) for c in row["consequents"])

    def test_empty_result(self):
        df = pd.DataFrame({"A": [1, 0], "B": [0, 1]})
        rules = mine(df, min_support=0.9, min_confidence=0.9)
        assert rules.empty or len(rules) == 0
