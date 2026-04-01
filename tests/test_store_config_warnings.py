"""Tests for ba.store, ba.config, ba.warnings."""

import pandas as pd
import pytest

from ba.store import DataStore, VarNamespace
from ba.config import Config
from ba.warnings import (
    check_table,
    check_data_weight,
    check_categorical_sparsity,
    check_truth_table_row,
)
from ba.core.contingency import ContingencyTable


# -- DataStore -------------------------------------------------------------


class TestDataStore:
    @pytest.fixture()
    def store(self):
        df = pd.DataFrame({
            "treatment": [1, 1, 1, 0, 0, 0],
            "outcome": [1, 1, 0, 0, 0, 1],
            "age_group": [1, 0, 1, 0, 1, 0],
        })
        return DataStore(df)

    def test_vars_attribute_access(self, store):
        assert store.vars.treatment == "treatment"
        assert store.vars.outcome == "outcome"

    def test_vars_unknown_raises(self, store):
        with pytest.raises(AttributeError):
            store.vars.nonexistent

    def test_vars_dir(self, store):
        assert "treatment" in dir(store.vars)

    def test_vars_is_binary(self, store):
        assert store.vars.is_binary("treatment")

    def test_vars_n_levels(self, store):
        assert store.vars.n_levels("treatment") == 2

    def test_contingency(self, store):
        ct = store.contingency("treatment", "outcome")
        assert ct.n == 6

    def test_contingency_cached(self, store):
        ct1 = store.contingency("treatment", "outcome")
        ct2 = store.contingency("treatment", "outcome")
        assert ct1 is ct2

    def test_all_pairs(self, store):
        pairs = store.all_pairs()
        # 3 choose 2 = 3 pairs
        assert len(pairs) == 3

    def test_all_pairs_with_outcome(self, store):
        pairs = store.all_pairs(outcome="outcome")
        # 2 pairs: treatment×outcome, age_group×outcome
        assert len(pairs) == 2
        for key in pairs:
            assert "outcome" in key

    def test_n_and_columns(self, store):
        assert store.n == 6
        assert store.columns == ["treatment", "outcome", "age_group"]


class TestVarNamespace:
    def test_binary_filter(self):
        df = pd.DataFrame({"a": [1, 0, 1], "b": [1, 2, 3], "c": [1, 0, 0]})
        ns = VarNamespace(df)
        assert set(ns.binary()) == {"a", "c"}

    def test_all(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        ns = VarNamespace(df)
        assert ns.all() == ["x", "y"]


# -- Config ----------------------------------------------------------------


class TestConfig:
    def test_get_default(self):
        cfg = Config()
        assert cfg["stats.ci_prob"] == 0.95

    def test_set(self):
        cfg = Config()
        cfg["stats.ci_prob"] = 0.99
        assert cfg["stats.ci_prob"] == 0.99

    def test_context_manager(self):
        cfg = Config()
        with cfg.context({"stats.ci_prob": 0.50}):
            assert cfg["stats.ci_prob"] == 0.50
        assert cfg["stats.ci_prob"] == 0.95

    def test_nested_context(self):
        cfg = Config()
        with cfg.context({"stats.ci_prob": 0.80}):
            assert cfg["stats.ci_prob"] == 0.80
            with cfg.context({"stats.ci_prob": 0.60}):
                assert cfg["stats.ci_prob"] == 0.60
            assert cfg["stats.ci_prob"] == 0.80
        assert cfg["stats.ci_prob"] == 0.95

    def test_reset(self):
        cfg = Config()
        cfg["stats.ci_prob"] = 0.50
        cfg.reset()
        assert cfg["stats.ci_prob"] == 0.95

    def test_contains(self):
        cfg = Config()
        assert "stats.ci_prob" in cfg
        assert "nonexistent" not in cfg

    def test_to_dict(self):
        cfg = Config()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "stats.ci_prob" in d


# -- Warnings --------------------------------------------------------------


class TestWarnings:
    def test_zero_cell(self):
        ct = ContingencyTable.from_counts(5, 0, 3, 7)
        warns = check_table(ct)
        assert any("Zero cell" in w for w in warns)

    def test_low_expected(self):
        ct = ContingencyTable.from_counts(1, 1, 1, 1)
        warns = check_table(ct)
        assert any("expected" in w.lower() for w in warns)

    def test_small_n(self):
        ct = ContingencyTable.from_counts(3, 2, 2, 3)
        warns = check_table(ct)
        assert any("Small sample" in w for w in warns)

    def test_no_warnings_for_large_balanced(self):
        ct = ContingencyTable.from_counts(50, 50, 50, 50)
        warns = check_table(ct)
        # No zero cells, expected counts = 50, n = 200
        assert len(warns) == 0

    def test_data_weight_dominated(self):
        warns = check_data_weight(0.3)
        assert any("Prior-dominated" in w for w in warns)

    def test_data_weight_influenced(self):
        warns = check_data_weight(0.7)
        assert any("Prior-influenced" in w for w in warns)

    def test_data_weight_ok(self):
        warns = check_data_weight(0.95)
        assert len(warns) == 0

    def test_categorical_sparsity(self):
        warns = check_categorical_sparsity(13, 5)
        assert len(warns) == 1

    def test_truth_table_row(self):
        warns = check_truth_table_row(1)
        assert len(warns) == 1
