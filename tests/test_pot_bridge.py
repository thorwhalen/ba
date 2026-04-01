"""Tests for ba.core.pot — the Pot ↔ ContingencyTable bridge."""

import numpy as np
import pandas as pd
import pytest

from spyn.ppi.pot import Pot

from ba.core.contingency import ContingencyTable, ContingencyTable2x2
from ba.core.pot import to_contingency, from_contingency
from ba.store import DataStore


class TestFromContingency:
    def test_2x2(self):
        ct = ContingencyTable.from_counts(10, 5, 3, 12)
        pot = from_contingency(ct)
        assert set(pot.vars) == {'X', 'Y'}
        assert pot.n == 4
        # Labels '1','0' are converted to ints 1,0 by _try_numeric
        assert int(pot.pval_of({'X': 1, 'Y': 1})) == 10

    def test_rxc(self):
        ct = ContingencyTable(
            counts=np.array([[5, 3, 2], [1, 4, 5]]),
            row_var='treatment',
            col_var='outcome',
            row_labels=('drug', 'placebo'),
            col_labels=('good', 'fair', 'poor'),
        )
        pot = from_contingency(ct)
        assert set(pot.vars) == {'treatment', 'outcome'}
        assert pot.n == 6
        assert int(pot.pval_of({'treatment': 'drug', 'outcome': 'good'})) == 5

    def test_preserves_total(self):
        ct = ContingencyTable.from_counts(7, 3, 2, 8)
        pot = from_contingency(ct)
        assert int(pot.values.sum()) == ct.n


class TestToContingency:
    def test_2x2(self):
        p = Pot({'X': [0, 0, 1, 1], 'Y': [0, 1, 0, 1], 'pval': [10, 5, 3, 12]})
        ct = to_contingency(p, 'X', 'Y')
        assert isinstance(ct, ContingencyTable2x2)
        assert ct.n == 30
        assert ct.counts[0, 0] == 10

    def test_rxc(self):
        p = Pot({
            'treatment': ['drug', 'drug', 'placebo', 'placebo'],
            'outcome': ['good', 'bad', 'good', 'bad'],
            'pval': [8, 2, 3, 7],
        })
        ct = to_contingency(p, 'treatment', 'outcome')
        assert ct.n == 20

    def test_projects_extra_vars(self):
        """If pot has more than 2 vars, project to the requested pair."""
        p = Pot({
            'A': [0, 0, 0, 0, 1, 1, 1, 1],
            'B': [0, 0, 1, 1, 0, 0, 1, 1],
            'C': [0, 1, 0, 1, 0, 1, 0, 1],
            'pval': [1, 2, 3, 4, 5, 6, 7, 8],
        })
        ct = to_contingency(p, 'A', 'B')
        assert ct.n == 36  # sum of all pvals
        assert ct.shape == (2, 2)


class TestRoundTrip:
    def test_2x2_roundtrip(self):
        ct_orig = ContingencyTable.from_counts(10, 5, 3, 12)
        pot = from_contingency(ct_orig)
        ct_back = to_contingency(pot, ct_orig.row_var, ct_orig.col_var)
        np.testing.assert_array_equal(ct_back.counts, ct_orig.counts)

    def test_contingency_to_pot_method(self):
        ct = ContingencyTable.from_counts(4, 6, 2, 8)
        pot = ct.to_pot()
        assert pot.n == 4
        assert int(pot.values.sum()) == 20


class TestDataStorePot:
    def test_pot_from_store(self):
        df = pd.DataFrame({
            'A': [1, 1, 0, 0, 1],
            'B': [1, 0, 1, 0, 1],
        })
        store = DataStore(df)
        p = store.pot('A', 'B')
        assert isinstance(p, Pot)
        assert set(p.vars) == {'A', 'B'}
        assert int(p.values.sum()) == 5

    def test_pot_single_var(self):
        df = pd.DataFrame({'X': [0, 1, 1, 0, 1]})
        store = DataStore(df)
        p = store.pot('X')
        assert p.n == 2
        assert int(p.pval_of({'X': 1})) == 3

    def test_pot_algebra_on_store_data(self):
        """Full workflow: DataStore → Pot → algebra → ContingencyTable."""
        df = pd.DataFrame({
            'treatment': [1, 1, 1, 0, 0, 0, 1, 0],
            'outcome':   [1, 1, 0, 0, 0, 1, 1, 0],
        })
        store = DataStore(df)
        joint = store.pot('treatment', 'outcome')

        # Use Pot algebra: P(outcome | treatment)
        conditional = joint / 'treatment'
        assert conditional.n == 4  # 2x2

        # Convert back to ContingencyTable for metrics
        ct = to_contingency(joint, 'treatment', 'outcome')
        assert ct.is_2x2
        assert ct.n == 8
