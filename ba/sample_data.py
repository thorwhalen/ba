"""Small sample datasets for tutorials and doctests.

>>> df = custody_data()
>>> df.shape
(13, 7)
>>> df.columns.tolist()
['treatment', 'surveillance', 'physical_illness', 'mental_illness', 'illegal_drug', 'support_network', 'retained_custody']
"""

from __future__ import annotations

import pandas as pd


def custody_data() -> pd.DataFrame:
    """Synthetic small-sample dataset inspired by child custody research.

    13 cases, 6 binary conditions, 1 binary outcome. Designed to exercise
    all ba features: small-n warnings, zero cells, QCA truth tables,
    association rules, and Bayesian inference.

    >>> df = custody_data()
    >>> int(df['retained_custody'].sum())
    6
    >>> int(df['treatment'].sum())
    7
    """
    return pd.DataFrame({
        "treatment":        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        "surveillance":     [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
        "physical_illness":  [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
        "mental_illness":    [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0],
        "illegal_drug":      [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
        "support_network":   [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        "retained_custody":  [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    })


def market_basket() -> pd.DataFrame:
    """Small categorical transaction dataset for association rule mining.

    20 transactions, 4 items. Demonstrates categorical encoding and rule mining.

    >>> df = market_basket()
    >>> len(df)
    20
    """
    return pd.DataFrame({
        "bread":  [1,1,0,1,1,0,1,1,0,1, 1,0,1,1,0,1,0,1,1,0],
        "milk":   [1,0,1,1,1,1,0,1,1,0, 1,1,0,1,1,0,1,1,0,1],
        "eggs":   [0,1,1,0,1,0,1,0,1,1, 0,1,0,1,0,1,0,0,1,1],
        "butter": [1,0,0,1,1,0,0,1,0,0, 1,0,1,0,0,1,0,1,0,0],
    })
