"""Bridge between spyn's Pot and ba's ContingencyTable.

Enables round-tripping between the algebraic world (Pot operators: *, /, [])
and the metric-computation world (ContingencyTable with measure registry).

>>> from spyn.ppi.pot import Pot
>>> from ba.core.contingency import ContingencyTable
>>> ct = ContingencyTable.from_counts(10, 5, 3, 12)
>>> pot = from_contingency(ct)
>>> pot.n
4
>>> ct2 = to_contingency(pot, 'X', 'Y')
>>> ct2.n
30
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ba.core.contingency import ContingencyTable, ContingencyTable2x2


def _try_numeric(val):
    """Try to convert a string label to int or float."""
    try:
        i = int(val)
        if str(i) == str(val):
            return i
    except (ValueError, TypeError):
        pass
    try:
        return float(val)
    except (ValueError, TypeError):
        return val


def to_contingency(
    pot,
    row_var: str,
    col_var: str,
) -> ContingencyTable:
    """Convert a 2-variable Pot (count-based) to a ContingencyTable.

    The Pot must contain the two named variables. The pval column
    is interpreted as counts.

    Args:
        pot: A spyn Pot with variables including row_var and col_var.
        row_var: Name of the row variable.
        col_var: Name of the column variable.

    Returns:
        ContingencyTable (or ContingencyTable2x2 if both variables are binary).

    >>> from spyn.ppi.pot import Pot
    >>> p = Pot({'X': [0, 0, 1, 1], 'Y': [0, 1, 0, 1], 'pval': [10, 5, 3, 12]})
    >>> ct = to_contingency(p, 'X', 'Y')
    >>> ct.n
    30
    >>> ct.is_2x2
    True
    """
    from ba.core.contingency import ContingencyTable

    tb = pot.tb.copy()
    pot_vars = set(c for c in tb.columns if c != 'pval')

    if not {row_var, col_var}.issubset(pot_vars):
        raise ValueError(
            f"Pot does not contain variables {row_var!r} and {col_var!r}. "
            f"Pot vars: {sorted(pot_vars)}"
        )

    if pot_vars != {row_var, col_var}:
        projected = pot[[row_var, col_var]]
        tb = projected.tb.copy()

    # Get ordered unique values for each variable (preserving data order)
    row_vals = list(dict.fromkeys(tb[row_var]))
    col_vals = list(dict.fromkeys(tb[col_var]))

    # Build counts matrix preserving the order from the data
    counts = np.zeros((len(row_vals), len(col_vals)), dtype=int)
    row_idx = {v: i for i, v in enumerate(row_vals)}
    col_idx = {v: i for i, v in enumerate(col_vals)}

    for _, row in tb.iterrows():
        ri = row_idx[row[row_var]]
        ci = col_idx[row[col_var]]
        counts[ri, ci] += int(row['pval'])

    row_labels = tuple(str(v) for v in row_vals)
    col_labels = tuple(str(v) for v in col_vals)

    ct = ContingencyTable(
        counts=counts,
        row_var=row_var,
        col_var=col_var,
        row_labels=row_labels,
        col_labels=col_labels,
    )
    if ct.is_2x2:
        return ct.as_2x2()
    return ct


def from_contingency(ct: ContingencyTable) -> "Pot":
    """Convert a ContingencyTable to a spyn Pot (count-based).

    The resulting Pot has two variables (row_var, col_var) and pval = counts.
    Labels are converted to numeric types where possible.

    >>> from ba.core.contingency import ContingencyTable
    >>> ct = ContingencyTable.from_counts(10, 5, 3, 12)
    >>> pot = from_contingency(ct)
    >>> pot.vars
    ['X', 'Y']
    >>> int(pot.values.sum())
    30
    """
    from spyn.ppi.pot import Pot

    row_values = [_try_numeric(rl) for rl in ct.row_labels]
    col_values = [_try_numeric(cl) for cl in ct.col_labels]

    rv = []
    cv = []
    pv = []
    for i, rval in enumerate(row_values):
        for j, cval in enumerate(col_values):
            rv.append(rval)
            cv.append(cval)
            pv.append(int(ct.counts[i, j]))

    return Pot({ct.row_var: rv, ct.col_var: cv, 'pval': pv})
