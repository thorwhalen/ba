"""QCA truth table construction from binary data."""

from __future__ import annotations

import pandas as pd


def truth_table(
    data: pd.DataFrame,
    outcome: str,
    conditions: list[str],
    *,
    incl_cut: float = 0.8,
    n_cut: int = 1,
) -> pd.DataFrame:
    """Build a QCA truth table from binary data.

    Each row of the truth table is a unique combination of condition values.
    For each row, computes the number of cases (``n``), consistency
    (proportion with positive outcome), and flags rows below ``n_cut``.

    All condition columns and the outcome must be binary (0/1).

    Args:
        data: DataFrame with binary columns.
        outcome: Name of the outcome column.
        conditions: List of condition column names.
        incl_cut: Consistency threshold for marking rows as sufficient.
        n_cut: Minimum number of cases for a row to be included.

    Returns:
        DataFrame with condition columns, n, consistency, outcome assignment,
        and a ``flag`` column for rows with n <= 2.

    Raises:
        ValueError: If any condition or outcome column is not binary.

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1,1,1,0,0,0,1,0],
    ...     'B': [1,0,1,0,1,0,0,1],
    ...     'Y': [1,1,1,0,0,0,1,0],
    ... })
    >>> tt = truth_table(df, 'Y', ['A', 'B'])
    >>> sorted(tt.columns.tolist())
    ['A', 'B', 'OUT', 'consistency', 'flag', 'n']
    """
    _validate_binary(data, outcome, conditions)

    cols = conditions + [outcome]
    grouped = data[cols].groupby(conditions, dropna=False)

    rows = []
    for config, group in grouped:
        if not isinstance(config, tuple):
            config = (config,)
        n = len(group)
        n_positive = group[outcome].sum()
        consistency = n_positive / n if n > 0 else 0.0

        row = dict(zip(conditions, config))
        row["n"] = n
        row["consistency"] = round(consistency, 4)
        row["OUT"] = (
            1 if consistency >= incl_cut else (0 if consistency <= (1 - incl_cut) else "C")
        )
        row["flag"] = "low_n" if n <= 2 else ""
        rows.append(row)

    tt = pd.DataFrame(rows)
    # Filter by n_cut
    tt = tt[tt["n"] >= n_cut].reset_index(drop=True)
    return tt


def _validate_binary(
    data: pd.DataFrame, outcome: str, conditions: list[str]
) -> None:
    """Raise ValueError if any column is not binary."""
    all_cols = conditions + [outcome]
    non_binary = []
    for col in all_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")
        unique = set(data[col].dropna().unique())
        if not unique.issubset({0, 1, True, False}):
            non_binary.append(col)

    if non_binary:
        raise ValueError(
            f"QCA requires binary conditions (0/1). Non-binary columns: "
            f"{non_binary}. Use ba.qca.calibrate() to binarize first."
        )
