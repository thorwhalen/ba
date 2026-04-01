"""Transaction encoding for association rule mining."""

from __future__ import annotations

import pandas as pd


def to_transactions(
    df: pd.DataFrame,
    *,
    binary_as_presence: bool = True,
    include_negation: bool = False,
) -> pd.DataFrame:
    """Convert a DataFrame to a boolean transaction DataFrame.

    Each variable-value pair becomes a column (item). Binary columns become
    presence items; categorical columns become one item per level.

    Args:
        df: Input DataFrame with binary or categorical columns.
        binary_as_presence: If True, binary columns map 1 → item present.
        include_negation: If True, add negated items for binary columns
            (e.g., ``~smoking`` when ``smoking=0``).

    Returns:
        Boolean DataFrame where each column is an item and each row is a
        transaction.

    >>> import pandas as pd
    >>> df = pd.DataFrame({'color': ['red', 'blue', 'red'], 'big': [1, 0, 1]})
    >>> trans = to_transactions(df)
    >>> sorted(trans.columns.tolist())
    ['big', 'color=blue', 'color=red']
    >>> trans['big'].tolist()
    [True, False, True]
    """
    result_cols: dict[str, pd.Series] = {}

    for col in df.columns:
        series = df[col]
        unique = series.dropna().unique()

        is_binary = set(unique).issubset({0, 1, True, False})

        if is_binary and binary_as_presence:
            result_cols[col] = series.astype(bool)
            if include_negation:
                result_cols[f"~{col}"] = ~series.astype(bool)
        else:
            # Categorical: one-hot encode
            for val in sorted(unique, key=str):
                item_name = f"{col}={val}"
                result_cols[item_name] = (series == val)

    return pd.DataFrame(result_cols)
