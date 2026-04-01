"""Calibration: transform categorical/numerical data to binary for QCA."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


def calibrate(
    df: pd.DataFrame,
    thresholds: dict[str, float | str | Callable],
) -> pd.DataFrame:
    """Binarize columns using explicit thresholds.

    Each key in ``thresholds`` is a column name; the value specifies the
    binarization rule:

    - **float/int**: ``value >= threshold`` → 1, else 0.
    - ``'any_present'``: any truthy / non-null / non-zero → 1.
    - ``'median'``: ``value >= median`` → 1.
    - **callable**: applied per-value, must return bool-like.

    Columns not in ``thresholds`` are passed through unchanged.

    >>> import pandas as pd
    >>> df = pd.DataFrame({'age': [25, 35, 45], 'ill': ['no', 'yes', 'no']})
    >>> calibrate(df, {'age': 30, 'ill': 'any_present'}).values.tolist()
    [[0, 0], [1, 1], [1, 0]]
    """
    result = df.copy()
    for col, rule in thresholds.items():
        if col not in result.columns:
            raise KeyError(
                f"Column '{col}' not found in DataFrame. "
                f"Available: {list(result.columns)}"
            )
        result[col] = _apply_rule(result[col], rule)
    return result


def _apply_rule(series: pd.Series, rule: Any) -> pd.Series:
    if callable(rule) and not isinstance(rule, str):
        return series.map(rule).astype(int)

    if isinstance(rule, str):
        rule_lower = rule.lower().strip()
        if rule_lower == "any_present":
            return _any_present(series)
        if rule_lower == "median":
            med = series.median()
            return (series >= med).astype(int)
        raise ValueError(
            f"Unknown string rule '{rule}'. Use 'any_present', 'median', "
            f"a numeric threshold, or a callable."
        )

    # Numeric threshold
    return (series >= rule).astype(int)


def _any_present(series: pd.Series) -> pd.Series:
    """Truthy/non-null/non-zero/non-'no'/non-'none' → 1."""
    def _is_present(val):
        if pd.isna(val):
            return 0
        if isinstance(val, str):
            return int(val.lower().strip() not in ("", "no", "none", "false", "0", "n/a", "na"))
        return int(bool(val))

    return series.map(_is_present).astype(int)
