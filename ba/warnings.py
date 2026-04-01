"""Small-sample and data-quality warning generators.

Warnings are returned as lists of strings, attached to result objects —
not printed to stderr.

>>> from ba.warnings import check_table
>>> from ba.core.contingency import ContingencyTable
>>> ct = ContingencyTable.from_counts(5, 0, 3, 7)
>>> warns = check_table(ct)
>>> any('zero' in w.lower() for w in warns)
True
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ba.core.contingency import ContingencyTable


def check_table(ct: ContingencyTable, *, small_n: int = 30) -> list[str]:
    """Generate warnings for a contingency table.

    >>> from ba.core.contingency import ContingencyTable
    >>> ct = ContingencyTable.from_counts(1, 0, 0, 2)
    >>> warns = check_table(ct)
    >>> len(warns) > 0
    True
    """
    warns: list[str] = []

    if ct.has_zero_cell:
        warns.append(
            f"Zero cell detected in {ct.row_var}×{ct.col_var} table. "
            f"Point estimates (OR, RR) may be undefined. "
            f"Use Bayesian posterior instead."
        )

    if ct.min_expected < 5:
        warns.append(
            f"Minimum expected count = {ct.min_expected:.1f} < 5 in "
            f"{ct.row_var}×{ct.col_var}. Chi-squared test is unreliable. "
            f"Use Fisher's exact test or Bayesian inference."
        )

    if ct.n < small_n:
        warns.append(
            f"Small sample (n={ct.n}) in {ct.row_var}×{ct.col_var}. "
            f"Point estimates have wide uncertainty. "
            f"Consider Bayesian credible intervals."
        )

    return warns


def check_data_weight(data_weight: float) -> list[str]:
    """Generate warnings about prior influence.

    >>> check_data_weight(0.4)
    ['Prior-dominated (data weight = 0.40). The prior contributes >50% to the posterior. Consider using a weaker prior or collecting more data.']
    """
    warns: list[str] = []
    if data_weight < 0.5:
        warns.append(
            f"Prior-dominated (data weight = {data_weight:.2f}). "
            f"The prior contributes >50% to the posterior. "
            f"Consider using a weaker prior or collecting more data."
        )
    elif data_weight < 0.8:
        warns.append(
            f"Prior-influenced (data weight = {data_weight:.2f}). "
            f"The prior contributes >20% to the posterior. "
            f"Run sensitivity analysis with ba.bayesian.sensitivity()."
        )
    return warns


def check_categorical_sparsity(n: int, n_levels: int) -> list[str]:
    """Warn if a categorical variable has too many levels for the sample size.

    >>> check_categorical_sparsity(13, 5)
    ['Variable has 5 levels with only n=13. Average cell count < 3. Consider collapsing levels or using Bayesian pooling.']
    """
    warns: list[str] = []
    avg_per_level = n / n_levels
    if avg_per_level < 3:
        warns.append(
            f"Variable has {n_levels} levels with only n={n}. "
            f"Average cell count < 3. "
            f"Consider collapsing levels or using Bayesian pooling."
        )
    return warns


def check_truth_table_row(n_cases: int) -> list[str]:
    """Warn about singleton truth table rows.

    >>> check_truth_table_row(1)
    ['Truth table row has only 1 case(s). Consistency is trivially 0 or 1 and unreliable.']
    """
    warns: list[str] = []
    if n_cases <= 2:
        warns.append(
            f"Truth table row has only {n_cases} case(s). "
            f"Consistency is trivially 0 or 1 and unreliable."
        )
    return warns
