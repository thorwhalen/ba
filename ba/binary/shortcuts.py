"""Convenience functions for 2×2 table analysis."""

from __future__ import annotations

from ba.bayesian.posteriors import posterior, BayesianResult
from ba.core.contingency import ContingencyTable, ContingencyTable2x2


def _ensure_2x2(ct: ContingencyTable) -> ContingencyTable2x2:
    """Validate and convert to 2×2."""
    return ct.as_2x2()


def odds_ratio(ct: ContingencyTable) -> float | None:
    """Odds ratio: ad / bc.

    >>> from ba.core.contingency import ContingencyTable
    >>> odds_ratio(ContingencyTable.from_counts(10, 5, 3, 12))
    8.0
    """
    return _ensure_2x2(ct).odds_ratio


def relative_risk(ct: ContingencyTable) -> float | None:
    """Relative risk: (a/(a+b)) / (c/(c+d)).

    >>> from ba.core.contingency import ContingencyTable
    >>> rr = relative_risk(ContingencyTable.from_counts(10, 5, 3, 12))
    >>> round(rr, 3)
    3.333
    """
    return _ensure_2x2(ct).relative_risk


def risk_difference(ct: ContingencyTable) -> float:
    """Risk difference: a/(a+b) - c/(c+d).

    >>> from ba.core.contingency import ContingencyTable
    >>> round(risk_difference(ContingencyTable.from_counts(10, 5, 3, 12)), 3)
    0.467
    """
    return _ensure_2x2(ct).risk_difference


def phi(ct: ContingencyTable) -> float:
    """Phi coefficient.

    >>> from ba.core.contingency import ContingencyTable
    >>> round(phi(ContingencyTable.from_counts(10, 5, 3, 12)), 3)
    0.471
    """
    return _ensure_2x2(ct).phi


def yules_q(ct: ContingencyTable) -> float | None:
    """Yule's Q: (ad - bc) / (ad + bc).

    >>> from ba.core.contingency import ContingencyTable
    >>> round(yules_q(ContingencyTable.from_counts(10, 5, 3, 12)), 3)
    0.778
    """
    return _ensure_2x2(ct).yules_q
