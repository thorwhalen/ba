"""Contingency tables: the universal input for all ba analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclass
class ContingencyTable:
    """An r×c contingency table with metadata.

    The universal input for all analyses in ba. Wraps an r×c numpy array of
    integer counts with variable names and level labels.

    >>> ct = ContingencyTable.from_dataframe(
    ...     pd.DataFrame({'X': [1,1,0,0,1], 'Y': [1,0,1,0,1]}), 'X', 'Y'
    ... )
    >>> ct.n
    5
    >>> ct.is_2x2
    True
    """

    counts: np.ndarray
    row_var: str
    col_var: str
    row_labels: tuple[str, ...] = ()
    col_labels: tuple[str, ...] = ()

    def __post_init__(self):
        self.counts = np.asarray(self.counts, dtype=int)
        if self.counts.ndim != 2:
            raise ValueError(f"counts must be 2D, got {self.counts.ndim}D")
        if not self.row_labels:
            self.row_labels = tuple(str(i) for i in range(self.counts.shape[0]))
        if not self.col_labels:
            self.col_labels = tuple(str(i) for i in range(self.counts.shape[1]))

    # -- Computed properties ---------------------------------------------------

    @property
    def n(self) -> int:
        """Grand total of all counts."""
        return int(self.counts.sum())

    @property
    def shape(self) -> tuple[int, int]:
        return self.counts.shape

    @property
    def row_margins(self) -> np.ndarray:
        return self.counts.sum(axis=1)

    @property
    def col_margins(self) -> np.ndarray:
        return self.counts.sum(axis=0)

    @property
    def expected(self) -> np.ndarray:
        """Expected counts under independence."""
        return np.outer(self.row_margins, self.col_margins) / self.n

    @property
    def is_2x2(self) -> bool:
        return self.counts.shape == (2, 2)

    @property
    def has_zero_cell(self) -> bool:
        return bool((self.counts == 0).any())

    @property
    def min_cell(self) -> int:
        return int(self.counts.min())

    @property
    def min_expected(self) -> float:
        return float(self.expected.min())

    # -- Conversion ------------------------------------------------------------

    def as_2x2(self) -> ContingencyTable2x2:
        """Convert to 2×2 specialization.

        Raises ValueError with guidance if the table is not 2×2.
        """
        if not self.is_2x2:
            raise ValueError(
                f"Table is {self.counts.shape}, not 2×2. "
                "Use ba.qca.calibrate() to binarize, or use r×c metrics."
            )
        a, b = int(self.counts[0, 0]), int(self.counts[0, 1])
        c, d = int(self.counts[1, 0]), int(self.counts[1, 1])
        return ContingencyTable2x2(
            a=a,
            b=b,
            c=c,
            d=d,
            row_var=self.row_var,
            col_var=self.col_var,
            row_labels=self.row_labels,
            col_labels=self.col_labels,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Labeled DataFrame representation of the table."""
        return pd.DataFrame(
            self.counts,
            index=pd.Index(self.row_labels, name=self.row_var),
            columns=pd.Index(self.col_labels, name=self.col_var),
        )

    def to_pot(self):
        """Convert to a spyn Pot (count-based). Requires spyn.

        >>> ct = ContingencyTable.from_counts(10, 5, 3, 12)
        >>> pot = ct.to_pot()
        >>> pot.n
        4
        """
        from ba.core.pot import from_contingency

        return from_contingency(self)

    def metrics(self, measures: list[str] | str = "default") -> dict[str, Any]:
        """Compute metrics via the global measure registry."""
        from ba.core.metrics import registry

        return registry.compute(self, measures)

    def summary(self) -> pd.DataFrame:
        """One-row-per-metric summary DataFrame."""
        m = self.metrics()
        return pd.DataFrame(
            [{"measure": k, "value": v} for k, v in m.items()]
        )

    # -- Construction ----------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        row_var: str,
        col_var: str,
    ) -> ContingencyTable:
        """Cross-tabulate two columns of a DataFrame.

        Returns a ContingencyTable2x2 if both variables have exactly 2 levels.
        """
        ct = pd.crosstab(df[row_var], df[col_var])
        row_labels = tuple(str(v) for v in ct.index)
        col_labels = tuple(str(v) for v in ct.columns)
        counts = ct.values

        table = cls(
            counts=counts,
            row_var=row_var,
            col_var=col_var,
            row_labels=row_labels,
            col_labels=col_labels,
        )
        if table.is_2x2:
            return table.as_2x2()
        return table

    @classmethod
    def from_counts(
        cls,
        a: int,
        b: int,
        c: int,
        d: int,
        *,
        row_var: str = "X",
        col_var: str = "Y",
        row_labels: tuple[str, str] = ("1", "0"),
        col_labels: tuple[str, str] = ("1", "0"),
    ) -> ContingencyTable2x2:
        """Construct a 2×2 table directly from cell counts.

        Layout:
                 Y=1  Y=0
            X=1 [  a    b ]
            X=0 [  c    d ]

        >>> t = ContingencyTable.from_counts(10, 5, 3, 12)
        >>> t.odds_ratio
        8.0
        """
        return ContingencyTable2x2(
            a=a,
            b=b,
            c=c,
            d=d,
            row_var=row_var,
            col_var=col_var,
            row_labels=row_labels,
            col_labels=col_labels,
        )

    def __repr__(self) -> str:
        return (
            f"ContingencyTable({self.row_var}×{self.col_var}, "
            f"shape={self.shape}, n={self.n})"
        )


@dataclass
class ContingencyTable2x2(ContingencyTable):
    """A 2×2 contingency table with binary-specific metrics.

    Layout:
             col=1  col=0
        row=1 [  a    b ]    row_total = a + b
        row=0 [  c    d ]    row_total = c + d

    >>> t = ContingencyTable2x2(a=10, b=5, c=3, d=12)
    >>> t.odds_ratio
    8.0
    >>> t.phi  # doctest: +ELLIPSIS
    0.470...
    """

    a: int = 0
    b: int = 0
    c: int = 0
    d: int = 0

    def __init__(
        self,
        a: int = 0,
        b: int = 0,
        c: int = 0,
        d: int = 0,
        *,
        row_var: str = "X",
        col_var: str = "Y",
        row_labels: tuple[str, ...] = ("1", "0"),
        col_labels: tuple[str, ...] = ("1", "0"),
    ):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.row_var = row_var
        self.col_var = col_var
        self.counts = np.array([[a, b], [c, d]], dtype=int)
        self.row_labels = row_labels if len(row_labels) == 2 else ("1", "0")
        self.col_labels = col_labels if len(col_labels) == 2 else ("1", "0")

    # -- Binary-specific metrics -----------------------------------------------

    @property
    def odds_ratio(self) -> float | None:
        """ad / bc. None if b=0 or c=0."""
        if self.b == 0 or self.c == 0:
            return None
        return (self.a * self.d) / (self.b * self.c)

    @property
    def log_odds_ratio(self) -> float | None:
        """Natural log of the odds ratio. None if undefined."""
        or_val = self.odds_ratio
        if or_val is None or or_val <= 0:
            return None
        return float(np.log(or_val))

    @property
    def relative_risk(self) -> float | None:
        """a/(a+b) divided by c/(c+d). None if c+d=0 or c=0."""
        n1 = self.a + self.b
        n0 = self.c + self.d
        if n0 == 0 or n1 == 0:
            return None
        p1 = self.a / n1
        p0 = self.c / n0
        if p0 == 0:
            return None
        return p1 / p0

    @property
    def risk_difference(self) -> float:
        """a/(a+b) - c/(c+d)."""
        n1 = self.a + self.b
        n0 = self.c + self.d
        p1 = self.a / n1 if n1 > 0 else 0.0
        p0 = self.c / n0 if n0 > 0 else 0.0
        return p1 - p0

    @property
    def phi(self) -> float:
        """Phi coefficient: (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))."""
        num = self.a * self.d - self.b * self.c
        denom = np.sqrt(
            (self.a + self.b)
            * (self.c + self.d)
            * (self.a + self.c)
            * (self.b + self.d)
        )
        if denom == 0:
            return 0.0
        return float(num / denom)

    @property
    def yules_q(self) -> float | None:
        """(ad - bc) / (ad + bc). None if ad + bc = 0."""
        ad = self.a * self.d
        bc = self.b * self.c
        denom = ad + bc
        if denom == 0:
            return None
        return (ad - bc) / denom

    @property
    def qca_consistency(self) -> float:
        """Consistency of sufficiency: a / (a + b) = P(Y|X)."""
        n1 = self.a + self.b
        return self.a / n1 if n1 > 0 else 0.0

    @property
    def qca_coverage(self) -> float:
        """Coverage of sufficiency: a / (a + c) = P(X|Y)."""
        m1 = self.a + self.c
        return self.a / m1 if m1 > 0 else 0.0

    @property
    def fisher_p(self) -> float:
        """Two-sided Fisher's exact test p-value."""
        _, p = scipy_stats.fisher_exact(self.counts)
        return float(p)

    def as_2x2(self) -> ContingencyTable2x2:
        return self

    def __repr__(self) -> str:
        return (
            f"ContingencyTable2x2({self.row_var}×{self.col_var}, "
            f"a={self.a}, b={self.b}, c={self.c}, d={self.d}, n={self.n})"
        )
