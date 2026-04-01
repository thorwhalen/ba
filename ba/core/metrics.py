"""Extensible registry of association/interestingness measures."""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
from scipy import stats as scipy_stats


class MeasureRegistry:
    """Registry of named measures that can be computed on contingency tables.

    Each measure declares its requirements (e.g. requires_2x2) and provides
    a compute function that takes a ContingencyTable and returns a float.

    >>> from ba.core.metrics import registry
    >>> from ba.core.contingency import ContingencyTable
    >>> ct = ContingencyTable.from_counts(10, 5, 3, 12)
    >>> result = registry.compute(ct, ['support', 'lift', 'phi'])
    >>> 'phi' in result
    True
    """

    def __init__(self):
        self._measures: dict[str, _MeasureEntry] = {}

    def register(
        self,
        name: str,
        func: Callable,
        *,
        requires_2x2: bool = False,
        requires_ordinal: bool = False,
        description: str = "",
    ):
        """Register a new measure."""
        self._measures[name] = _MeasureEntry(
            func=func,
            requires_2x2=requires_2x2,
            requires_ordinal=requires_ordinal,
            description=description,
        )

    def compute(
        self,
        ct,
        measures: list[str] | str = "default",
    ) -> dict[str, Any]:
        """Compute requested measures on a contingency table.

        'default' computes all compatible measures. 'all' includes even those
        that require shape/type conditions not met (with warnings).
        """
        if isinstance(measures, str):
            if measures == "default" or measures == "all":
                names = list(self._measures)
            else:
                names = [measures]
        else:
            names = list(measures)

        result = {}
        for name in names:
            entry = self._measures.get(name)
            if entry is None:
                warnings.warn(
                    f"Unknown measure '{name}'. "
                    f"Available: {', '.join(sorted(self._measures))}",
                    stacklevel=2,
                )
                continue
            if entry.requires_2x2 and not ct.is_2x2:
                if measures != "default":
                    warnings.warn(
                        f"Measure '{name}' requires a 2×2 table but got "
                        f"{ct.shape}. Skipping. Use ba.qca.calibrate() to "
                        f"binarize, or use r×c measures.",
                        stacklevel=2,
                    )
                continue
            try:
                result[name] = entry.func(ct)
            except (ZeroDivisionError, ValueError):
                result[name] = None
        return result

    def available(self, ct=None) -> list[str]:
        """List measures, optionally filtered by compatibility with ct."""
        if ct is None:
            return sorted(self._measures)
        return sorted(
            name
            for name, entry in self._measures.items()
            if not (entry.requires_2x2 and not ct.is_2x2)
        )

    def describe(self, name: str) -> str:
        """Return the description of a measure."""
        entry = self._measures.get(name)
        if entry is None:
            raise KeyError(f"Unknown measure: {name}")
        return entry.description

    def __contains__(self, name: str) -> bool:
        return name in self._measures

    def __len__(self) -> int:
        return len(self._measures)


class _MeasureEntry:
    __slots__ = ("func", "requires_2x2", "requires_ordinal", "description")

    def __init__(self, func, requires_2x2, requires_ordinal, description):
        self.func = func
        self.requires_2x2 = requires_2x2
        self.requires_ordinal = requires_ordinal
        self.description = description


# ---------------------------------------------------------------------------
# Built-in measures
# ---------------------------------------------------------------------------

def _support(ct) -> float:
    """P(X ∩ Y) = a / n (top-left cell proportion)."""
    return float(ct.counts[0, 0] / ct.n)


def _confidence(ct) -> float:
    """P(Y|X) = a / (a + b) (row 0 proportion for col 0)."""
    row_total = ct.counts[0].sum()
    return float(ct.counts[0, 0] / row_total) if row_total > 0 else 0.0


def _lift(ct) -> float:
    """P(Y|X) / P(Y) = confidence / col_margin proportion."""
    conf = _confidence(ct)
    p_y = ct.col_margins[0] / ct.n
    return float(conf / p_y) if p_y > 0 else 0.0


def _conviction(ct) -> float | None:
    """P(X)P(~Y) / P(X ∩ ~Y)."""
    p_x = ct.counts[0].sum() / ct.n
    p_not_y = ct.col_margins[1] / ct.n if ct.shape[1] > 1 else 0.0
    p_x_not_y = ct.counts[0, 1] / ct.n if ct.shape[1] > 1 else 0.0
    if p_x_not_y == 0:
        return None  # perfect rule
    return float((p_x * p_not_y) / p_x_not_y)


def _leverage(ct) -> float:
    """P(X ∩ Y) - P(X)P(Y)."""
    p_xy = ct.counts[0, 0] / ct.n
    p_x = ct.counts[0].sum() / ct.n
    p_y = ct.col_margins[0] / ct.n
    return float(p_xy - p_x * p_y)


def _cosine(ct) -> float:
    """Ochiai coefficient: a / sqrt(row_0_total * col_0_total)."""
    row_total = ct.counts[0].sum()
    col_total = ct.col_margins[0]
    denom = np.sqrt(row_total * col_total)
    return float(ct.counts[0, 0] / denom) if denom > 0 else 0.0


def _jaccard(ct) -> float:
    """a / (a + b + c) for 2x2; a / (row0 + col0 - a) general."""
    a = ct.counts[0, 0]
    row_total = ct.counts[0].sum()
    col_total = ct.col_margins[0]
    denom = row_total + col_total - a
    return float(a / denom) if denom > 0 else 0.0


def _kulczynski(ct) -> float:
    """Average of the two directional confidences."""
    row_total = ct.counts[0].sum()
    col_total = ct.col_margins[0]
    a = ct.counts[0, 0]
    c1 = a / row_total if row_total > 0 else 0.0
    c2 = a / col_total if col_total > 0 else 0.0
    return float((c1 + c2) / 2)


def _chi_squared(ct) -> float:
    """Pearson chi-squared statistic."""
    chi2, _, _, _ = scipy_stats.chi2_contingency(ct.counts, correction=False)
    return float(chi2)


def _cramers_v(ct) -> float:
    """Cramér's V = sqrt(chi2 / (n * (min(r,c) - 1)))."""
    chi2 = _chi_squared(ct)
    k = min(ct.shape) - 1
    if k == 0:
        return 0.0
    return float(np.sqrt(chi2 / (ct.n * k)))


def _mutual_info(ct) -> float:
    """Mutual information I(X;Y) in nats."""
    p = ct.counts / ct.n
    p_row = ct.row_margins / ct.n
    p_col = ct.col_margins / ct.n
    mi = 0.0
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            if p[i, j] > 0:
                mi += p[i, j] * np.log(p[i, j] / (p_row[i] * p_col[j]))
    return float(mi)


def _g_test(ct) -> float:
    """G-test (log-likelihood ratio) statistic."""
    g, _, _, _ = scipy_stats.chi2_contingency(ct.counts, lambda_="log-likelihood")
    return float(g)


def _fisher_p(ct) -> float:
    """Fisher's exact test p-value (2×2 only; Freeman-Halton for r×c)."""
    if ct.is_2x2:
        _, p = scipy_stats.fisher_exact(ct.counts)
        return float(p)
    # For r×c, use chi2 p-value as fallback (Fisher-Freeman-Halton not in scipy)
    _, p, _, _ = scipy_stats.chi2_contingency(ct.counts)
    return float(p)


def _goodman_kruskal_gamma(ct) -> float:
    """Goodman-Kruskal γ: (C - D) / (C + D) over concordant/discordant pairs.

    Generalizes Yule's Q to ordinal r×c tables. For 2×2, γ = Q.

    >>> from ba.core.contingency import ContingencyTable
    >>> ct = ContingencyTable.from_counts(10, 5, 3, 12)
    >>> round(_goodman_kruskal_gamma(ct), 4)
    0.7778
    """
    counts = ct.counts
    r, c = counts.shape
    concordant = 0.0
    discordant = 0.0
    for i in range(r):
        for j in range(c):
            n_ij = counts[i, j]
            if n_ij == 0:
                continue
            # Sum cells below-right (concordant)
            for i2 in range(i + 1, r):
                for j2 in range(j + 1, c):
                    concordant += n_ij * counts[i2, j2]
            # Sum cells below-left (discordant)
            for i2 in range(i + 1, r):
                for j2 in range(0, j):
                    discordant += n_ij * counts[i2, j2]
    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return float((concordant - discordant) / denom)


def _uncertainty_coefficient(ct) -> float:
    """Theil's U = I(X;Y) / H(Y)."""
    mi = _mutual_info(ct)
    p_col = ct.col_margins / ct.n
    h_y = -float(np.sum(p_col[p_col > 0] * np.log(p_col[p_col > 0])))
    return float(mi / h_y) if h_y > 0 else 0.0


# Binary-only measures (require ct to be ContingencyTable2x2)

def _odds_ratio(ct) -> float | None:
    return ct.as_2x2().odds_ratio


def _relative_risk(ct) -> float | None:
    return ct.as_2x2().relative_risk


def _risk_difference(ct) -> float:
    return ct.as_2x2().risk_difference


def _phi(ct) -> float:
    return ct.as_2x2().phi


def _yules_q(ct) -> float | None:
    return ct.as_2x2().yules_q


def _qca_consistency(ct) -> float:
    return ct.as_2x2().qca_consistency


def _qca_coverage(ct) -> float:
    return ct.as_2x2().qca_coverage


# ---------------------------------------------------------------------------
# Global registry instance with built-in measures
# ---------------------------------------------------------------------------

registry = MeasureRegistry()

# General r×c measures
registry.register("support", _support, description="P(X ∩ Y) = a/n")
registry.register("confidence", _confidence, description="P(Y|X) = a/(a+b)")
registry.register("lift", _lift, description="P(Y|X)/P(Y)")
registry.register("conviction", _conviction, description="P(X)P(~Y)/P(X∩~Y)")
registry.register("leverage", _leverage, description="P(X∩Y) - P(X)P(Y)")
registry.register("cosine", _cosine, description="a / sqrt(row0 * col0)")
registry.register("jaccard", _jaccard, description="a / (row0 + col0 - a)")
registry.register("kulczynski", _kulczynski, description="mean of two confidences")
registry.register("chi_squared", _chi_squared, description="Pearson chi-squared")
registry.register("cramers_v", _cramers_v, description="sqrt(chi2 / (n*(min(r,c)-1)))")
registry.register("mutual_info", _mutual_info, description="I(X;Y) in nats")
registry.register(
    "goodman_kruskal_gamma",
    _goodman_kruskal_gamma,
    description="(C-D)/(C+D) concordant/discordant pairs",
)
registry.register("g_test", _g_test, description="G-test (log-likelihood ratio)")
registry.register("fisher_p", _fisher_p, description="Fisher exact / chi2 p-value")
registry.register(
    "uncertainty_coefficient",
    _uncertainty_coefficient,
    description="Theil's U = I(X;Y)/H(Y)",
)

# Binary-only measures
registry.register(
    "odds_ratio", _odds_ratio, requires_2x2=True, description="ad/bc"
)
registry.register(
    "relative_risk", _relative_risk, requires_2x2=True, description="(a/(a+b))/(c/(c+d))"
)
registry.register(
    "risk_difference", _risk_difference, requires_2x2=True, description="a/(a+b) - c/(c+d)"
)
registry.register(
    "phi", _phi, requires_2x2=True, description="(ad-bc)/sqrt((a+b)(c+d)(a+c)(b+d))"
)
registry.register(
    "yules_q", _yules_q, requires_2x2=True, description="(ad-bc)/(ad+bc)"
)
registry.register(
    "qca_consistency",
    _qca_consistency,
    requires_2x2=True,
    description="a/(a+b) — sufficiency consistency",
)
registry.register(
    "qca_coverage",
    _qca_coverage,
    requires_2x2=True,
    description="a/(a+c) — sufficiency coverage",
)
