"""Boolean minimization via the Quine-McCluskey algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class QCASolution:
    """Result of Boolean minimization.

    >>> sol = QCASolution(
    ...     expression='A*B + ~A*C',
    ...     prime_implicants=[{'A': 1, 'B': 1}, {'A': 0, 'C': 1}],
    ...     essential_implicants=[{'A': 1, 'B': 1}],
    ...     coverage={'A*B': [0, 1], '~A*C': [2]},
    ... )
    >>> sol.expression
    'A*B + ~A*C'
    """

    expression: str
    prime_implicants: list[dict[str, int | None]]
    essential_implicants: list[dict[str, int | None]]
    coverage: dict[str, list[int]] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Prime implicant chart as a DataFrame."""
        return pd.DataFrame(self.prime_implicants)

    def __repr__(self) -> str:
        return f"QCASolution({self.expression})"


def minimize(
    tt: pd.DataFrame,
    *,
    outcome_col: str = "OUT",
    conditions: list[str] | None = None,
    include: str = "1",
    method: str = "qmc",
) -> QCASolution:
    """Minimize a truth table using Quine-McCluskey.

    Args:
        tt: Truth table (output of ``ba.qca.truth_table``).
        outcome_col: Column marking positive (1), negative (0), or
            don't-care ('C'/'?') rows.
        conditions: Condition column names. If None, inferred as all
            columns except ``outcome_col``, ``n``, ``consistency``, ``flag``.
        include: Which rows to include as positive:
            ``'1'`` = only positive rows;
            ``'?'`` or ``'C'`` = also include don't-care/remainder rows.
        method: ``'qmc'`` for Quine-McCluskey (only option currently).

    Returns:
        QCASolution with minimized Boolean expression.

    >>> import pandas as pd
    >>> tt = pd.DataFrame({
    ...     'A': [0, 0, 1, 1], 'B': [0, 1, 0, 1],
    ...     'OUT': [0, 1, 1, 1], 'n': [3, 2, 2, 4],
    ...     'consistency': [0.0, 1.0, 1.0, 1.0], 'flag': ['', '', '', ''],
    ... })
    >>> sol = minimize(tt)
    >>> 'A' in sol.expression or 'B' in sol.expression
    True
    """
    meta_cols = {outcome_col, "n", "consistency", "flag"}
    if conditions is None:
        conditions = [c for c in tt.columns if c not in meta_cols]

    # Collect minterms and don't-cares
    minterms = []
    dontcares = []

    for _, row in tt.iterrows():
        out = row[outcome_col]
        bits = tuple(int(row[c]) for c in conditions)
        idx = _bits_to_int(bits)
        if out == 1 or out == "1":
            minterms.append(idx)
        elif include in ("?", "C") and out in ("C", "?", "R"):
            dontcares.append(idx)

    if not minterms:
        return QCASolution(
            expression="0 (no positive rows)",
            prime_implicants=[],
            essential_implicants=[],
        )

    n_vars = len(conditions)
    primes = _quine_mccluskey(minterms, dontcares, n_vars)
    essentials, cover = _select_essential(primes, minterms)

    # Format expression
    terms = [_implicant_to_expr(imp, conditions) for imp in essentials]
    expression = " + ".join(terms) if terms else "0"

    # Convert implicants to dicts
    prime_dicts = [_implicant_to_dict(imp, conditions) for imp in primes]
    essential_dicts = [_implicant_to_dict(imp, conditions) for imp in essentials]

    return QCASolution(
        expression=expression,
        prime_implicants=prime_dicts,
        essential_implicants=essential_dicts,
        coverage=cover,
    )


# ---------------------------------------------------------------------------
# Quine-McCluskey internals
# ---------------------------------------------------------------------------


def _bits_to_int(bits: tuple[int, ...]) -> int:
    """Convert a tuple of bits to an integer. MSB first."""
    result = 0
    for b in bits:
        result = (result << 1) | b
    return result


def _count_ones(n: int) -> int:
    return bin(n).count("1")


def _quine_mccluskey(
    minterms: list[int],
    dontcares: list[int],
    n_vars: int,
) -> list[tuple[frozenset[int], tuple[int | None, ...]]]:
    """Core QMC: find all prime implicants.

    Each implicant is (covered_minterms, pattern) where pattern is a tuple
    of 0, 1, or None (don't-care bit).
    """
    all_terms = set(minterms) | set(dontcares)

    # Initialize: each minterm is a group-0 implicant
    groups: dict[int, list[tuple[frozenset[int], tuple[int | None, ...]]]] = {}
    for m in all_terms:
        ones = _count_ones(m)
        pattern = tuple((m >> (n_vars - 1 - i)) & 1 for i in range(n_vars))
        groups.setdefault(ones, []).append((frozenset([m]), pattern))

    primes: list[tuple[frozenset[int], tuple[int | None, ...]]] = []

    while groups:
        new_groups: dict[int, list[tuple[frozenset[int], tuple[int | None, ...]]]] = {}
        used = set()
        sorted_keys = sorted(groups.keys())

        for i in range(len(sorted_keys) - 1):
            k1 = sorted_keys[i]
            k2 = sorted_keys[i + 1]
            if k2 - k1 != 1:
                continue
            for idx1, (mints1, pat1) in enumerate(groups[k1]):
                for idx2, (mints2, pat2) in enumerate(groups[k2]):
                    combined = _try_combine(pat1, pat2)
                    if combined is not None:
                        used.add((k1, idx1))
                        used.add((k2, idx2))
                        merged_mints = mints1 | mints2
                        entry = (merged_mints, combined)
                        new_groups.setdefault(k1, []).append(entry)

        # Collect unused implicants as primes
        for k, imps in groups.items():
            for idx, imp in enumerate(imps):
                if (k, idx) not in used:
                    primes.append(imp)

        # Deduplicate new groups
        deduped: dict[int, list[tuple[frozenset[int], tuple[int | None, ...]]]] = {}
        for k, imps in new_groups.items():
            seen = set()
            for mints, pat in imps:
                if pat not in seen:
                    seen.add(pat)
                    deduped.setdefault(k, []).append((mints, pat))

        groups = deduped

    return primes


def _try_combine(
    pat1: tuple[int | None, ...],
    pat2: tuple[int | None, ...],
) -> tuple[int | None, ...] | None:
    """Combine two patterns differing in exactly one position."""
    diff_count = 0
    result = list(pat1)
    for i in range(len(pat1)):
        if pat1[i] != pat2[i]:
            if pat1[i] is None or pat2[i] is None:
                return None
            diff_count += 1
            result[i] = None
            if diff_count > 1:
                return None
    if diff_count != 1:
        return None
    return tuple(result)


def _select_essential(
    primes: list[tuple[frozenset[int], tuple[int | None, ...]]],
    minterms: list[int],
) -> tuple[
    list[tuple[frozenset[int], tuple[int | None, ...]]],
    dict[str, list[int]],
]:
    """Select essential prime implicants via a greedy cover."""
    remaining = set(minterms)
    selected = []
    coverage_map = {}

    # Find essential primes (only prime covering a minterm)
    for m in list(remaining):
        covering = [p for p in primes if m in p[0]]
        if len(covering) == 1:
            p = covering[0]
            if p not in selected:
                selected.append(p)
                remaining -= p[0]

    # Greedy cover for remaining minterms
    while remaining:
        best = max(primes, key=lambda p: len(p[0] & remaining))
        if not (best[0] & remaining):
            break
        selected.append(best)
        remaining -= best[0]

    # Build coverage dict
    for p in selected:
        expr = _implicant_to_expr(p[1], [f"v{i}" for i in range(len(p[1]))])
        coverage_map[expr] = sorted(p[0] & set(minterms))

    return selected, coverage_map


def _implicant_to_expr(
    pattern: tuple[int | None, ...], conditions: list[str]
) -> str:
    """Format a pattern tuple as a Boolean expression."""
    terms = []
    for i, val in enumerate(pattern):
        if val is None:
            continue
        name = conditions[i] if i < len(conditions) else f"v{i}"
        if val == 1:
            terms.append(name)
        else:
            terms.append(f"~{name}")
    return "*".join(terms) if terms else "1"


def _implicant_to_dict(
    imp: tuple[frozenset[int], tuple[int | None, ...]],
    conditions: list[str],
) -> dict[str, int | None]:
    """Convert implicant pattern to a {condition: value} dict."""
    _, pattern = imp
    return {
        conditions[i]: v
        for i, v in enumerate(pattern)
        if i < len(conditions)
    }
