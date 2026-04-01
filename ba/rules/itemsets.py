"""Frequent itemset mining: built-in brute-force and optional mlxtend wrapper."""

from __future__ import annotations

from itertools import combinations

import pandas as pd


def mine_itemsets(
    transactions: pd.DataFrame,
    *,
    min_support: float = 0.1,
    max_len: int | None = None,
    algorithm: str = "auto",
) -> pd.DataFrame:
    """Find frequent itemsets in a boolean transaction DataFrame.

    Args:
        transactions: Boolean DataFrame (output of ``to_transactions``).
        min_support: Minimum support threshold (fraction of transactions).
        max_len: Maximum itemset length. None = no limit.
        algorithm: ``'auto'`` tries mlxtend fpgrowth first, falls back to
            brute-force. ``'builtin'`` forces brute-force (fine for n < 200).
            ``'fpgrowth'`` or ``'apriori'`` require mlxtend.

    Returns:
        DataFrame with columns ``itemsets`` (frozenset) and ``support``.

    >>> import pandas as pd
    >>> trans = pd.DataFrame({
    ...     'bread': [True, True, False, True],
    ...     'milk':  [True, False, True, True],
    ...     'eggs':  [False, True, True, False],
    ... })
    >>> result = mine_itemsets(trans, min_support=0.5)
    >>> len(result) > 0
    True
    >>> 'support' in result.columns
    True
    """
    if algorithm == "auto":
        try:
            return _mlxtend_fpgrowth(transactions, min_support, max_len)
        except ImportError:
            return _builtin_bruteforce(transactions, min_support, max_len)
    elif algorithm == "builtin":
        return _builtin_bruteforce(transactions, min_support, max_len)
    elif algorithm in ("fpgrowth", "apriori"):
        return _mlxtend_algo(transactions, min_support, max_len, algorithm)
    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Use 'auto', 'builtin', 'fpgrowth', or 'apriori'."
        )


def _builtin_bruteforce(
    transactions: pd.DataFrame,
    min_support: float,
    max_len: int | None,
) -> pd.DataFrame:
    """Simple brute-force itemset mining for small datasets."""
    n = len(transactions)
    items = list(transactions.columns)
    min_count = min_support * n

    if max_len is None:
        max_len = len(items)

    results = []

    for size in range(1, max_len + 1):
        for combo in combinations(items, size):
            # Count transactions where all items in combo are True
            mask = transactions[list(combo)].all(axis=1)
            count = mask.sum()
            if count >= min_count:
                results.append({
                    "itemsets": frozenset(combo),
                    "support": count / n,
                })

    return pd.DataFrame(results) if results else pd.DataFrame(columns=["itemsets", "support"])


def _mlxtend_fpgrowth(
    transactions: pd.DataFrame,
    min_support: float,
    max_len: int | None,
) -> pd.DataFrame:
    """Wrap mlxtend's fpgrowth."""
    from mlxtend.frequent_patterns import fpgrowth

    kwargs = {"min_support": min_support, "use_colnames": True}
    if max_len is not None:
        kwargs["max_len"] = max_len
    return fpgrowth(transactions, **kwargs)


def _mlxtend_algo(
    transactions: pd.DataFrame,
    min_support: float,
    max_len: int | None,
    algorithm: str,
) -> pd.DataFrame:
    """Wrap mlxtend's fpgrowth or apriori."""
    if algorithm == "fpgrowth":
        from mlxtend.frequent_patterns import fpgrowth as func
    else:
        from mlxtend.frequent_patterns import apriori as func

    kwargs = {"min_support": min_support, "use_colnames": True}
    if max_len is not None:
        kwargs["max_len"] = max_len
    return func(transactions, **kwargs)
