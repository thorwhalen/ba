"""Association rule mining with optional Bayesian augmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from ba.rules.encoding import to_transactions
from ba.rules.itemsets import mine_itemsets


def mine(
    data: pd.DataFrame,
    *,
    min_support: float = 0.1,
    min_confidence: float = 0.5,
    outcome: str | None = None,
    appearance: dict | None = None,
    algorithm: str = "auto",
    measures: tuple[str, ...] = ("support", "confidence", "lift"),
    max_len: int = 3,
    bayesian: bool = True,
    prior_alpha: float = 0.5,
    prior_beta: float = 0.5,
    ci_prob: float = 0.95,
) -> pd.DataFrame:
    """Mine association rules from a DataFrame.

    Args:
        data: Input DataFrame (binary or categorical columns).
        min_support: Minimum support threshold.
        min_confidence: Minimum confidence threshold.
        outcome: If given, constrain RHS to items containing this column name.
        appearance: Dict with keys ``'rhs'``, ``'lhs'``, ``'none'`` mapping
            to lists of item names for constrained mining.
        algorithm: Itemset mining algorithm (see ``mine_itemsets``).
        measures: Tuple of measure names to compute.
        max_len: Maximum rule length (LHS + RHS items).
        bayesian: If True, append credible interval columns.
        prior_alpha: Beta prior alpha for Bayesian CIs.
        prior_beta: Beta prior beta for Bayesian CIs.
        ci_prob: Credible interval probability.

    Returns:
        DataFrame with columns: antecedents, consequents, and requested
        measures (plus CI columns if bayesian=True).

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1,1,0,0,1,1,0,1],
    ...     'B': [1,0,1,0,1,1,0,0],
    ...     'Y': [1,1,0,0,1,1,0,0],
    ... })
    >>> rules = mine(df, min_support=0.2, min_confidence=0.5)
    >>> 'antecedents' in rules.columns
    True
    """
    trans = to_transactions(data)
    itemsets = mine_itemsets(trans, min_support=min_support, max_len=max_len, algorithm=algorithm)

    if itemsets.empty:
        return pd.DataFrame(columns=["antecedents", "consequents"] + list(measures))

    n = len(trans)
    # Build support lookup
    support_map: dict[frozenset, float] = {}
    for _, row in itemsets.iterrows():
        support_map[row["itemsets"]] = row["support"]

    rules_rows = []
    for _, row in itemsets.iterrows():
        itemset = row["itemsets"]
        if len(itemset) < 2:
            continue
        for item in itemset:
            consequent = frozenset([item])
            antecedent = itemset - consequent

            # Apply appearance constraints
            if appearance:
                if "rhs" in appearance and item not in appearance["rhs"]:
                    continue
                if "lhs" in appearance:
                    if not antecedent.issubset(set(appearance["lhs"])):
                        continue
                if "none" in appearance:
                    if antecedent & set(appearance["none"]) or consequent & set(appearance["none"]):
                        continue

            # Apply outcome constraint
            if outcome and not any(outcome in str(c) for c in consequent):
                continue

            s_xy = support_map.get(itemset, 0)
            s_x = support_map.get(antecedent, 0)
            s_y = support_map.get(consequent, 0)

            if s_x == 0:
                continue

            conf = s_xy / s_x
            if conf < min_confidence:
                continue

            rule_row: dict = {
                "antecedents": antecedent,
                "consequents": consequent,
            }

            if "support" in measures:
                rule_row["support"] = s_xy
            if "confidence" in measures:
                rule_row["confidence"] = conf
            if "lift" in measures:
                rule_row["lift"] = conf / s_y if s_y > 0 else 0.0
            if "conviction" in measures:
                p_not_y = 1 - s_y
                p_x_not_y = s_x - s_xy
                rule_row["conviction"] = (s_x * p_not_y) / p_x_not_y if p_x_not_y > 0 else None
            if "leverage" in measures:
                rule_row["leverage"] = s_xy - s_x * s_y

            if bayesian:
                _add_bayesian_ci(rule_row, s_xy, s_x, n, prior_alpha, prior_beta, ci_prob)

            rules_rows.append(rule_row)

    return pd.DataFrame(rules_rows) if rules_rows else pd.DataFrame(
        columns=["antecedents", "consequents"] + list(measures)
    )


def _add_bayesian_ci(
    row: dict,
    s_xy: float,
    s_x: float,
    n: int,
    prior_a: float,
    prior_b: float,
    ci_prob: float,
) -> None:
    """Append Bayesian credible interval for confidence to a rule row."""
    n_xy = round(s_xy * n)
    n_x = round(s_x * n)
    n_x_not_y = n_x - n_xy

    post_a = prior_a + n_xy
    post_b = prior_b + n_x_not_y

    alpha_lo = (1 - ci_prob) / 2
    alpha_hi = 1 - alpha_lo

    row["confidence_ci_low"] = float(beta_dist.ppf(alpha_lo, post_a, post_b))
    row["confidence_ci_high"] = float(beta_dist.ppf(alpha_hi, post_a, post_b))
