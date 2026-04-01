"""ba: Bayesian Association — unified probabilistic framework for categorical data.

Three-tier API:

**Façade** (one-liner)::

    result = ba.analyze(df, outcome='Y')
    result.summary()

**Paradigm** (per-tradition)::

    ba.bayesian.posterior(table, prior='jeffreys')
    ba.rules.mine(df, min_support=0.1)
    ba.qca.truth_table(binary_df, outcome='Y', conditions=['A','B'])

**Primitives** (direct access)::

    from ba.core import ContingencyTable, MeasureRegistry
"""

from ba.core.contingency import ContingencyTable, ContingencyTable2x2
from ba.core.metrics import registry as measures
from ba.config import Config
from ba.store import DataStore

import ba.bayesian
import ba.binary
import ba.qca
import ba.rules

config = Config()


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def contingency_table(
    a: int,
    b: int,
    c: int,
    d: int,
    *,
    row_var: str = "X",
    col_var: str = "Y",
) -> ContingencyTable2x2:
    """Create a 2×2 contingency table from cell counts.

    Layout::

             Y=1  Y=0
        X=1 [  a    b ]
        X=0 [  c    d ]

    >>> ct = contingency_table(10, 5, 3, 12)
    >>> ct.n
    30
    >>> ct.odds_ratio
    8.0
    """
    return ContingencyTable.from_counts(a, b, c, d, row_var=row_var, col_var=col_var)


def from_dataframe(df, row_var: str, col_var: str) -> ContingencyTable:
    """Cross-tabulate two columns into a contingency table.

    Returns ContingencyTable2x2 if both variables have exactly 2 levels.

    >>> import pandas as pd
    >>> df = pd.DataFrame({'X': [1,1,0,0,1], 'Y': [1,0,1,0,1]})
    >>> ct = from_dataframe(df, 'X', 'Y')
    >>> ct.n
    5
    """
    return ContingencyTable.from_dataframe(df, row_var, col_var)


# ---------------------------------------------------------------------------
# Façade: analyze()
# ---------------------------------------------------------------------------


def analyze(
    data,
    *,
    outcome: str | None = None,
    variables: list[str] | None = None,
    prior: str = "jeffreys",
    bayesian: bool = True,
    rules: bool = False,
    min_support: float | None = None,
) -> "AnalysisResult":
    """Analyze all pairwise associations in a DataFrame.

    This is the top-level entry point. It computes contingency tables,
    metrics, and optionally Bayesian posteriors and association rules
    for all variable pairs.

    Args:
        data: DataFrame or path to CSV.
        outcome: If given, only pairs involving this variable.
        variables: Subset of columns. Default: all.
        prior: Bayesian prior specification.
        bayesian: Compute Bayesian posteriors (default True).
        rules: Mine association rules (default False).
        min_support: For rule mining; defaults to 2/n.

    Returns:
        AnalysisResult with all computed outputs.

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1,1,0,0,1,0],
    ...     'B': [1,0,1,0,1,0],
    ...     'Y': [1,1,0,0,1,0],
    ... })
    >>> result = analyze(df, outcome='Y')
    >>> len(result.contingency_tables) == 2
    True
    >>> result.summary() is not None
    True
    """
    import pandas as pd
    from ba.bayesian.posteriors import posterior as compute_posterior
    from ba.warnings import check_table, check_data_weight

    if isinstance(data, (str, bytes)):
        data = pd.read_csv(data)

    store = DataStore(data)
    pairs = store.all_pairs(outcome=outcome, variables=variables)

    ct_dict = {}
    metrics_rows = []
    posteriors = {}
    all_warnings: list[str] = []

    for (v1, v2), ct in pairs.items():
        key = f"{v1}×{v2}"
        ct_dict[key] = ct

        # Metrics
        m = ct.metrics()
        m_row = {"pair": key, "row_var": v1, "col_var": v2, "n": ct.n, **m}

        # Warnings
        all_warnings.extend(check_table(ct))

        # Bayesian
        if bayesian:
            result = compute_posterior(ct, prior=prior)
            posteriors[key] = result
            m_row["data_weight"] = result.data_weight
            all_warnings.extend(check_data_weight(result.data_weight))

            if ct.is_2x2 and result.mc_samples:
                rd_ci = result.credible_interval.get("risk_difference", (None, None))
                m_row["rd_ci_low"] = rd_ci[0]
                m_row["rd_ci_high"] = rd_ci[1]
                m_row["prob_rd_gt_0"] = float(
                    (result.mc_samples["risk_difference"] > 0).mean()
                )
                bf = ba.bayesian.bayes_factor(ct)
                m_row["bayes_factor"] = bf

        metrics_rows.append(m_row)

    metrics_df = pd.DataFrame(metrics_rows)

    # Optional rule mining
    rules_df = None
    if rules:
        if min_support is None:
            min_support = 2 / len(data) if len(data) > 0 else 0.1
        rules_df = ba.rules.mine(data, min_support=min_support, outcome=outcome)

    return AnalysisResult(
        observed_data=data,
        contingency_tables=ct_dict,
        metrics=metrics_df,
        posterior=posteriors if bayesian else None,
        rules=rules_df,
        config={"prior": prior, "outcome": outcome},
        warnings=all_warnings,
    )


# ---------------------------------------------------------------------------
# AnalysisResult
# ---------------------------------------------------------------------------


class AnalysisResult:
    """Container for all analysis outputs from ``ba.analyze()``.

    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1,0,1,0], 'B': [1,1,0,0]})
    >>> result = analyze(df)
    >>> len(result.contingency_tables) == 1
    True
    """

    def __init__(
        self,
        observed_data,
        contingency_tables,
        metrics,
        posterior,
        rules,
        config,
        warnings,
    ):
        self.observed_data = observed_data
        self.contingency_tables = contingency_tables
        self.metrics = metrics
        self.posterior = posterior
        self.rules = rules
        self.config = config
        self.warnings = warnings

    def summary(self, sort_by: str | None = None) -> "pd.DataFrame":
        """Metrics DataFrame, optionally sorted.

        >>> import pandas as pd
        >>> result = analyze(pd.DataFrame({'A': [1,0], 'B': [0,1]}))
        >>> 'pair' in result.summary().columns
        True
        """
        df = self.metrics.copy()
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        return df

    def top_pairs(self, n: int = 10, *, sort_by: str = "bayes_factor") -> "pd.DataFrame":
        """Top n pairs by the given metric."""
        df = self.summary(sort_by=sort_by)
        return df.head(n)

    def top_rules(self, n: int = 10, *, sort_by: str = "lift") -> "pd.DataFrame":
        """Top n rules (if rules were mined)."""
        if self.rules is None or self.rules.empty:
            import pandas as pd
            return pd.DataFrame()
        df = self.rules.copy()
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        return df.head(n)

    def __repr__(self) -> str:
        n_pairs = len(self.contingency_tables)
        n_warn = len(self.warnings)
        return (
            f"AnalysisResult({n_pairs} pairs, "
            f"{n_warn} warnings, "
            f"rules={'yes' if self.rules is not None else 'no'})"
        )
