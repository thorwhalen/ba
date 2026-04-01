# Design Specification for `ba`

**ba** (Bayesian Association) provides a unified probabilistic framework for
exploring associations in categorical data, with dedicated support for
Qualitative Comparative Analysis (QCA) and Association Rule Mining (ARM).

This document specifies the architecture, data model, API surface, and design
decisions. It draws on the research in [research.md](research.md) and the
detailed analyses in [resources/](resources/).

---

## 0. Guiding Principles

1. **Categorical core, binary specialization.** The core handles r×c contingency
   tables and Dirichlet-Multinomial inference. Binary (2×2) is a specialization,
   not the general case. Only QCA and a few named metrics (OR, RR, phi, Yule's Q)
   are fundamentally binary.

2. **Uncertainty first.** At small n (10–50), point estimates are nearly
   meaningless. Every metric should be accompanied by a credible interval or
   posterior distribution. Bayes factors replace p-values as the default
   evidence measure.

3. **Progressive disclosure.** Three tiers: façade (one-liner analysis),
   paradigm (per-tradition APIs), primitives (Pot, ContingencyTable, registry).
   Simple things require zero configuration; full control is always reachable.

4. **spyn's Pot algebra as the mathematical engine.** Factor product (`*`),
   conditioning/normalization (`/`), and marginalization (`[]`) — inherited from
   spyn — are the core operators for probabilistic reasoning. ba extends but
   does not replace this algebra.

5. **DataFrame as interchange format.** Every result can be converted to a
   pandas DataFrame. But domain objects (ContingencyTable, Pot, AnalysisResult)
   carry type information and domain-specific methods.

6. **Lightweight core, optional extras.** Core depends only on numpy, scipy,
   pandas, and spyn. Heavy dependencies (pymc, mlxtend, visualization) are
   optional extras installed via `pip install ba[pymc]`, `ba[rules]`,
   `ba[viz]`, `ba[full]`.

---

## 1. Package Architecture

```
ba/
├── __init__.py              # Façade: analyze(), top-level convenience
├── core/
│   ├── contingency.py       # ContingencyTable (r×c) and ContingencyTable2x2
│   ├── pot.py               # Integration layer: spyn Pot ↔ ba types
│   ├── metrics.py           # MeasureRegistry — extensible metric computation
│   └── types.py             # Shared dataclasses (AnalysisResult, BayesianResult, etc.)
├── bayesian/
│   ├── posteriors.py         # Dirichlet-Multinomial / Beta-Binomial posteriors
│   ├── bayes_factors.py      # Gunel-Dickey BFs (all 4 sampling schemes, r×c)
│   ├── priors.py             # Prior construction: from_quantiles, from_mean_kappa, named priors
│   └── sensitivity.py        # Prior sensitivity analysis, data weight, KL divergence
├── rules/
│   ├── itemsets.py           # Frequent itemset mining (wraps mlxtend or built-in)
│   ├── encoding.py           # DataFrame → transaction encoding (binary + categorical)
│   └── filtering.py          # Interestingness filtering, Bayesian augmentation of rules
├── qca/
│   ├── calibrate.py          # Categorical/numerical → binary (explicit thresholds)
│   ├── truth_table.py        # Binary data → truth table with consistency/coverage
│   ├── minimize.py           # Quine-McCluskey Boolean minimization
│   └── necessity.py          # Necessity/sufficiency analysis
├── binary/
│   ├── __init__.py           # Convenience re-exports: 2×2 shortcuts + qca
│   └── shortcuts.py          # odds_ratio(), relative_risk(), phi(), yules_q(), etc.
├── store.py                  # DataStore: lazy, cached pot/contingency computation
├── config.py                 # Configuration with context-manager scoping
└── warnings.py               # Small-sample flags, zero-cell handling, sparsity warnings
```

### Dependency Map

| Module | Required deps | Optional deps |
|--------|--------------|---------------|
| `core/` | numpy, scipy, pandas | — |
| `bayesian/` | numpy, scipy | — |
| `rules/` | numpy, pandas | mlxtend (for fpgrowth/apriori) |
| `qca/` | numpy, pandas | — |
| `binary/` | (re-exports from core, bayesian, qca) | — |
| `store.py` | pandas, spyn | — |
| `config.py` | — | — |

spyn is a required dependency for `store.py` and `core/pot.py`. The rest of ba
operates on ContingencyTable objects and numpy arrays, independent of spyn.

---

## 2. Core Data Model

### 2.1 ContingencyTable (r×c)

The universal input for all analyses. Wraps an r×c numpy array of counts with
metadata.

```python
@dataclass
class ContingencyTable:
    counts: np.ndarray            # shape (r, c)
    row_var: str                  # name of the row variable
    col_var: str                  # name of the column variable
    row_labels: tuple[str, ...]   # one label per row level
    col_labels: tuple[str, ...]   # one label per column level

    # Derived (computed properties, not stored)
    n: int                        # grand total
    row_margins: np.ndarray       # shape (r,)
    col_margins: np.ndarray       # shape (c,)
    expected: np.ndarray          # expected counts under independence
    is_2x2: bool                  # True if shape == (2, 2)
    has_zero_cell: bool
    min_expected: float           # smallest expected count (for chi-sq validity)

    def as_2x2(self) -> ContingencyTable2x2: ...
    def to_dataframe(self) -> pd.DataFrame: ...
    def to_pot(self) -> Pot: ...
    def metrics(self, measures='default') -> dict: ...
    def summary(self) -> pd.DataFrame: ...

    @classmethod
    def from_dataframe(cls, df, row_var, col_var) -> ContingencyTable: ...
    @classmethod
    def from_counts(cls, a, b, c, d, *, row_var='X', col_var='Y') -> ContingencyTable2x2: ...
```

### 2.2 ContingencyTable2x2

Inherits from ContingencyTable. Adds binary-specific metrics as properties.

```python
@dataclass
class ContingencyTable2x2(ContingencyTable):
    # Named cells for readability
    a: int; b: int; c: int; d: int

    # Binary-specific computed properties
    odds_ratio: float | None
    relative_risk: float | None
    risk_difference: float
    phi: float
    yules_q: float | None
    qca_consistency: float        # = a / (a + b) = confidence(X → Y)
    qca_coverage: float           # = a / (a + c)
```

The `as_2x2()` method on the parent class validates shape and returns a
`ContingencyTable2x2`. Calling a binary-only metric on an r×c table raises
`ValueError` with guidance ("Use ba.qca.calibrate() to binarize, or use r×c
metrics.").

### 2.3 Pot Integration (spyn bridge)

spyn's `Pot` class is ba's mathematical engine for probabilistic reasoning. The
bridge is bidirectional:

```python
# ContingencyTable → Pot
pot = table.to_pot()

# Pot → ContingencyTable
table = ba.core.pot.to_contingency(pot, row_var='X', col_var='Y')

# Direct from data via DataStore
store = ba.DataStore(df)
joint = store.pot('treatment', 'outcome')       # returns spyn Pot
table = store.contingency('treatment', 'outcome') # returns ContingencyTable
```

**Modernization notes for spyn** (see [research.md](research.md), spyn
analysis):

- Replace `from numpy import *` with explicit imports.
- Add PEP 484 type hints throughout.
- Use `functools.cached_property` instead of custom `lazyprop`.
- Fix `order_vars()` to return a new Pot instead of mutating in-place.
- Remove custom `OrderedSet` (dict preserves insertion order since Python 3.7).
- Consider `match`/`case` (PEP 634) for the polymorphic `__getitem__`.

ba should depend on spyn and contribute improvements upstream, not fork it.

### 2.4 MeasureRegistry

An extensible registry of interestingness/association measures, inspired by
R's `arules::interestMeasure()`.

```python
class MeasureRegistry:
    def register(self, name, func, *,
                 requires_2x2=False,
                 requires_ordinal=False,
                 description=''):
        ...

    def compute(self, ct: ContingencyTable,
                measures: list[str] | str = 'default') -> dict[str, float]:
        """Compute requested measures. Skips incompatible measures with warning."""
        ...

    def available(self, ct: ContingencyTable) -> list[str]:
        """List measures compatible with this table's shape."""
        ...

    def describe(self, name: str) -> str:
        """Return formula and description for a measure."""
        ...
```

**Built-in measures (initial set, ≥23):**

- **General r×c:** support, confidence, lift, conviction, leverage, cosine,
  Jaccard, Kulczynski, chi_squared, cramers_v, mutual_info, g_test,
  fisher_p, goodman_kruskal_gamma, uncertainty_coefficient
- **Binary-only (2×2):** odds_ratio, relative_risk, risk_difference, phi,
  yules_q, qca_consistency, qca_coverage

Users extend the registry:
```python
ba.measures.register('my_metric', my_func, requires_2x2=False)
```

### 2.5 AnalysisResult

A single container for all analysis outputs, following ArviZ's
`InferenceData` pattern.

```python
@dataclass
class AnalysisResult:
    observed_data: pd.DataFrame
    contingency_tables: dict[str, ContingencyTable]
    metrics: pd.DataFrame                      # all computed measures per pair
    posterior: dict[str, BayesianResult] | None
    rules: pd.DataFrame | None                 # association rules (if mined)
    truth_table: pd.DataFrame | None           # QCA truth table (if computed)
    qca_solution: QCASolution | None
    config: dict                               # parameters used
    warnings: list[str]                        # small-sample flags, etc.

    def summary(self, kind='stats') -> pd.DataFrame: ...
    def top_pairs(self, n=10, *, sort_by='bayes_factor') -> pd.DataFrame: ...
    def top_rules(self, n=10, *, sort_by='lift') -> pd.DataFrame: ...
    def to_dataframe(self) -> pd.DataFrame: ...
```

### 2.6 BayesianResult

```python
@dataclass
class BayesianResult:
    posterior_params: dict              # e.g. {'alpha': [...], 'beta': [...]}
    posterior_mean: np.ndarray
    credible_interval: tuple[float, float]
    bayes_factor: float | None
    data_weight: float                  # n / (n + ESS_prior)
    prior_params: dict
    prior_name: str
    mc_samples: dict[str, np.ndarray] | None  # RD, RR, OR samples (2×2 only)

    def summary(self) -> pd.DataFrame: ...
    def prob_gt(self, threshold=0.0) -> float: ...
```

---

## 3. API Design (Three Tiers)

### Tier 1: Façade

One-liner entry point for the most common workflow.

```python
import ba

# Analyze all pairwise associations in a DataFrame
result = ba.analyze(df, outcome='custody_retained')
result.summary()                  # DataFrame of all pairs with metrics + CIs
result.top_pairs(10)              # strongest, most certain associations
result.top_rules(10)              # if ARM was run

# Single-pair analysis
table = ba.contingency(df, 'treatment', 'outcome')
table.summary()
```

`ba.analyze()` automatically:
- Detects variable types (binary vs. categorical)
- Computes all pairwise contingency tables
- Runs Bayesian inference with default priors (Jeffreys)
- Computes appropriate metrics per table shape
- Flags small-sample warnings
- Optionally mines association rules (if `rules=True`)

### Tier 2: Paradigm

Per-tradition APIs for users who know what they want.

```python
# Bayesian
posterior = ba.bayesian.posterior(table, prior='jeffreys')
bf = ba.bayesian.bayes_factor(table, sampling='independent')
sensitivity = ba.bayesian.sensitivity(table, priors=['jeffreys', 'uniform', 'beta(2,2)'])

# Association Rule Mining
rules = ba.rules.mine(df, min_support=2/13, outcome='Y')
rules = ba.rules.mine(df, appearance={'rhs': ['outcome=yes']})
quality = ba.measures.compute(table, ['lift', 'phi', 'conviction', 'fisher_p'])

# QCA
binary_df = ba.qca.calibrate(df, {'age': 30, 'illness': 'any_present'})
tt = ba.qca.truth_table(binary_df, outcome='retained', conditions=['A', 'B', 'C'])
solution = ba.qca.minimize(tt, include='?')

# Binary shortcuts
ba.binary.odds_ratio(table)
ba.binary.phi(table)
```

### Tier 3: Primitives

Direct access to core objects for library developers and advanced users.

```python
from ba.core import ContingencyTable, ContingencyTable2x2
from ba.core.metrics import registry as measures
from ba.core.pot import to_contingency, from_contingency
from ba.bayesian.priors import from_quantiles, from_mean_kappa, jeffreys, uniform
from ba.qca.minimize import quine_mccluskey
from ba.rules.encoding import to_transactions
from spyn import Pot
```

---

## 4. DataStore (spyn/odus heritage)

A lazy, cached interface from raw data to computed objects, generalizing odus's
`pstore` pattern.

```python
class DataStore:
    def __init__(self, data: pd.DataFrame): ...

    @cached_property
    def vars(self) -> VarNamespace:
        """Attribute namespace for all columns. Usage: store.vars.age"""
        ...

    def pot(self, *var_names: str) -> Pot:
        """Joint potential (count-based) for the given variables. Cached."""
        ...

    def contingency(self, row_var: str, col_var: str) -> ContingencyTable:
        """Contingency table for two variables. Auto-detects 2×2."""
        ...

    def all_pairs(self, *, outcome: str | None = None) -> dict[tuple[str,str], ContingencyTable]:
        """All pairwise contingency tables. If outcome given, only pairs with outcome."""
        ...
```

`VarNamespace` provides attribute access (`store.vars.age`) and metadata
(type, n_levels, is_binary) for each column.

---

## 5. Bayesian Layer

### 5.1 Posteriors

```python
def posterior(
    ct: ContingencyTable,
    *,
    prior: str | tuple | np.ndarray = 'jeffreys',
    n_mc: int = 100_000,
) -> BayesianResult:
    """Conjugate posterior for row-conditional proportions.

    For 2×2: Beta-Binomial. Derived quantities (RD, RR, OR) via MC.
    For r×c: Dirichlet-Multinomial per row.
    """
```

### 5.2 Priors

Named priors and construction helpers:

```python
def jeffreys(k: int = 2) -> np.ndarray:
    """Jeffreys prior: Dirichlet(0.5, ..., 0.5) for k categories."""

def uniform(k: int = 2) -> np.ndarray:
    """Uniform: Dirichlet(1, ..., 1) for k categories."""

def from_mean_kappa(mean: float | np.ndarray, kappa: float) -> np.ndarray:
    """Mean + concentration. kappa=2 → uniform, kappa=50 → strong."""

def from_quantiles(q1: float, p1: float, q2: float, p2: float) -> tuple[float, float]:
    """Solve for Beta(α, β) matching P(θ<q1)=p1, P(θ<q2)=p2."""

def from_counts(successes: int, failures: int) -> tuple[float, float]:
    """'Imaginary data' framing: Beta(successes+1, failures+1)."""
```

### 5.3 Bayes Factors

```python
def bayes_factor(
    ct: ContingencyTable,
    *,
    sampling: str = 'independent',  # 'poisson', 'joint', 'independent', 'hypergeometric'
    a0: float = 1.0,               # Dirichlet concentration
) -> float:
    """Gunel-Dickey BF₁₀ (association vs. independence).

    Closed-form for 2×2; log-space gammaln for r×c.
    """
```

### 5.4 Sensitivity

```python
def sensitivity(
    ct: ContingencyTable,
    *,
    priors: list[str | tuple] = ('jeffreys', 'uniform', 'beta(2,2)'),
    n_mc: int = 100_000,
) -> pd.DataFrame:
    """Compute posterior under multiple priors. Returns comparison table.

    Flags 'prior-influenced' (data_weight < 0.8) and
    'prior-dominated' (data_weight < 0.5).
    """
```

---

## 6. QCA Layer

QCA is the **binary-only** analytical layer. It accepts binary data and raises
on non-binary input with guidance toward `calibrate()`.

### 6.1 Calibration Gateway

```python
def calibrate(
    df: pd.DataFrame,
    thresholds: dict[str, float | str | Callable],
) -> pd.DataFrame:
    """Binarize categorical/numerical columns with explicit thresholds.

    thresholds values:
      - float: >= threshold → 1
      - 'any_present': any non-null/non-zero/non-'none' → 1
      - 'median': >= median → 1
      - callable: applied per-value, returns bool
    """
```

### 6.2 Truth Table

```python
def truth_table(
    data: pd.DataFrame,
    outcome: str,
    conditions: list[str],
    *,
    incl_cut: float = 0.8,
    n_cut: int = 1,
) -> pd.DataFrame:
    """Build QCA truth table from binary data.

    Raises ValueError with calibration guidance if non-binary columns found.
    Flags rows with ≤2 cases.
    """
```

### 6.3 Boolean Minimization

```python
def minimize(
    truth_table: pd.DataFrame,
    *,
    include: str = '1',    # '1' = positive only, '?' = include remainders
    method: str = 'qmc',   # Quine-McCluskey
) -> QCASolution:
    """Minimize truth table to disjunction of conjunctions.

    Returns QCASolution with .expression, .prime_implicants,
    .essential_implicants, .coverage, and .to_dataframe().
    """
```

---

## 7. Association Rule Mining Layer

Wraps mlxtend (optional dependency) or provides a minimal built-in
implementation for small datasets.

```python
def mine(
    data: pd.DataFrame,
    *,
    min_support: float = 0.1,
    min_confidence: float = 0.5,
    outcome: str | None = None,
    appearance: dict | None = None,   # {'rhs': [...], 'lhs': [...], 'none': [...]}
    algorithm: str = 'fpgrowth',      # or 'apriori'
    measures: list[str] = ('support', 'confidence', 'lift'),
    bayesian: bool = True,            # attach credible intervals to metrics
) -> pd.DataFrame:
    """Mine association rules. Returns DataFrame with rule columns + metrics.

    When bayesian=True, appends CI columns for each metric.
    When outcome is specified, constrains RHS to outcome values.
    """
```

### Encoding

```python
def to_transactions(
    df: pd.DataFrame,
    *,
    binary_as_presence: bool = True,  # binary cols: 1 → item present
    include_negation: bool = False,   # add negated items for binary cols
) -> pd.DataFrame:
    """Convert DataFrame to transaction-encoded boolean DataFrame.

    Categorical columns become one item per level (col=value).
    Binary columns become presence items (and optionally negation items).
    """
```

---

## 8. Configuration

ArviZ-style configuration with scoped overrides.

```python
# Global defaults
ba.config['stats.ci_prob'] = 0.95
ba.config['stats.default_prior'] = 'jeffreys'
ba.config['rules.min_support'] = 0.05
ba.config['qca.incl_cut'] = 0.8
ba.config['warnings.small_n_threshold'] = 30

# Scoped override
with ba.config.context({'stats.ci_prob': 0.89}):
    result.summary()   # uses 89% intervals
```

---

## 9. Warnings and Small-Sample Handling

`ba.warnings` provides cross-cutting small-sample awareness:

- **Zero-cell warning:** When any cell count is 0, flag and recommend Bayesian
  posterior over point estimates.
- **Low expected count:** When any expected count < 5, flag chi-squared as
  unreliable and recommend Fisher/Freeman-Halton exact test.
- **Prior dominance:** When data_weight < 0.5, flag "prior-dominated."
- **QCA singleton rows:** When truth table rows have ≤2 cases, flag
  consistency as unreliable.
- **ARM sparsity:** When n < 30, warn that metrics without CIs are misleading.
- **Categorical sparsity:** When a categorical variable has > n/4 levels, warn
  that most cells will be empty.

Warnings attach to result objects (not printed to stderr) so the user sees them
in `.warnings` and in `.summary()` output.

---

## 10. Design Decisions and Rationale

### Why categorical core, not binary-only?

Only QCA and ~5 named metrics are fundamentally binary. ARM, Bayesian
inference, chi-squared, mutual information, and Cramér's V are natively
categorical. Restricting to binary would mean adding constraints, not
simplifying. spyn's Pot already handles arbitrary discrete distributions.
See [resources/Unified Statistical Framework...](resources/Unified%20Statistical%20Framework%20for%20Categorical%20Pattern%20Analysis%20--%20QCA%2C%20Association%20Rule%20Mining%2C%20and%20Bayesian%20Inference.md) §0.

### Why spyn Pot, not a new algebra?

Pot's operator overloading (`*` for factor product, `/` for
normalization/conditioning, `[]` for marginalization) achieves notational
economy that no other Python library matches. It already handles categorical
data. The cost of wrapping or forking is higher than the cost of modernizing
spyn in place. ba depends on spyn and contributes improvements upstream.

### Why not PyMC as a required dependency?

PyMC has 1–15 second initialization overhead for trivial models. For conjugate
Beta-Binomial / Dirichlet-Multinomial inference, scipy gives exact, instant
results. PyMC is an optional dependency for hierarchical or non-conjugate
models only.

### Why registry-based metrics, not hardcoded functions?

R's arules supports 45+ measures via a registry. Users always want more
measures. A registry with declared shape requirements (requires_2x2,
requires_ordinal) enables: (a) user extension, (b) automatic filtering by
table shape, (c) helpful errors when a measure doesn't apply.

### Why explicit calibration for QCA?

QCA requires binary input but real data is often categorical or numerical.
Making binarization explicit (`ba.qca.calibrate()`) prevents the silent
data loss that implicit binarization causes. The calibration step is
visible in the code, not hidden inside truth_table().

### Why attach warnings to results, not stderr?

In interactive and notebook settings, stderr warnings are easily missed.
Attaching warnings to the result object makes them discoverable via
`.warnings` and displayable in `.summary()`. This follows ArviZ's pattern.
