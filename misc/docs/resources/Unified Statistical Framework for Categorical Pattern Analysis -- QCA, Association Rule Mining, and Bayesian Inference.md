# Unified Statistical Framework for Categorical Pattern Analysis: QCA, Association Rule Mining, and Bayesian Inference

**Author:** Thor Whalen  
**Date:** April 2026  
**Status:** Draft (v2 — generalized from binary-only to categorical-capable with binary specialization)

---

## 0. Design Decision: Categorical Core, Binary Specialization

An early architectural question was whether `ba` should restrict itself to binary data. The answer is no: **the core should handle categorical (and, where meaningful, numerical) data, with a dedicated binary specialization layer for 2×2-specific statistics and QCA.**

The reasoning:

| Technique | Binary-only? | Generalization | Cost to support both |
|---|---|---|---|
| QCA truth tables + Quine-McCluskey | **Yes** | mvQCA is a different algorithm | High — separate code path |
| Necessity/sufficiency (crisp sets) | **Yes** | fsQCA for graded membership | Moderate |
| Yule's Q | **Yes** | Goodman-Kruskal γ | Trivial |
| Phi coefficient | **Yes** | Cramér's V (φ is V for 2×2) | Trivial |
| Odds ratio (simple 2×2) | **Yes** | Generalized OR, polytomous LR | Moderate |
| Relative risk / risk difference | **Yes** | Multinomial contrasts | Moderate |
| ARM (support, confidence, lift, etc.) | No | **Native** — designed for categorical | Zero |
| Bayesian posterior | No | Beta→Dirichlet, same algebra | Low |
| Bayes factor (contingency) | No | r×c Gunel-Dickey | Low |
| Fisher's exact test | No | Freeman-Halton for r×c | Zero (scipy) |
| Cosine, Jaccard, Kulczynski | No | Native on itemsets | Zero |
| Chi-squared, Cramér's V, mutual info | No | **Native** — r×c is the natural case | Zero |

**Only QCA and a handful of named metrics are fundamentally binary.** Everything else is natively categorical or generalizes trivially. The marginal cost of categorical support is near-zero for the ARM and Bayesian layers — spyn's `Pot` already handles arbitrary discrete distributions. Restricting to binary would be adding constraints, not simplifying.

Practically, the motivating dataset (`truth_table_data.csv`) is not fully binary: columns like `Physical Illness`, `Mental Illness`, `Using Illegal Drug`, and `Intense Surveillance` are multi-valued strings that were forced into binary for QCA. A categorical-capable `ba` lets users analyze them at their natural resolution.

**Architectural consequence:**

```
ba/
├── core/
│   ├── contingency.py    # r×c tables as general case; 2×2 as specialization
│   ├── pot.py             # spyn Pot — already handles arbitrary discrete
│   └── metrics.py         # Registry of measures; each knows which table shapes it supports
├── bayesian/              # Dirichlet-Multinomial (Beta-Binomial falls out as r=c=2)
├── rules/                 # ARM — natively categorical
├── qca/                   # Binary-only layer; accepts binary data, raises on non-binary
└── binary/                # Convenience: 2×2 shortcuts, OR, RR, phi, Yule's Q, QCA re-exports
```

Throughout this report, formulas are presented in both the general r×c form and the 2×2 specialization where applicable.

---

## 1. The Shared Data Structure

### 1.1 The Contingency Table as Universal Input

Every computation in all three traditions begins with the same object. Given a condition variable $X$ (with $r$ levels) and an outcome variable $Y$ (with $c$ levels) observed across $n$ cases, the **r×c contingency table** is the sufficient statistic.

**General case (r×c):**

$$C = \begin{pmatrix} n_{11} & n_{12} & \cdots & n_{1c} \\ n_{21} & n_{22} & \cdots & n_{2c} \\ \vdots & & \ddots & \vdots \\ n_{r1} & n_{r2} & \cdots & n_{rc} \end{pmatrix}$$

with row margins $r_i = \sum_j n_{ij}$, column margins $c_j = \sum_i n_{ij}$, and grand total $n = \sum_{ij} n_{ij}$.

**2×2 specialization:** When both $X$ and $Y$ are binary, this reduces to:

|  | $Y=1$ | $Y=0$ | Row Total |
|---|---|---|---|
| $X=1$ | $a$ | $b$ | $r_1 = a+b$ |
| $X=0$ | $c$ | $d$ | $r_2 = c+d$ |
| **Col Total** | $c_1 = a+c$ | $c_2 = b+d$ | $n$ |

Each tradition names these cells differently but computes from the same underlying counts:

- **QCA** calls $a$ the count of cases compatible with configuration $S_k$ and showing outcome $R$. QCA requires binary encoding (crisp sets) or graded membership (fuzzy sets).
- **ARM** calls $n_{ij}/n$ the "support" of the itemset $\{X=i, Y=j\}$. The transaction database is the data matrix; each row is a transaction; each variable-value pair is an item. **ARM is natively categorical** — each variable can have any number of levels, and each level becomes a separate item.
- **Bayesian inference** treats each row of the contingency table as a multinomial observation. For 2×2: $(a, b) \sim \text{Binomial}(r_1, \theta_1)$ and $(c, d) \sim \text{Binomial}(r_2, \theta_2)$. For r×c: each row $i$ has $(n_{i1}, \ldots, n_{ic}) \sim \text{Multinomial}(r_i, \boldsymbol{\theta}_i)$.

### 1.2 Generalization to k-tuple Conditions

For a conjunction of $k$ conditions $X_1 = x_1 \wedge X_2 = x_2 \wedge \cdots \wedge X_k = x_k$, the contingency table generalizes: the "condition met" row aggregates cases matching the full conjunction. For categorical conditions, the number of possible configurations is $\prod_{i=1}^{k} |X_i|$ (which equals $2^k$ when all conditions are binary).

ARM handles this natively — a $k$-itemset is simply a set of $k$ item-value pairs. QCA requires that all conditions be binary (or binarized for crisp-set analysis).

### 1.3 What Happens When $n = 13$ and $k = 28$

This is the critical regime for the motivating dataset. With $k = 28$ binary conditions, the QCA truth table has $2^{28} \approx 268$ million rows. With only **13 cases, at most 13 rows are populated** — over 99.999995% are logical remainders (empty). Each case likely occupies its own unique truth table row, making the truth table a trivial enumeration of individuals.

For the 2×2 table: with $n = 13$, the average cell count is 3.25. For variables with skewed marginals (e.g., prevalence 2/13), individual cells regularly contain 0 or 1 cases. For categorical variables with $>2$ levels, cells are even sparser.

For $k$-tuple conditions with $k \geq 3$: the number of cases matching a three-variable conjunction is almost always 0 or 1. **Higher-order analysis ($k \geq 3$) is essentially impossible at $n = 13$** without strong prior assumptions.

---

## 2. Metric Equivalence Table

### 2.1 General r×c Metrics

These metrics apply to contingency tables of any size. They form the core of `ba`'s `metrics.py`.

| # | Metric Name(s) | Formula (r×c) | Type | Small-$n$ Behavior |
|---|---|---|---|---|
| 1 | **Chi-squared statistic** | $\chi^2 = \sum_{ij} \frac{(n_{ij} - e_{ij})^2}{e_{ij}}$, $e_{ij} = r_i c_j / n$ | Point est. | Unreliable when any $e_{ij} < 5$ (common at $n = 13$). Use Fisher/Freeman-Halton instead. |
| 2 | **Cramér's V** | $V = \sqrt{\chi^2 / (n \cdot (\min(r,c) - 1))}$ | Point est. | Inherits chi-squared unreliability. For 2×2: $V = |\phi|$. |
| 3 | **Mutual Information** | $I(X;Y) = \sum_{ij} \frac{n_{ij}}{n} \ln \frac{n_{ij} \cdot n}{r_i \cdot c_j}$ | Point est. | Positively biased at small $n$. Use bias-corrected estimator or Bayesian MI. |
| 4 | **G-test (log-likelihood ratio)** | $G = 2 \sum_{ij} n_{ij} \ln \frac{n_{ij}}{e_{ij}}$ | Point est. | Same expected-count issues as $\chi^2$. |
| 5 | **Fisher-Freeman-Halton exact test** | Exact $p$ via multivariate hypergeometric | $p$-value | **Valid at any $n$**, any table size. Computationally exact. Use this at $n < 30$. |
| 6 | **Goodman-Kruskal $\gamma$** | $\gamma = (C - D)/(C + D)$, concordant/discordant pairs | Point est. | Generalizes Yule's Q to ordinal categorical. |
| 7 | **Goodman-Kruskal $\lambda$** | PRE measure for nominal association | Point est. | Asymmetric; can be 0 even with strong association. |
| 8 | **Uncertainty Coefficient (Theil's U)** | $U(Y|X) = I(X;Y) / H(Y)$ | Point est. | Asymmetric version of mutual information, normalized to [0,1]. |

### 2.2 ARM Metrics (Natively Categorical)

These metrics are defined on association rules $X \Rightarrow Y$ where $X$ and $Y$ are itemsets (arbitrary variable-value pairs). They apply identically to binary and categorical data.

| # | Metric | Formula | Prob. Interpretation | Small-$n$ ($n \approx 13$) |
|---|---|---|---|---|
| 9 | **Support** | $\text{supp}(X \Rightarrow Y) = n_{XY}/n$ | $P(X \cap Y)$ | Granularity $= 1/13$. One co-occurrence qualifies. |
| 10 | **Confidence** (= QCA Consistency for binary) | $\text{conf}(X \Rightarrow Y) = n_{XY}/n_X$ | $P(Y \mid X)$ | Trivially 1.0 or 0.0 when $n_X \leq 2$. |
| 11 | **Lift** | $n_{XY} \cdot n / (n_X \cdot n_Y)$ | $P(Y \mid X)/P(Y)$ | Unstable: $\pm 1$ case shifts lift by 30–50%. |
| 12 | **Conviction** | $\frac{n_X \cdot (n - n_Y)}{n \cdot (n_X - n_{XY})}$ | $\frac{P(X)P(\bar{Y})}{P(X \cap \bar{Y})}$ | Undefined when $n_X = n_{XY}$ (perfect rule). |
| 13 | **Leverage** | $n_{XY}/n - n_X n_Y/n^2$ | $P(X \cap Y) - P(X)P(Y)$ | Tiny values, granularity $\sim 1/n$. |
| 14 | **Cosine (Ochiai)** | $n_{XY}/\sqrt{n_X \cdot n_Y}$ | $\sqrt{P(Y|X) \cdot P(X|Y)}$ | Less sensitive to zero cells than Jaccard. |
| 15 | **Jaccard** | $n_{XY}/(n_X + n_Y - n_{XY})$ | $P(X \cap Y)/P(X \cup Y)$ | Null-invariant. Granularity $\sim 1/13$. |
| 16 | **Kulczynski** | $\frac{1}{2}(n_{XY}/n_X + n_{XY}/n_Y)$ | $\frac{1}{2}[P(Y|X) + P(X|Y)]$ | Average of two unstable proportions. |

### 2.3 Binary-Specific Metrics (2×2 Only)

These metrics are defined only for $r = c = 2$. They live in the `ba.binary` module and require a `ContingencyTable2x2`.

| # | Metric | Formula (in $a,b,c,d$) | Small-$n$ ($n \approx 13$) |
|---|---|---|---|
| 17 | **QCA Consistency (sufficiency)** = ARM Confidence | $a/(a+b)$ | Trivially 1.0 for single-case rows. |
| 18 | **QCA Coverage (sufficiency)** = QCA Consistency (necessity) | $a/(a+c)$ | Same instability. |
| 19 | **Odds Ratio** | $ad/(bc)$ | Undefined when $b=0$ or $c=0$ (common). |
| 20 | **Relative Risk** | $a(c+d)/[c(a+b)]$ | Undefined when $c=0$. Wide CIs. |
| 21 | **Risk Difference** | $a/(a+b) - c/(c+d)$ | Steps of $\sim 1/6$ to $1/8$. |
| 22 | **Phi Coefficient** | $(ad-bc)/\sqrt{(a+b)(c+d)(a+c)(b+d)}$ | $= \sqrt{\chi^2/n}$ for 2×2. |
| 23 | **Yule's Q** | $(ad-bc)/(ad+bc)$ | Saturates to $\pm 1$ when any cell = 0. |

### 2.4 Key Algebraic Relationships

These identities enable a DRY engine to compute all metrics from a minimal set of base counts. They hold for both general and binary cases:

$$\text{Confidence}(X \Rightarrow Y) = n_{XY}/n_X$$
$$\text{Coverage}(X \Rightarrow Y) = \text{Confidence}(Y \Rightarrow X) = n_{XY}/n_Y$$
$$\text{Lift} = \text{Confidence}(X \Rightarrow Y) / P(Y) = \text{Confidence} \cdot n / n_Y$$
$$\text{Cosine} = \sqrt{\text{Confidence}(X \Rightarrow Y) \cdot \text{Confidence}(Y \Rightarrow X)}$$

For 2×2 only:

$$\phi = \text{Leverage} / \sqrt{P(X)P(\bar{X})P(Y)P(\bar{Y})}$$
$$\text{Yule's } Q = (\text{OR} - 1)/(\text{OR} + 1)$$
$$\text{Cramér's } V = |\phi| \quad \text{(since } \min(r,c)-1 = 1 \text{)}$$

### 2.5 Bayes Factors for Contingency Tables

**General r×c (Gunel-Dickey).** For the joint multinomial with Dirichlet$(a_0)$ prior, the Bayes factor is:

$$\text{BF}_{10}^{M} = \frac{\prod_{ij} \Gamma(n_{ij} + a_0) \cdot \Gamma(rc \cdot a_0)}{\prod_i \Gamma(r_i + c \cdot a_0) \cdot \prod_j \Gamma(c_j + r \cdot a_0) / \Gamma(n + rc \cdot a_0)} \cdot (\text{normalization})$$

The full derivation for all four sampling schemes (Poisson, joint multinomial, independent multinomial, hypergeometric) is in Jamil et al. [3].

**2×2 specialization (closed-form).** For independent multinomial with $a_0 = 1$:

$$\text{BF}_{01}^{I} = \frac{\binom{c_1}{a}\binom{c_2}{b}}{\binom{n}{r_1}} \cdot \frac{(c_1 + 1)(c_2 + 1)}{n + 1}$$

All can be computed in log-space via `scipy.special.gammaln`. At $n = 13$, BF typically ranges 0.3 to 5 (anecdotal evidence) unless OR exceeds ~10 [3].

---

## 3. What Each Tradition Adds That the Others Lack

### 3.1 QCA: Boolean Logic (Binary-Only Layer)

QCA's unique contribution is its **logical framework**: necessity vs. sufficiency, equifinality (multiple causal "recipes"), and Boolean minimization. **This entire apparatus requires binary data.** It constitutes `ba`'s `qca/` module, which accepts binary input and raises on non-binary.

**Quine-McCluskey algorithm.** Input: truth table with rows marked 1 (leads to outcome), 0, or don't-care (logical remainder). Output: minimized Boolean expression (disjunction of conjunctions). The algorithm is computationally infeasible at $k > 15$–$20$ but fast for the 3–5 theory-driven conditions typical in QCA practice.

**At $n = 13$:** The user must pre-select 3–5 conditions from the full variable set, reducing the truth table to $2^3 = 8$ or $2^5 = 32$ rows. The implementation accepts an arbitrary condition subset and builds the truth table only for that subset.

**Logical remainder handling.** Three strategies: conservative (remainders → 0), parsimonious (remainders → don't-care), intermediate (easy counterfactuals only). At $n = 13$, the parsimonious solution is heavily driven by remainder assumptions — flag this.

**For non-binary variables:** Users must binarize before entering the QCA layer. `ba` should provide calibration utilities (`ba.qca.calibrate()`) that binarize categorical or numerical variables with explicit, user-specified thresholds.

### 3.2 ARM: Efficient Search (Natively Categorical)

ARM's unique contribution is its **search algorithms** — Apriori and FP-growth — that avoid enumerating all $\prod |X_i|$ possible configurations. These are natively categorical: each item is a variable-value pair, and the algorithms don't care how many values each variable has.

**At $n = 13$:** Both algorithms terminate instantly. The bottleneck is interpretation: with 28+ variables, the number of frequent itemsets with $\text{support} \geq 1/13$ can reach thousands. Default to $s_{\min} = 2/n$ and attach Bayesian credible intervals to every rule.

**The encoding for QCA-ARM bridge [1].** To represent negated binary conditions, each variable $V_j$ becomes two items: $V_j$ (presence) and $v_j$ (absence). For categorical variables with $m$ levels, $V_j$ becomes $m$ items: $V_j{=}\text{level}_1$, ..., $V_j{=}\text{level}_m$. This encoding is the bridge between ARM's itemset world and QCA's Boolean world.

### 3.3 Bayesian Inference: Honest Uncertainty (Generalizes Cleanly)

The Bayesian layer provides distributional answers. It generalizes from binary to categorical with the same algebra.

**Binary (Beta-Binomial).** Prior: $\theta \sim \text{Beta}(\alpha_0, \beta_0)$. Posterior: $\theta \mid \text{data} \sim \text{Beta}(\alpha_0 + a, \beta_0 + b)$.

**Categorical (Dirichlet-Multinomial).** Prior: $\boldsymbol{\theta}_i \sim \text{Dir}(\alpha_1, \ldots, \alpha_c)$. Posterior: $\boldsymbol{\theta}_i \mid \text{data} \sim \text{Dir}(\alpha_1 + n_{i1}, \ldots, \alpha_c + n_{ic})$.

The Beta-Binomial is the $c = 2$ special case of the Dirichlet-Multinomial. Implementing the general case covers both.

**Posterior of derived quantities (binary-specific).** Risk difference $\delta = \theta_1 - \theta_2$, relative risk $\rho = \theta_1/\theta_2$, and odds ratio $\omega$ have no closed-form posteriors. Use Monte Carlo: draw $N = 100{,}000$ samples from each Beta posterior, compute derived quantities sample-wise, and summarize. At $n = 13$, this is instantaneous.

**Prior sensitivity.** Always compute and display the **data weight** $w = n_{\text{obs}}/(n_{\text{obs}} + \text{ESS}_{\text{prior}})$. Flag $w < 0.8$ as "prior-influenced" and $w < 0.5$ as "prior-dominated."

---

## 4. DRY Implementation Architecture

### 4.1 Module Hierarchy

```
ba/
├── core/
│   ├── contingency.py       # Layer 0: Data matrix → r×c tables (2×2 as subclass)
│   ├── pot.py                # Layer 0: spyn Pot — general discrete potentials
│   ├── metrics.py            # Layer 1: Cell counts → ALL named metrics (registry-based)
│   └── types.py              # Shared dataclasses
├── bayesian/
│   ├── posteriors.py         # Layer 2: Dirichlet-Multinomial (Beta-Binomial as r=c=2)
│   ├── bayes_factors.py      # Layer 2: Gunel-Dickey BFs (r×c, all sampling schemes)
│   └── sensitivity.py        # Layer 2: Prior sensitivity, KL, data weight
├── rules/
│   ├── itemsets.py           # Layer 3: Apriori/FP-growth (mlxtend wrapper)
│   ├── encoding.py           # Layer 3: Variable-value encoding (binary + categorical)
│   └── filtering.py          # Layer 3: Interestingness filtering + Bayesian augmentation
├── qca/
│   ├── truth_table.py        # Layer 3: Binary data → truth table (condition subset)
│   ├── minimize.py           # Layer 3: Quine-McCluskey (binary only)
│   ├── necessity.py          # Layer 3: Necessity analysis
│   └── calibrate.py          # Layer 3: Categorical/numerical → binary for QCA input
├── binary/
│   ├── shortcuts.py          # Convenience: 2×2-specific functions (OR, RR, phi, Yule's Q)
│   └── __init__.py           # Re-exports from qca/ and binary-specific metrics
├── warnings.py               # Cross-cutting: small-sample flags, zero-cell handling
└── config.py                 # ArviZ-style configuration with context manager
```

### 4.2 Layer 0 — `core/contingency.py`

**General r×c table:**

```python
@dataclass
class ContingencyTable:
    """r×c contingency table. The universal input for all analyses."""
    counts: np.ndarray           # r×c array of counts
    row_var: str                 # Name of the row variable
    col_var: str                 # Name of the column variable
    row_labels: list[str]        # Labels for each row level
    col_labels: list[str]        # Labels for each column level

    @property
    def n(self) -> int: return self.counts.sum()
    @property
    def row_margins(self) -> np.ndarray: return self.counts.sum(axis=1)
    @property
    def col_margins(self) -> np.ndarray: return self.counts.sum(axis=0)
    @property
    def expected(self) -> np.ndarray:
        return np.outer(self.row_margins, self.col_margins) / self.n
    @property
    def is_2x2(self) -> bool: return self.counts.shape == (2, 2)
    @property
    def has_zero_cell(self) -> bool: return (self.counts == 0).any()
    @property
    def min_cell(self) -> int: return int(self.counts.min())

    def as_2x2(self) -> 'ContingencyTable2x2':
        """Convert to 2×2 specialization. Raises ValueError if not 2×2."""
        if not self.is_2x2:
            raise ValueError(f"Table is {self.counts.shape}, not 2×2. "
                             "Use ba.qca.calibrate() to binarize, or use r×c metrics.")
        return ContingencyTable2x2(
            a=self.counts[0,0], b=self.counts[0,1],
            c=self.counts[1,0], d=self.counts[1,1],
            row_var=self.row_var, col_var=self.col_var)
```

**2×2 specialization (inherits general, adds binary-specific):**

```python
@dataclass
class ContingencyTable2x2(ContingencyTable):
    """2×2 contingency table. Adds OR, RR, phi, Yule's Q."""
    a: int; b: int; c: int; d: int

    # Binary-specific properties
    @property
    def odds_ratio(self) -> float | None: ...
    @property
    def relative_risk(self) -> float | None: ...
    @property
    def phi(self) -> float: ...
    @property
    def yules_q(self) -> float | None: ...
    # ... etc
```

### 4.3 Layer 0 — `core/pot.py` — spyn Integration

The `Pot` class from spyn is `ba`'s mathematical core for probabilistic reasoning. It already handles arbitrary discrete distributions. Key operators:

```python
from ba.core import Pot

joint = data.pot('treatment', 'outcome')    # Joint potential from data
conditional = joint / joint['treatment']    # P(outcome | treatment)
marginal = joint['outcome']                 # Marginalize to outcome
posterior = (likelihood * prior) / []       # Bayesian update + normalize
```

**Proposed improvements to spyn for `ba` integration:**

1. **`.to_contingency()`**: Convert a 2-variable Pot directly to a `ContingencyTable`, bridging the algebraic world and the metric-computation world.
2. **`.credible_interval(alpha=0.05)`**: When a Pot represents a posterior (e.g., from Beta-Binomial), compute HDI or ETI directly.
3. **`.data_weight`**: When a Pot is the result of a Bayesian update, track and expose how much the posterior was driven by data vs. prior.
4. **Sparse storage option**: For high-cardinality categoricals, the full joint table is wasteful. An optional sparse backend (scipy.sparse or dict-of-counts) would help.
5. **`.from_contingency(ct)`**: Construct a Pot from a ContingencyTable, enabling round-trips.

### 4.4 Layer 1 — `core/metrics.py` — Registry-Based Measures

A registry pattern (inspired by R's `arules::interestMeasure()`) where measures are named callables that declare their requirements:

```python
class MeasureRegistry:
    """Extensible registry of interestingness/association measures."""

    def register(self, name: str, func: Callable, requires_2x2: bool = False,
                 requires_ordinal: bool = False, description: str = ""):
        ...

    def compute(self, ct: ContingencyTable, measures: list[str] | str = 'all') -> dict:
        """Compute requested measures from a contingency table.
        Skips measures incompatible with the table shape, with warnings."""
        ...

# Built-in registrations:
registry.register('support', compute_support, requires_2x2=False)
registry.register('confidence', compute_confidence, requires_2x2=False)
registry.register('lift', compute_lift, requires_2x2=False)
# ...
registry.register('odds_ratio', compute_or, requires_2x2=True)
registry.register('yules_q', compute_yules_q, requires_2x2=True)
registry.register('cramers_v', compute_cramers_v, requires_2x2=False)
# ...

# User extension:
registry.register('my_custom', my_func, requires_2x2=False)
```

When a user requests `odds_ratio` on a 3×4 table, the registry returns a warning ("odds_ratio requires a 2×2 table; use ba.qca.calibrate() to binarize") rather than silently failing.

### 4.5 Layer 2 — `bayesian/` — Dirichlet-Multinomial Core

```python
def posterior_proportions(
    ct: ContingencyTable,
    prior: np.ndarray | tuple[float, float] | str = 'jeffreys',
    n_mc: int = 100_000
) -> BayesianResult:
    """Bayesian posterior for row-conditional proportions.
    
    For 2×2: uses Beta-Binomial (fast, exact).
    For r×c: uses Dirichlet-Multinomial (fast, exact).
    Derived quantities (RR, OR, risk diff) computed via MC for 2×2 only.
    """
    if ct.is_2x2:
        return _beta_binomial(ct.as_2x2(), prior, n_mc)
    else:
        return _dirichlet_multinomial(ct, prior)
```

### 4.6 Layer 3 — `qca/` — Binary-Only with Calibration Gateway

```python
def truth_table(data: pd.DataFrame, outcome: str, conditions: list[str],
                incl_cut: float = 0.8, n_cut: int = 1) -> pd.DataFrame:
    """Build a QCA truth table. All conditions must be binary (0/1, True/False).
    Raises ValueError with calibration guidance if non-binary columns are found."""
    non_binary = [c for c in conditions if data[c].nunique() > 2]
    if non_binary:
        raise ValueError(
            f"QCA requires binary conditions. Non-binary columns: {non_binary}. "
            f"Use ba.qca.calibrate(data, {non_binary[0]!r}, threshold=...) first.")
    ...
```

---

## 5. Open Source Libraries to Leverage or Reference

### Python — use directly

| Library | Use for | Categorical support | Notes |
|---|---|---|---|
| **`scipy.stats`** | Fisher's exact, Beta/Dirichlet, `gammaln` for BF | r×c via `chi2_contingency`, `fisher_exact` (2×2 only, Freeman-Halton via `monte_carlo_test`) | Foundation layer. |
| **`mlxtend`** | Apriori, FP-growth, association rules | **Natively categorical** via TransactionEncoder | Gold standard Python ARM. |
| **`statsmodels`** | `Table2x2`, `Table` for r×c, Cochran-Mantel-Haenszel | General r×c support | Complementary for frequentist. |
| **`spyn`** | Potential algebra — `ba`'s mathematical core | **Already handles arbitrary discrete** | Our package. Extend as needed. |

### Python — implement from scratch

| Component | Why | Reference |
|---|---|---|
| **Gunel-Dickey BF** (r×c) | No Python package. Closed-form for 2×2; Laplace approx for r×c. | R `BayesFactor::contingencyTableBF()`; Jamil et al. [3] |
| **QCA truth table + minimization** | No production-ready Python QCA. | R `QCA` v3.23 by Dușa [6] |
| **Measure registry** | No Python equivalent of arules' 45+ measures with `interestMeasure()`. | R `arules::interestMeasure()`; Hahsler's compendium [7] |

### R — reference for correctness checking

| Package | Provides | Use |
|---|---|---|
| **`QCA`** | Complete QCA workflow, Boolean minimization | Gold standard; validate our QCA layer |
| **`cna`** | Coincidence Analysis, redundancy-free models | Alternative minimization; compare results |
| **`arules`** | 45+ interestingness measures, S4 typed objects | Validate our measure registry |
| **`BayesFactor`** | Gunel-Dickey BFs, all sampling schemes | Validate our BF implementation |

---

## 6. Gotchas for Small Samples ($n = 10$–$13$, up to $n \leq 50$)

### 6.1 ARM Metrics Become Descriptive, Not Inferential

With $n = 13$, minimum support of $1/13 \approx 0.077$ means a single co-occurrence qualifies. Set minimum support to $2/n$. Confidence values are ratios of small integers with ~10 distinct possible values. **Never display ARM metrics without Bayesian credible intervals at $n < 30$.** This applies equally to binary and categorical variables — categorical makes it worse because counts are split across more levels.

### 6.2 QCA Consistency Misleads with Singleton Rows

When a truth table row contains a single case, consistency is trivially 1.0 or 0.0. Any truth table with $> 4$ conditions will have most rows containing 0–1 cases at $n = 13$. Display case counts alongside consistency; flag rows with $\leq 2$ cases; default to `n_cut = 2`.

### 6.3 Prior Dominance Is Quantifiable

Data weight $w = n_{\text{obs}}/(n_{\text{obs}} + \text{ESS}_{\text{prior}})$. For Jeffreys (ESS = 1) and $n_1 = 3$: $w = 0.75$, prior contributes 25%. For $n_1 = 1$: $w = 0.5$, prior and data contribute equally. Automate sensitivity: compute under three priors, flag when conclusions differ.

### 6.4 Categorical Variables Worsen Sparsity

A categorical variable with 4 levels splits $n = 13$ cases into ~3.25 per level on average. Cross-tabulation of two 4-level variables produces a $4 \times 4 = 16$-cell table with average cell count $< 1$. **Categorical variables with $> 3$ levels are essentially unanalyzable at $n = 13$** without collapsing levels or using Bayesian pooling.

### 6.5 Combinatorial Explosion

378 pairwise comparisons with 28 variables. After Bonferroni at $\alpha = 0.05$: per-test threshold $= 0.000132$, essentially unreachable. Use Bayes factors as continuous evidence measures instead of dichotomous testing.

### 6.6 Zero-Cell Handling

Prefer Bayesian posterior (always finite) over Haldane correction (add 0.5, inflates $n$ by ~15%). Provide exact conditional MLE via `scipy.stats.contingency.odds_ratio(kind='conditional')` as alternative. For r×c with zero cells: Dirichlet prior with $\alpha_0 = 0.5$ (Jeffreys) smooths naturally.

---

## 7. References

[1] Dom Luís A, Benítez R, Bas MC. Bridging Crisp-Set Qualitative Comparative Analysis and Association Rule Mining: A Formal and Computational Integration. Mathematics. 2025;13(12):1939. https://doi.org/10.3390/math13121939

[2] Gunel E, Dickey J. Bayes factors for independence in contingency tables. Biometrika. 1974;61(3):545-557.

[3] Jamil T, Ly A, Morey RD, Love J, Marsman M, Wagenmakers EJ. Default "Gunel and Dickey" Bayes factors for contingency tables. Behav Res Methods. 2017;49(2):638-652. https://doi.org/10.3758/s13428-016-0739-8

[4] Han J, Pei J, Yin Y. Mining frequent patterns without candidate generation. Proc ACM SIGMOD. 2000:1-12.

[5] Morey RD, Rouder JN. BayesFactor: Computation of Bayes Factors for Common Designs. R package v0.9.12. 2024. https://cran.r-project.org/package=BayesFactor

[6] Dușa A. QCA with R: A Comprehensive Resource. Springer; 2019. https://doi.org/10.1007/978-3-319-75668-4

[7] Hahsler M. A Probabilistic Comparison of Commonly Used Interest Measures for Association Rules. 2015 (updated 2024). https://mhahsler.github.io/arules/docs/measures

[8] Schneider CQ, Wagemann C. Set-Theoretic Methods for the Social Sciences. Cambridge University Press; 2012.

[9] Ragin CC. The Comparative Method. UC Press; 1987.

[10] Ragin CC. Redesigning Social Inquiry. U Chicago Press; 2008.

[11] Agrawal R, Imielinski T, Swami A. Mining Association Rules Between Sets of Items in Large Databases. Proc ACM SIGMOD. 1993:207-216.

[12] Agresti A. Categorical Data Analysis. 3rd ed. Wiley; 2013.

[13] Tan PN, Kumar V, Srivastava J. Selecting the Right Objective Measure for Association Analysis. Inf Syst. 2004;29(4):293-313.

[14] Raschka S. MLxtend. J Open Source Softw. 2018;3(24):638. https://github.com/rasbt/mlxtend

[15] Whalen T. spyn — Potentials and probabilistic inference. https://github.com/thorwhalen/spyn
