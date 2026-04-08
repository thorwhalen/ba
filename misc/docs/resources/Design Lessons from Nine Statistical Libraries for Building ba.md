# Design Lessons from Nine Statistical Libraries for Building `ba`

**Author:** Thor Whalen  
**Date:** April 2026  
**Status:** Draft (v2 — updated for categorical-capable core with binary specialization)

---

## 0. Framing: Why Categorical Core, Binary Specialization

The original analysis assumed `ba` would be binary-only. After mapping which techniques are fundamentally binary (only QCA and a handful of named metrics like Yule's Q, phi, OR, RR) vs. natively categorical (ARM, Bayesian inference, chi-squared family, mutual information), the architecture shifted: **design the core for categorical data, with a dedicated `ba.binary` module for 2×2-specific statistics and a `ba.qca` module as a binary-only analytical layer.**

This decision is reinforced by the library analysis: spyn's `Pot` already handles arbitrary discrete distributions, mlxtend's ARM pipeline is natively categorical, and scipy's `dirichlet` generalizes `beta` with the same algebra. Restricting to binary would mean *adding* constraints, not simplifying.

**The most successful statistical libraries share a core insight: use DataFrames as the universal interchange format, wrap domain concepts in typed objects with algebraic operators, and separate computation from presentation through progressive disclosure.** This report maps those patterns across mlxtend, scipy.stats, statsmodels, PyMC, ArviZ, spyn, odus, R's QCA, and R's arules, extracting what `ba` should adopt, adapt, and avoid.

---

## 1. The DataFrame Consensus and Its Limits

Every library converges on **pandas DataFrames as the primary user-facing data structure** — but the best libraries don't stop there. mlxtend's entire association rule pipeline flows DataFrame-in, DataFrame-out: `TransactionEncoder` produces a boolean DataFrame, `fpgrowth()` returns a DataFrame of itemsets with support values, and `association_rules()` returns a DataFrame of rules with metric columns [1]. This leverages the full pandas ecosystem for filtering, sorting, and export without users learning any custom API.

Yet pure DataFrames have limits. R's arules wraps its core data in **typed S4 classes** — `transactions`, `itemsets`, and `rules` — each backed by sparse matrices with quality metrics as an extensible attached data.frame [2]. This enables operations like `is.redundant()`, `is.significant()`, and `is.closed()` that mlxtend cannot express. The lesson: DataFrames should be the *display layer* and *interchange format*, but domain objects should carry type information and domain-specific operations. Every object should have a `.to_dataframe()` escape hatch.

statsmodels' `Table2x2` class demonstrates the intermediate position: wrap a numpy array in a typed class with **lazy computed properties** (`t.oddsratio`, `t.riskratio`, `t.log_oddsratio_se`) and a `.summary()` method that produces a complete formatted report in one call [3]. This pattern — instantiate once, access many derived quantities — is the right model for `ba`'s contingency table analysis.

**For `ba`:** `ContingencyTable` (r×c general) and `ContingencyTable2x2` (binary specialization) both expose `.to_dataframe()` but carry typed properties and methods appropriate to their dimensionality.

---

## 2. How Composition Patterns Diverge

### 2.1 Functional Pipeline (mlxtend, scipy.stats)

mlxtend chains pure functions with consistent signatures — `apriori()`, `fpgrowth()`, `fpmax()`, and `hmine()` all accept the same DataFrame schema and return the same output schema, making algorithms **interchangeable with a single function swap** [1]. scipy.stats takes the same approach with `fisher_exact()`, `barnard_exact()`, and `chi2_contingency()`.

**Weakness:** Pipeline metadata gets lost between steps — `association_rules()` needs the original transaction count but the frequent itemset DataFrame doesn't carry it, forcing users to pass `num_itemsets=len(df.index)` manually [1]. This is a brittle coupling `ba` must avoid.

### 2.2 Operator-Overloaded Algebra (spyn — `ba`'s mathematical core)

This is spyn's signature contribution and `ba`'s most important inheritance. The `Pot` class represents factors (probability tables) as pandas DataFrames with MultiIndex, but the interaction model is through Python operators that mirror mathematical notation [5]:

```python
posterior = (evidence * prior) / []      # Bayes' rule: product, then normalize
conditional = joint / joint[given_vars]  # P(X|Y) = P(X,Y) / P(Y)
marginal = joint[var_name]               # Sum out all but named variables
```

`*` is factor product, `/` is normalization or conditioning, `[]` is marginalization. **No other library in this analysis achieves this level of notational economy for probabilistic inference.** Crucially, **this algebra is already categorical** — `Pot` handles any number of variable levels. The binary case falls out naturally.

### 2.3 Context-Manager-Based Model Building (PyMC)

The `with pm.Model() as model:` pattern provides declarative syntax but relies on hidden thread-local global state [6]. PyMC's own developers acknowledged this is a learning barrier — Bambi was created because simpler interfaces promote broader adoption [6]. For `ba`: use explicit parameter passing, not context managers.

---

## 3. spyn and odus: Our Packages, Our Foundations

Since spyn and odus are our own packages, we can extend them as needed. This section details what to preserve, what to improve, and what to carry into `ba`.

### 3.1 spyn — The Potential Algebra

spyn's `Pot` class is the most mathematically elegant API in this analysis. Its core insight: a probability table is a mapping from variable assignments to values, and the algebra of probability (conditioning, marginalization, product) maps directly to Python operators.

**Key design patterns to preserve in `ba`:**

- `__mul__` for factor product: `joint = p_a * p_b_given_a`
- `__truediv__` with `[]` for normalization: `normalized = pot / []`
- `__truediv__` with variable names for conditioning: `conditional = joint / joint['X']`
- `__getitem__` for marginalization: `marginal = joint['X']`
- DataFrame-backed storage with MultiIndex for variable assignments
- **Categorical generality** — already handles any number of variable levels

**Proposed improvements to spyn (or extensions in `ba`):**

1. **`.to_contingency(row_var, col_var) → ContingencyTable`**: Bridge from the algebraic world to the metric-computation world. A 2-variable Pot with observed counts should convert directly to a `ContingencyTable` (r×c) or `ContingencyTable2x2` (if both variables are binary).

2. **`.from_contingency(ct) → Pot`**: The reverse bridge. Construct a Pot from a ContingencyTable, enabling round-trips between the algebra layer and the metrics layer.

3. **`.credible_interval(alpha=0.05) → tuple`**: When a Pot represents a posterior (e.g., from Bayesian update with count data), compute HDI or equal-tailed interval directly, without requiring the user to extract parameters and call scipy.

4. **`.data_weight → float`**: When a Pot results from a Bayesian update, track and expose how much the posterior was driven by data vs. prior. This is the ratio $w = n_{\text{obs}} / (n_{\text{obs}} + \text{ESS}_{\text{prior}})$.

5. **`.prior_sensitivity(priors: list[Pot]) → DataFrame`**: Compute the posterior under multiple priors and return a comparison table showing how conclusions shift.

6. **Sparse storage option**: For high-cardinality categoricals, the full joint table can be large. An optional sparse backend (dict-of-counts, or scipy.sparse for the underlying array) would help scale to variables with many levels without changing the API.

7. **Warnings integration**: When a Pot is computed from very small counts (e.g., a cell has 0 or 1 observations), attach warnings to the Pot object that propagate through subsequent operations.

8. **`.metrics(measures='all') → dict`**: Shortcut that converts to ContingencyTable internally, calls the measure registry, and returns results — enabling `joint.metrics(['lift', 'phi'])` as a one-liner.

### 3.2 odus — The Store Pattern and PVar

odus introduces two patterns worth adopting:

**`PVar` (probabilistic variable) as a first-class object** with attribute access: `v = PVar(df); v.heroin` returns a variable handle usable in store lookups, marginalization, etc. Arithmetic on PVars (`v.heroin - 1` for time-lagged variables) is clever and domain-appropriate.

**`pstore` as a lazy-computing Mapping**: `pstore[v.alcohol, v.tobacco]` returns the joint potential on demand, computing and caching from underlying data. This follows the Mapping/MutableMapping pattern.

**What to adapt for `ba`:**

1. **Generalize PVar beyond domain-specific data.** odus's PVar is coupled to drug use data. `ba` should generalize PVar to any variable with metadata: original name, type (binary/categorical/numerical), flip-logic flag for negative-connotation reframing, domain label, and calibration info (for QCA binarization thresholds).

2. **Store with multiple backends.** The store pattern should support: in-memory DataFrame, parquet file, or even a REST API (for the React frontend ↔ Python backend architecture).

3. **Categorical-aware store.** `store.pot(v.treatment, v.outcome)` should auto-detect whether variables are binary or categorical and return the appropriately-typed `Pot`. If both are binary, `store.table2x2(v.treatment, v.outcome)` should be available as a convenience.

4. **Store-level caching.** At $n = 13$, computation is trivial, but for larger datasets the store should cache computed potentials/contingency tables and invalidate on data change.

**Potential improvements to odus itself:**

- Factor out the PVar and store patterns into a reusable module (perhaps within spyn or as a standalone `graze`-like data-access layer).
- Add a `.variables` property that returns a namespace object with attribute access to all column-derived PVars, rather than requiring manual PVar construction.

---

## 4. Community Feedback: Five Recurring Complaints

### 4.1 Bad Defaults Waste Everyone's Time

mlxtend's `use_colnames=False` forces virtually every user to write `use_colnames=True` [1]. scipy's `fisher_exact()` historically returned the unconditional MLE odds ratio while R returns the conditional MLE, causing years of confusion [4]. `ba` must choose defaults that match the most common use case.

### 4.2 Silent Failures Are Worse Than Errors

statsmodels' `mcnemar()` silently truncates tables larger than 2×2 [3]. `ba` must follow QCA's example: context-aware error messages that guide users toward the right function or transformation.

### 4.3 Users Always Need More Interestingness Measures

mlxtend supports 12 measures; R's arules supports 45+ [1][2]. `ba` should adopt arules' registry-based architecture from day one. The registry should declare which table shapes each measure supports, so requesting a 2×2-specific measure on an r×c table produces a helpful error, not a crash.

### 4.4 Prediction and Scoring Are Expected but Missing

mlxtend users request applying mined rules to new data (GitHub #343) [1]. arules provides `predict()` on rules. `ba` should include this.

### 4.5 Overhead for Simple Problems Frustrates Users

PyMC users report 1–15 second initialization for trivial models [6]. For `ba`: conjugate Beta-Binomial / Dirichlet-Multinomial updates are exact and instant via scipy. PyMC should be an **optional dependency** for hierarchical or non-conjugate models only, installed via `pip install ba[pymc]`.

---

## 5. The Result Object and ArviZ's Container Pattern

ArviZ's `InferenceData` — a single container holding named groups (posterior, prior, observed_data, log_likelihood, sample_stats) — is the right model [8]. For `ba`, this becomes:

```python
@dataclass
class AnalysisResult:
    """Single container for all analysis outputs."""
    observed_data: pd.DataFrame               # Original data
    contingency_tables: dict[str, ContingencyTable]  # r×c tables (or 2×2)
    metrics: pd.DataFrame                     # All computed measures
    posterior: dict[str, BayesianResult]       # Per-pair posteriors
    rules: pd.DataFrame | None                # Association rules (if mined)
    truth_table: pd.DataFrame | None          # QCA truth table (if computed)
    qca_solution: QCASolution | None          # Minimized Boolean expression
    metadata: dict                            # Config, warnings, timing

    def summary(self, kind='stats') -> pd.DataFrame: ...
    def save(self, path: str) -> None: ...
    def to_dataframe(self) -> pd.DataFrame: ...
```

**Plot API consistency** (following ArviZ): all `ba.plot_*()` functions share parameters `var_names`, `filter_vars`, `figsize`, `backend`, with `~` prefix for exclusion.

---

## 6. Translating R's Gold Standards

### 6.1 R QCA → Python

R's `calibrate → truthTable → minimize` pipeline translates to:

```python
solution = (ba.qca(df, outcome='SURV')
              .calibrate(DEV={'thresholds': [165, 175, 185]})  # → binary
              .truth_table(incl_cut=0.8, n_cut=1)
              .minimize(include='?')
              .with_details())
```

**Key adaptation:** `calibrate()` is now explicitly the gateway from categorical/numerical to binary. It's not optional boilerplate — it's the deliberate binarization step that QCA requires, and `ba` makes this visible rather than implicit.

QCA's unified API across crisp-set, fuzzy-set, and multi-value variants should be preserved: the same functions auto-detect data type from value ranges.

### 6.2 R arules → Python

arules' `appearance` constraints (which items on LHS/RHS/neither) are critical for targeted mining [2]:

```python
rules = ba.mine_rules(data, min_support=0.1,
                      appearance={'rhs': ['outcome=yes'], 'none': ['id_column']})
```

arules' `interestMeasure()` registry pattern — 45+ measures, computed on demand with cached base counts — is the architecture `ba` needs:

```python
# Compute multiple measures in one call, reusing base counts
quality = ba.measures.compute(rules, ['lift', 'phi', 'conviction', 'fisher_p'])
```

---

## 7. Concrete Design Recommendations

### ADOPT — Carry directly into `ba`

**A1. spyn's potential algebra.** The `Pot` class is `ba`'s mathematical foundation. It's already categorical-capable. Every probabilistic computation should flow through `Pot` operators:

```python
from ba.core import Pot

joint = data.pot('treatment', 'outcome')        # Any cardinality
p_out_given_treat = joint / joint['treatment']   # Conditioning
p_out = joint['outcome']                         # Marginalization
posterior = (likelihood * prior) / []            # Bayesian update
```

**A2. ArviZ's container pattern.** Single `AnalysisResult` object with named groups. Pass one object everywhere.

**A3. arules' extensible measure registry.** Measures as named callables declaring their shape requirements (`requires_2x2`, `requires_ordinal`, etc.). 45+ measures from day one.

**A4. DataFrame-in/DataFrame-out with interchangeable algorithms.**

```python
itemsets = ba.fpgrowth(df, min_support=0.1)   # Categorical-native
itemsets = ba.apriori(df, min_support=0.1)     # Same schema, swap freely
```

**A5. Conjugate Bayesian as fast path.** Beta-Binomial for 2×2, Dirichlet-Multinomial for r×c. Both exact, instant, dependency-light. PyMC only for hierarchical models.

**A6. Consistent plot API vocabulary.** Shared parameters across all plot functions.

### ADAPT — Modify for `ba`'s context

**B1. odus's store pattern — generalized.**

```python
store = ba.DataStore(df)
v = store.vars                                  # Attribute namespace for all columns
joint = store.pot(v.treatment, v.outcome)        # Lazy, cached
table = store.contingency(v.treatment, v.outcome) # → ContingencyTable (r×c or 2×2)
```

**B2. ContingencyTable hierarchy.** `ContingencyTable` (r×c general) → `ContingencyTable2x2` (adds OR, RR, phi, Yule's Q). The 2×2 class inherits from the general class and adds binary-specific metrics. `.as_2x2()` converts with a clear error if not 2×2.

**B3. QCA's pipeline — with explicit calibration gateway.**

```python
# Categorical → binary is explicit, not hidden
binary_df = ba.qca.calibrate(df, {'illness': {'threshold': 'any_present'},
                                   'age': {'threshold': 30}})
solution = ba.qca.minimize(binary_df, outcome='retained_custody')
```

**B4. ArviZ's rcParams — scoped to `ba`.**

```python
ba.config['stats.ci_prob'] = 0.95
ba.config['rules.min_support'] = 0.05
ba.config['qca.incl_cut'] = 0.8
ba.config['bayesian.default_prior'] = 'jeffreys'

with ba.config.context({'stats.ci_prob': 0.89}):
    result.summary()
```

**B5. spyn improvements.** Add `.to_contingency()`, `.from_contingency()`, `.credible_interval()`, `.data_weight`, `.metrics()`, sparse storage, and warnings propagation as described in Section 3.1.

### AVOID — Anti-patterns to keep out of `ba`

**C1. Bad defaults.** Common case must not require extra configuration. `use_colnames=True` equivalent must be the default.

**C2. Silent semantic shifts.** One name, one meaning. Never make `.statistic` mean different things based on input shape.

**C3. Silent truncation.** Never silently discard data or reshape tables. Raise `ValueError` with guidance.

**C4. Hidden global state.** No context managers for model building. Explicit parameter passing always.

**C5. Boilerplate encoding.** `ba.encode_transactions(baskets)` — one line, not three.

**C6. Disconnected pipeline steps.** Pipeline objects carry all metadata needed by downstream steps. No manual `num_itemsets` passing.

**C7. Scattered namespaces.** Organize by paradigm: `ba.bayesian`, `ba.rules`, `ba.qca`, `ba.binary`, not scattered across submodules.

**C8. Heavy required dependencies.** Core = numpy, scipy, pandas, spyn. Everything else optional: `ba[pymc]`, `ba[viz]`, `ba[full]`.

---

## 8. Three-Tier Architecture

### Tier 1: Façade

```python
result = ba.analyze('data.csv', outcome='custody_retained')
result.summary()          # Everything at a glance
result.top_rules(10)      # Most interesting associations
result.plot()             # Default visualization
```

### Tier 2: Paradigm

```python
ba.bayesian.posterior(table, prior='jeffreys')      # Dirichlet-Multinomial
ba.rules.mine(df, min_support=2/13, outcome='Y')    # ARM (categorical-native)
ba.qca.minimize(binary_df, incl_cut=0.8)            # QCA (binary-only)
ba.binary.odds_ratio(table_2x2)                      # 2×2-specific shortcuts
ba.contingency.all_metrics(table)                    # Registry-based (r×c aware)
```

### Tier 3: Primitives

```python
from ba.core import Pot, ContingencyTable, ContingencyTable2x2
from ba.core.metrics import registry as measures
from ba.qca.minimize import quine_mccluskey
from ba.rules.encoding import categorical_to_transactions
```

This three-tier architecture — façade, paradigm, primitive — is the organizational pattern that none of the analyzed libraries fully achieves but that `ba` is positioned to deliver. The categorical core ensures the framework isn't artificially limited, while the binary specialization layer (`ba.binary` + `ba.qca`) provides the focused, efficient API that the motivating research project needs.

---

## References

[1] Raschka S. MLxtend. J Open Source Softw. 2018;3(24):638. https://github.com/rasbt/mlxtend

[2] Hahsler M, Grün B, Hornik K. arules — Mining association rules. J Stat Softw. 2005;14(15). https://cran.r-project.org/package=arules

[3] statsmodels developers. Contingency tables. https://www.statsmodels.org/stable/contingency_tables.html

[4] SciPy developers. scipy.stats. https://docs.scipy.org/doc/scipy/reference/stats.html

[5] Whalen T. spyn — Potentials and probabilistic inference. https://github.com/thorwhalen/spyn

[6] PyMC developers. PyMC. https://github.com/pymc-devs/pymc

[7] Dușa A. QCA with R. 2024. https://bookdown.org/dusadrian/QCAbook/

[8] Kumar R, Carroll C, et al. ArviZ. J Open Source Softw. 2019;4(33). https://github.com/arviz-devs/arviz

[9] Whalen T. odus — Analyses of drug use trajectories. https://github.com/thorwhalen/odus

[10] Hahsler M. arulespy. arXiv:2305.15263. 2023. https://pypi.org/project/arulespy/
