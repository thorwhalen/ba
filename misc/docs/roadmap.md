# Roadmap

Implementation plan for `ba`, organized into phases. Each phase delivers a
usable, testable increment. The design specification is in
[design.md](design.md); research notes in [research.md](research.md).

---

## Phase 0: Foundation (Core Data Model)

**Goal:** The atom — ContingencyTable — exists, is tested, and round-trips
with spyn's Pot.

### 0.1 ContingencyTable and ContingencyTable2x2 — DONE

- [x] `ba/core/contingency.py` — dataclass with computed properties (n,
  margins, expected, is_2x2, has_zero_cell, min_expected).
- [x] `as_2x2()` with validation and helpful ValueError.
- [x] `from_dataframe(df, row_var, col_var)` class method — cross-tabulate any
  two columns, auto-detect 2×2.
- [x] `from_counts(a, b, c, d)` class method for quick 2×2 construction.
- [x] `to_dataframe()` — labeled DataFrame representation.
- [x] `summary()` — one-line-per-metric DataFrame.
- [x] Tests: construction, properties, round-trip, error on non-2×2 `as_2x2()`.

### 0.2 Pot Bridge — TODO

- [ ] `ba/core/pot.py` — `to_contingency(pot, row_var, col_var)` and
  `from_contingency(ct)` functions.
- [ ] `ContingencyTable.to_pot()` convenience method.
- [ ] Tests: Pot ↔ ContingencyTable round-trip, verify counts preserved.
- [ ] `DataStore.pot()` method.

**Prerequisite:** spyn modernization (0.4) recommended first, then add spyn as
a dependency in pyproject.toml.

### 0.3 Shared Types — DONE

- [x] `BayesianResult` in `ba/bayesian/posteriors.py`.
- [x] `AnalysisResult` in `ba/__init__.py`.
- [x] `QCASolution` in `ba/qca/minimize.py`.

### 0.4 spyn Modernization (upstream contributions) — TODO

Detailed instructions in
[spyn_modernization.md](spyn_modernization.md).

- [ ] Replace `from numpy import *` with explicit imports.
- [ ] Add type hints to Pot's public API.
- [ ] Replace custom `lazyprop` with `functools.cached_property`.
- [ ] Fix `order_vars()` to return new Pot (not mutate in-place).
- [ ] Remove custom `OrderedSet` (use plain dict, Python 3.7+).
- [ ] Add a test suite for Pot (pytest, covering operators, edge cases).

---

## Phase 1: Metrics and Bayesian Core — DONE

### 1.1 MeasureRegistry — DONE

- [x] `ba/core/metrics.py` — `MeasureRegistry` class with `register()`,
  `compute()`, `available()`, `describe()`.
- [x] 22 built-in measures: 15 general r×c + 7 binary-only.
- [x] Shape-aware dispatch with warning on mismatch.
- [x] `ContingencyTable.metrics()` delegates to global registry.
- [x] Tests: each measure against known values, shape filtering, user
  registration.

### 1.2 Bayesian Posteriors — DONE

- [x] Beta-Binomial (2×2) with MC samples for RD, RR, OR.
- [x] Dirichlet-Multinomial (r×c) per row.
- [x] `BayesianResult` with posterior_params, credible_interval, data_weight,
  mc_samples.
- [x] Tests: conjugate update, CI existence, data_weight, shrinkage,
  `prob_gt()`.

### 1.3 Prior Construction — DONE

- [x] `jeffreys(k)`, `uniform(k)`, `from_mean_kappa()`, `from_quantiles()`,
  `from_counts()`.
- [x] String-based resolution: `'jeffreys'`, `'uniform'`, `'haldane'`,
  `'beta(a,b)'`, `'dirichlet(...)'`.
- [x] Tests: all constructors, resolution, error handling.

### 1.4 Bayes Factors — DONE

- [x] Gunel-Dickey BFs for all 4 sampling schemes.
- [x] Log-space gammaln for r×c.
- [x] Tests: association detection, independence detection, r×c, error handling.

### 1.5 Sensitivity Analysis — DONE

- [x] Multi-prior comparison DataFrame.
- [x] prior_influenced / prior_dominated flags.
- [x] Tests: flag thresholds, custom priors.

---

## Phase 2: Binary Layer and QCA — DONE

### 2.1 Binary Shortcuts — DONE

- [x] `odds_ratio()`, `relative_risk()`, `risk_difference()`, `phi()`,
  `yules_q()` as standalone functions.
- [x] Each validates 2×2 via `as_2x2()`.
- [x] Tests via test_contingency.py (known-value checks).

### 2.2 QCA Calibration — DONE

- [x] `calibrate(df, thresholds)` with float, 'any_present', 'median',
  callable rules.
- [x] Tests: each rule type, error handling.

### 2.3 QCA Truth Table — DONE

- [x] `truth_table()` with binary validation, incl_cut, n_cut, low_n flagging.
- [x] Tests: basic, n_cut filtering, non-binary rejection, flag detection.

### 2.4 Boolean Minimization — DONE

- [x] Quine-McCluskey implementation in `ba/qca/minimize.py`.
- [x] `QCASolution` with expression, prime_implicants, essential_implicants,
  coverage, `to_dataframe()`.
- [x] Tests: simple OR, no positive rows, single minterm.

### 2.5 Necessity/Sufficiency Analysis — DONE

- [x] `necessity()` and `sufficiency()` with Bayesian CIs.
- [x] Auto-detect conditions when not specified.
- [x] Tests: column structure, known values.

---

## Phase 3: Association Rule Mining — DONE

### 3.1 Transaction Encoding — DONE

- [x] `to_transactions(df)` — binary + categorical + optional negation.
- [x] Tests: binary, categorical, mixed, negation.

### 3.2 Frequent Itemset Mining — DONE

- [x] Built-in brute-force for small datasets.
- [x] Auto-fallback to mlxtend fpgrowth/apriori when available.
- [x] Tests: basic, min_support filtering, max_len, algorithm selection.

### 3.3 Rule Mining — DONE

- [x] `mine()` with Bayesian CI augmentation, outcome constraint, appearance
  constraints.
- [x] Tests: basic, bayesian CI, no-bayesian, outcome constraint, empty result.

---

## Phase 4: DataStore and Façade — DONE

### 4.1 DataStore — DONE

- [x] `DataStore(df)` with `vars`, `contingency()`, `all_pairs()`.
- [x] `VarNamespace` with attribute access, `is_binary()`, `n_levels()`,
  `binary()`, `all()`.
- [x] Caching of contingency tables.
- [x] Tests: attribute access, caching, all_pairs, outcome filtering.
- [ ] `pot()` method (pending Phase 0.2).

### 4.2 Configuration — DONE

- [x] `Config` with `[]` access, `context()` manager, `reset()`.
- [x] Default values for all keys.
- [x] Tests: get/set, context scoping, nested contexts, reset, contains.

### 4.3 Warnings — DONE

- [x] `check_table()`, `check_data_weight()`, `check_categorical_sparsity()`,
  `check_truth_table_row()`.
- [x] Warnings attach to result objects, not stderr.
- [x] Tests: each trigger condition.

### 4.4 Façade — DONE

- [x] `ba.analyze(df, outcome, ...)` orchestrator returning `AnalysisResult`.
- [x] `ba.contingency_table()` and `ba.from_dataframe()` convenience.
- [x] `ba.measures` global registry.
- [x] `AnalysisResult` with `summary()`, `top_pairs()`, `top_rules()`.
- [x] Tests: end-to-end, outcome filtering, bayesian on/off, rules on/off,
  categorical data, repr.

---

## Phase 5: Polish and Documentation — DONE

### 5.1 Documentation — DONE

- [x] Module docstrings throughout.
- [x] README with description, install, quickstart, per-tradition examples,
  feature list.

### 5.2 Sample Data — DONE

- [x] `ba.sample_data.custody_data()` — 13 cases, 7 binary columns.
- [x] `ba.sample_data.market_basket()` — 20 transactions, 4 items.

### 5.3 CI/CD — DONE

- [x] 213 tests pass (48 doctests + 165 unit tests).
- [x] Ruff formatting and linting pass.

### 5.4 PyPI Release — PARTIAL

- [x] pyproject.toml dependencies updated (numpy, scipy, pandas).
- [x] Optional dependency groups: `[rules]`, `[pymc]`, `[viz]`, `[full]`.
- [x] Version bumped to 0.3.0.
- [ ] Tag and publish (manual step).

---

## Remaining Work

Two items are deferred, both dependent on spyn:

1. **spyn modernization** — detailed in
   [spyn_modernization.md](spyn_modernization.md). This is upstream work on
   the spyn package, not ba code.

2. **Pot bridge** (Phase 0.2) — `ba/core/pot.py` and `DataStore.pot()`.
   Connects the spyn algebraic world (`*`, `/`, `[]` operators) to ba's
   contingency/metrics world. Best done after spyn modernization.

---

## Dependency Summary

| Phase | New dependencies added |
|-------|----------------------|
| 0 | numpy, scipy, pandas (+ spyn for Phase 0.2) |
| 1 | (none — scipy already present) |
| 2 | (none) |
| 3 | mlxtend (optional, via `ba[rules]`) |
| 4 | (none) |
| 5 | (none new — docs tooling already configured) |
