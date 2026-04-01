# Research

Key takeaways from the reference documents in
[resources/](resources/).
An agent reading this file can decide whether to read the full documents for
more detail.

---

## 1. Statistical Methods for Small-Sample Binary Data

**Source:** [Statistical Methods for Small-Sample Binary Data - A Hierarchical Survey.md](resources/Statistical%20Methods%20for%20Small-Sample%20Binary%20Data%20-%20A%20Hierarchical%20Survey.md)

**Central thesis:** Every statistical method for binary associations operates on
the same 2×2 contingency table. The Bayesian posterior (Beta-Binomial) is the
natural "trunk" from which all other techniques branch.

Key takeaways:

- **Conjugate Beta-Binomial model** gives closed-form posteriors for each row
  proportion. Derived quantities (risk difference, relative risk, odds ratio)
  are computed via Monte Carlo (100k draws, instantaneous).
- **Gunel-Dickey Bayes factors** for 2×2 tables have closed-form expressions
  under all four sampling schemes (Poisson, joint multinomial, independent
  multinomial, hypergeometric). No MCMC needed — just `gammaln`. At n=13, BFs
  typically stay in the "anecdotal" range (BF < 3) unless OR > ~10.
- **Fisher's exact test** is conservative at small n due to discreteness and
  double conditioning. **Barnard's** and **Boschloo's** tests are more powerful
  when only row totals are fixed. All available in scipy.
- **Effect-size CIs:** Baptista-Pike mid-p for OR, Koopman score for RR,
  Agresti-Caffo for RD — all well-calibrated at n ≥ 10.
- **QCA consistency = ARM confidence = P(Y|X).** This identity is the bridge
  across traditions. QCA adds Boolean minimization (equifinality, conjunctural
  causation); ARM adds efficient search (Apriori/FP-growth); Bayes adds honest
  uncertainty.
- **The sparsity wall:** With k=28 binary variables and n=13, higher-order
  joint analysis (k ≥ 3) is impossible without structural assumptions. Naïve
  Bayes (conditional independence) reduces parameters from 2^k to k per class.
- **Prior elicitation:** Mean + concentration (κ) parameterization is the most
  implementation-friendly. "Prior sample size" interpretation makes influence
  transparent. Always show data weight w = n/(n + ESS_prior).

---

## 2. Bayesian Prior Elicitation for Beta Distributions

**Source:** [Bayesian prior elicitation for Beta distributions- a practical survey.md](resources/Bayesian%20prior%20elicitation%20for%20Beta%20distributions-%20a%20practical%20survey.md)

**Central thesis:** Non-statisticians elicit more accurate priors when asked
about observable quantities ("How many out of 20?") rather than abstract
parameters.

Key takeaways:

- **Four elicitation families:** (1) quantile-based (solve CDF equations),
  (2) mean + concentration κ (no optimization needed), (3) roulette/histogram
  (chips in bins — best for non-experts per Goldstein & Rothschild),
  (4) predictive ("How many out of N?").
- **Mean + κ is the simplest "skepticism slider."** Beta(κ·μ, κ·(1−μ)) with
  κ=2 → uniform, κ=50 → strongly informative. No optimization required.
- **Default prior recommendation:** Beta(1,1) as transparent default;
  Beta(2,2) as regularized alternative (prevents boundary estimates). Always
  display ESS alongside sample size.
- **Sensitivity analysis is mandatory at n < 50.** Compute posterior under
  multiple priors; if they diverge, the data is too thin. Power-scaling
  (priorsense) is the state of the art.
- **Production tools:** PreliZ (Python, ArviZ ecosystem) for maxent fitting and
  roulette; SHELF (R) for structured elicitation; distBuilder (JS) for
  browser-based histogram widgets; jStat for JS Beta CDF/PDF.
- **UX design patterns that work:** range slider for credible intervals,
  frequency framing ("X out of 20"), progressive disclosure (simple default →
  one-slider confidence → advanced quantile/histogram on request).

---

## 3. Unified Statistical Framework for Categorical Pattern Analysis

**Source:** [Unified Statistical Framework for Categorical Pattern Analysis -- QCA, Association Rule Mining, and Bayesian Inference.md](resources/Unified%20Statistical%20Framework%20for%20Categorical%20Pattern%20Analysis%20--%20QCA%2C%20Association%20Rule%20Mining%2C%20and%20Bayesian%20Inference.md)

**Central thesis:** The core should handle categorical data (r×c tables), with
binary (2×2) as a specialization. Only QCA and a handful of named metrics are
fundamentally binary; everything else generalizes trivially.

Key takeaways:

- **Contingency table is the universal input** across all three traditions. The
  r×c table is the general case; 2×2 is just r=c=2.
- **Metric equivalence table:** 23+ metrics organized into three tiers —
  general r×c (chi-squared, Cramér's V, mutual info, Fisher-Freeman-Halton),
  ARM-native (support, confidence, lift, conviction, cosine, Jaccard,
  Kulczynski), and binary-only (OR, RR, phi, Yule's Q).
- **Algebraic relationships enable DRY computation:** confidence = QCA
  consistency; lift = confidence / P(Y); cosine = √(conf × coverage);
  Yule's Q = (OR−1)/(OR+1); Cramér's V = |phi| for 2×2.
- **Bayes factors for r×c tables** (Gunel-Dickey) are closed-form in log-space
  via gammaln. All four sampling schemes covered.
- **Dirichlet-Multinomial** generalizes Beta-Binomial seamlessly. Implementing
  the general case covers the binary case for free.
- **Module hierarchy:** core/ (contingency, pot, metrics), bayesian/
  (posteriors, BFs, sensitivity), rules/ (itemsets, encoding, filtering), qca/
  (truth table, minimize, necessity, calibrate), binary/ (shortcuts).
- **At n=13:** categorical variables with >3 levels are essentially
  unanalyzable without collapsing levels or Bayesian pooling. Default ARM
  min_support to 2/n. Never display ARM metrics without credible intervals.

---

## 4. Design Lessons from Nine Statistical Libraries

**Source:** [Design Lessons from Nine Statistical Libraries for Building ba.md](resources/Design%20Lessons%20from%20Nine%20Statistical%20Libraries%20for%20Building%20ba.md)

**Central thesis:** The best statistical libraries use DataFrames as interchange
format, wrap domain concepts in typed objects with algebraic operators, and
separate computation from presentation through progressive disclosure.

Key takeaways:

- **ADOPT from spyn:** The Pot operator algebra (`*` for factor product, `/`
  for normalization/conditioning, `[]` for marginalization) is ba's
  mathematical foundation. Already handles categorical data.
- **ADOPT from ArviZ:** Single `AnalysisResult` container with named groups
  (observed_data, contingency_tables, metrics, posterior, rules, truth_table).
- **ADOPT from arules:** Extensible measure registry (45+ measures). Each
  measure declares its shape requirements (requires_2x2, requires_ordinal).
- **ADOPT from mlxtend:** DataFrame-in/DataFrame-out with interchangeable
  algorithms (apriori/fpgrowth swap with one function change).
- **ADAPT from odus:** Generalized DataStore with PVar namespace for attribute
  access to columns. Lazy, cached pot/contingency computation.
- **ADAPT from statsmodels:** ContingencyTable hierarchy — general r×c with
  lazy computed properties, 2×2 subclass adding OR/RR/phi/Yule's Q.
- **AVOID:** Bad defaults (mlxtend's use_colnames=False), silent failures
  (statsmodels truncating >2×2), hidden global state (PyMC context managers),
  heavy required dependencies.
- **Three-tier architecture:** Façade (`ba.analyze()` → summary),
  Paradigm (`ba.bayesian`, `ba.rules`, `ba.qca`), Primitives (`Pot`,
  `ContingencyTable`, `MeasureRegistry`).
- **Dependencies:** Core = numpy, scipy, pandas, spyn. Optional: pymc (for
  hierarchical models), viz extras, mlxtend (for ARM algorithms).

---

## 5. Visualization for Uncertainty-Aware Binary Data Exploration

**Source:** [Visualization for Uncertainty-Aware Binary Data Exploration.md](resources/Visualization%20for%20Uncertainty-Aware%20Binary%20Data%20Exploration.md)

**Central thesis:** When every estimate has wide intervals, uncertainty must be
the primary visual signal, not an overlay on point estimates.

Key takeaways:

- **VSUP heatmaps** (Value-Suppressing Uncertainty Palette): structurally
  impossible to over-interpret uncertain cells. Best for pairwise overview.
- **Forest plots** ranked by lower CI bound (not point estimate) identify the
  "most confidently positive" effects.
- **HOPs** (Hypothetical Outcome Plots): animated draws from posterior at
  ~400ms/frame. 35-41 percentage points more accurate than error bars for
  probability estimation.
- **Quantile dotplots** (static alternative to HOPs): represent distribution as
  equally-likely dots. Better than PDFs/CDFs for decisions.
- **UpSet plots** for Boolean patterns: which variable combinations actually
  occur in the data (critical at small n where most configurations are empty).
- **Icon arrays** for transparent small-sample display: one icon per case,
  colored by 2×2 cell membership. Forces awareness of how few cases support
  each estimate.
- **Recommended tech stack (frontend):** Visx (React-native SVG) + D3
  (math/scales) + Motion (animation for HOPs) + Zustand (state) +
  use-coordination (linked views) + stdlib-js (Beta PDF/CDF).
- **Prior elicitation UI:** Three approaches in order of complexity —
  (1) interval slider mapped to Beta, (2) roulette/histogram allocation,
  (3) observable-space ("How many out of 30?"). All with real-time posterior
  feedback.
- **Progressive disclosure:** Glance (hover: direction + color) → Tooltip
  (200ms: table + CI + sparkline) → Detail panel (click: full posterior + HOPs
  + mosaic + icon array).
- **Gap:** No production JS/TS library for Bayesian/uncertainty visualization
  exists. ba's frontend must build from lower-level components.

---

## Cross-Cutting Themes

1. **The 2×2 table is the atom.** Every tradition (QCA, ARM, Bayesian)
   computes from the same cell counts. A unified `ContingencyTable` is the
   right foundation.

2. **Categorical core, binary specialization.** Only QCA and a few named
   metrics require binary data. Everything else generalizes to r×c with
   near-zero marginal cost.

3. **Bayesian inference is essential at small n.** Point estimates are
   meaningless when credible intervals span most of [0,1]. Posteriors,
   credible intervals, and Bayes factors should be the default output.

4. **spyn's Pot algebra is the mathematical engine.** Its operator overloading
   (`*`, `/`, `[]`) achieves unmatched notational economy for probabilistic
   inference and already handles arbitrary discrete distributions.

5. **Prior transparency over prior perfection.** Always show ESS, data weight,
   and sensitivity to prior choice. The user should see how much the prior
   matters, not just trust a default.

6. **Progressive disclosure is the UX principle.** Simple things simple
   (`ba.analyze(df)` → summary), complex things possible (custom priors,
   measure registries, QCA minimization with remainder strategies).
