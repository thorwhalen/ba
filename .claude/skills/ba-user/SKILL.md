---
name: ba-user
description: >-
  Guide for using the ba (Bayesian Association) library to analyze categorical
  data. Use this skill whenever someone wants to explore associations in a
  dataset, compute contingency table metrics, run Bayesian inference on binary
  or categorical data, perform QCA (Qualitative Comparative Analysis), mine
  association rules, choose priors for small-sample analysis, or interpret
  Bayes factors and credible intervals. Also triggers on questions about
  odds ratios, relative risk, phi coefficient, lift, support, confidence,
  truth tables, Boolean minimization, or prior sensitivity analysis.
---

# Using ba

ba analyzes associations in categorical data with Bayesian uncertainty
quantification. It's especially useful when sample sizes are small (n=10–50)
and point estimates alone are unreliable.

## Quick Start

```python
import ba

# One-liner: analyze all pairwise associations
result = ba.analyze(df, outcome='retained_custody')
result.summary()          # metrics + Bayesian CIs for every pair
result.top_pairs(5)       # strongest, most certain associations
```

## The Three Tiers

### Tier 1: Façade (start here)

```python
# Analyze everything at once
result = ba.analyze(df, outcome='Y')
result.summary()
result.top_pairs(5, sort_by='bayes_factor')

# With association rules
result = ba.analyze(df, outcome='Y', rules=True)
result.top_rules(10)
```

`ba.analyze()` automatically detects variable types, computes all pairwise
contingency tables, runs Bayesian inference with Jeffreys prior, computes
appropriate metrics, and flags small-sample warnings.

### Tier 2: Paradigm (when you know what you want)

```python
# Single contingency table
ct = ba.contingency_table(a=10, b=5, c=3, d=12)
ct.odds_ratio       # 8.0
ct.phi               # 0.471
ct.metrics(['lift', 'phi', 'fisher_p'])

# From a DataFrame
ct = ba.from_dataframe(df, 'treatment', 'outcome')

# Bayesian posterior
post = ba.bayesian.posterior(ct, prior='jeffreys')
post.credible_interval['risk_difference']  # (0.12, 0.71)
post.prob_gt(0.0, 'risk_difference')       # P(RD > 0 | data)

# Bayes factor
bf = ba.bayesian.bayes_factor(ct)  # BF > 1 favors association

# Prior sensitivity
ba.bayesian.sensitivity(ct, priors=['jeffreys', 'uniform', 'beta(2,2)'])

# QCA
binary_df = ba.qca.calibrate(df, {'age': 30, 'illness': 'any_present'})
tt = ba.qca.truth_table(binary_df, 'Y', ['A', 'B', 'C'])
solution = ba.qca.minimize(tt)
ba.qca.necessity(binary_df, 'Y', ['A', 'B'])

# Association rules
rules = ba.rules.mine(df, min_support=0.1, outcome='Y')
```

### Tier 3: Primitives (full control)

```python
from ba.core import ContingencyTable, registry
from ba.core.pot import to_contingency, from_contingency
from ba.bayesian.priors import from_mean_kappa, from_quantiles
```

## Choosing a Prior

For small samples, the prior matters. ba offers several approaches:

| Prior | Code | ESS | When to use |
|-------|------|-----|-------------|
| Jeffreys | `'jeffreys'` | 1 | Default. Minimal influence. |
| Uniform | `'uniform'` | 2 | Transparent, slightly more regularized. |
| Beta(2,2) | `'beta(2,2)'` | 4 | Prevents boundary estimates (0 or 1). |
| Custom mean+strength | `from_mean_kappa(0.3, 10)` | 10 | "I think it's about 30%." |
| Custom interval | `from_quantiles(0.2, 0.05, 0.6, 0.95)` | varies | "I'm 90% sure it's between 0.2 and 0.6." |
| Imaginary data | `from_counts(2, 8)` | 11 | "Imagine 2 successes and 8 failures." |

**Rule of thumb:** Always check `result.data_weight`. If it's below 0.8, the
prior is noticeably influencing results — run `ba.bayesian.sensitivity()`.

## Interpreting Bayes Factors

| BF₁₀ | Evidence |
|-------|----------|
| > 10 | Strong for association |
| 3–10 | Moderate |
| 1–3 | Anecdotal |
| 1/3–1 | Anecdotal for independence |
| < 1/3 | Moderate for independence |

At n=13, BFs typically stay below 10 unless the effect is very large.

## Working with the DataStore

The `DataStore` provides lazy, cached access to contingency tables and
Pot objects from a DataFrame:

```python
store = ba.DataStore(df)
store.vars.treatment          # 'treatment' (attribute access)
store.vars.binary()           # list of binary columns
ct = store.contingency('treatment', 'outcome')  # cached
pairs = store.all_pairs(outcome='Y')            # all pairs with Y

# Pot algebra (requires spyn)
joint = store.pot('treatment', 'outcome')
conditional = joint / 'treatment'   # P(outcome | treatment)
marginal = joint['outcome']         # marginalize to outcome
```

## QCA Workflow

QCA requires binary data. Use `calibrate()` to binarize first:

```python
# 1. Binarize
binary_df = ba.qca.calibrate(df, {
    'age': 30,                    # >= 30 → 1
    'illness': 'any_present',     # truthy → 1
    'score': 'median',            # >= median → 1
    'custom': lambda x: x > 100,  # custom function
})

# 2. Build truth table (choose 3-5 conditions — not all 28!)
tt = ba.qca.truth_table(binary_df, 'Y', ['A', 'B', 'C'], n_cut=2)

# 3. Minimize
solution = ba.qca.minimize(tt)
print(solution.expression)  # e.g., "A*B + ~A*C"

# 4. Necessity/sufficiency
ba.qca.necessity(binary_df, 'Y', ['A', 'B', 'C'])
ba.qca.sufficiency(binary_df, 'Y', ['A', 'B', 'C'])
```

## Small-Sample Warnings

ba generates warnings automatically. Check `result.warnings`:

- **Zero cell:** Point estimates (OR, RR) may be undefined. Use Bayesian posterior.
- **Low expected count (<5):** Chi-squared unreliable. Use Fisher's exact test.
- **Small sample:** Wide credible intervals. Consider Bayesian approach.
- **Prior-dominated (data_weight < 0.5):** The prior drives >50% of the posterior.

## Extending the Measure Registry

```python
# Register a custom measure
def my_measure(ct):
    return ct.counts[0, 0] / ct.n

ba.measures.register('my_metric', my_measure, description='top-left proportion')

# Use it
ct.metrics(['my_metric', 'lift', 'phi'])
```

## Sample Data

```python
from ba.sample_data import custody_data, market_basket

df = custody_data()   # 13 cases, 7 binary columns
df = market_basket()  # 20 transactions, 4 items
```
