
# ba

Bayesian Association — a unified probabilistic framework for exploring
associations in categorical data, with dedicated support for QCA and
Association Rule Mining.

To install:	```pip install ba```

## Claude Code skills

ba ships with two [Claude Code skills](https://docs.anthropic.com/en/docs/claude-code/skills)
in `.claude/skills/`:

- **ba-user** — For anyone *using* ba as a library: choosing priors,
  interpreting Bayes factors, writing QCA workflows, mining rules, handling
  small-sample warnings. Useful prompts that would trigger this skill
  automatically (if it's in scope):
  - *"Analyze my DataFrame for pairwise associations with Bayesian credible intervals"*
  - *"What prior should I use for a dataset with only 15 cases?"*
  - *"Walk me through a QCA analysis with calibration and Boolean minimization"*
- **ba-contributor** — For developers modifying ba's source: adding measures
  to the registry, extending the Bayesian or QCA layers, writing tests,
  understanding the ContingencyTable hierarchy and Pot bridge.

You can invoke a skill explicitly via the slash command:

```
/ba-user Help me set up a prior sensitivity analysis for this 2×2 table
```

To make the skill available in another project, symlink it into that project's
`.claude/skills/` directory:

```bash
# Make ba-user available in a specific project
mkdir -p /path/to/my-project/.claude/skills
ln -s /path/to/ba/.claude/skills/ba-user /path/to/my-project/.claude/skills/ba-user

# Or make it globally available (triggers across all projects)
ln -s /path/to/ba/.claude/skills/ba-user ~/.claude/skills/ba-user
```

See the [skills documentation](https://docs.anthropic.com/en/docs/claude-code/skills)
for more on skill scoping, and the
[Claude Code configuration guide](https://docs.anthropic.com/en/docs/claude-code/settings)
for managing project vs. user-level settings.

## Quick start

```python
import ba

# Analyze all pairwise associations in a DataFrame
result = ba.analyze(df, outcome='Y')
result.summary()          # metrics + Bayesian CIs for every pair
result.top_pairs(5)       # strongest, most certain associations
```

## Single-pair analysis

```python
ct = ba.contingency_table(a=10, b=5, c=3, d=12)
ct.odds_ratio             # 8.0
ct.phi                    # 0.471
ct.metrics(['lift', 'phi', 'cramers_v'])

# Bayesian posterior
post = ba.bayesian.posterior(ct, prior='jeffreys')
post.credible_interval['risk_difference']   # (0.12, 0.71)
post.prob_gt(0.0, 'risk_difference')        # 0.996
```

## Per-tradition APIs

```python
# Bayesian
ba.bayesian.posterior(ct, prior='uniform')
ba.bayesian.bayes_factor(ct)
ba.bayesian.sensitivity(ct, priors=['jeffreys', 'uniform', 'beta(2,2)'])

# QCA
binary_df = ba.qca.calibrate(df, {'age': 30, 'illness': 'any_present'})
tt = ba.qca.truth_table(binary_df, outcome='Y', conditions=['A', 'B', 'C'])
solution = ba.qca.minimize(tt)
ba.qca.necessity(binary_df, 'Y', ['A', 'B'])

# Association Rule Mining
rules = ba.rules.mine(df, min_support=0.1, outcome='Y')
```

## Key features

- **Categorical core, binary specialization.** The core handles r×c contingency
  tables. Binary (2×2) is a specialization with extra metrics (OR, RR, phi,
  Yule's Q, QCA consistency/coverage).
- **Uncertainty first.** Every metric comes with Bayesian credible intervals.
  Bayes factors replace p-values as the default evidence measure.
- **22 built-in measures** in an extensible registry — from support/lift to
  Cramér's V, mutual information, and Goodman-Kruskal gamma.
- **QCA pipeline:** calibrate → truth table → Quine-McCluskey minimization →
  necessity/sufficiency analysis, all with Bayesian CIs.
- **Association rules** with Bayesian augmentation, appearance constraints,
  and built-in brute-force mining (no mlxtend dependency required).
- **Progressive disclosure.** `ba.analyze()` for one-liners;
  `ba.bayesian`/`ba.rules`/`ba.qca` for per-tradition control;
  `ContingencyTable`/`MeasureRegistry` for full access.
