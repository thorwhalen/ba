# ba — Project Instructions

## What ba Is

ba (Bayesian Association) is a Python library for exploring associations in
categorical data, with dedicated support for QCA (Qualitative Comparative
Analysis) and Association Rule Mining (ARM). It emphasizes uncertainty
quantification through Bayesian inference, especially at small sample sizes
(n=10–50).

## Architecture

Three-tier progressive disclosure:

- **Façade:** `ba.analyze(df, outcome='Y')` → `AnalysisResult`
- **Paradigm:** `ba.bayesian`, `ba.rules`, `ba.qca`, `ba.binary`
- **Primitives:** `ContingencyTable`, `MeasureRegistry`, `Pot` bridge

Key design decisions:

- **Categorical core, binary specialization.** The core handles r×c
  contingency tables. Binary (2×2) is a specialization via
  `ContingencyTable2x2`, not the general case.
- **spyn's Pot algebra** (`*`, `/`, `[]`) is the mathematical engine for
  probabilistic reasoning. ba wraps it via `ba.core.pot`.
- **Lightweight core.** Required: numpy, scipy, pandas, spyn. Optional:
  mlxtend (`ba[rules]`), pymc (`ba[pymc]`), matplotlib (`ba[viz]`).

## Module Map

```
ba/
├── __init__.py          # Façade: analyze(), AnalysisResult, config
├── core/
│   ├── contingency.py   # ContingencyTable, ContingencyTable2x2
│   ├── metrics.py       # MeasureRegistry (22 built-in measures)
│   └── pot.py           # Pot ↔ ContingencyTable bridge
├── bayesian/
│   ├── posteriors.py     # Beta-Binomial / Dirichlet-Multinomial
│   ├── priors.py         # Prior construction (5 methods + string resolver)
│   ├── bayes_factors.py  # Gunel-Dickey BFs (4 sampling schemes)
│   └── sensitivity.py    # Multi-prior comparison
├── qca/
│   ├── calibrate.py      # Binarization (float/string/callable thresholds)
│   ├── truth_table.py    # Truth table with validation + flagging
│   ├── minimize.py       # Quine-McCluskey Boolean minimization
│   └── necessity.py      # Necessity/sufficiency with Bayesian CIs
├── rules/
│   ├── encoding.py       # DataFrame → transaction encoding
│   ├── itemsets.py       # Brute-force + mlxtend wrapper
│   └── mining.py         # Rule mining with Bayesian CI augmentation
├── binary/
│   └── shortcuts.py      # odds_ratio(), phi(), yules_q(), etc.
├── store.py              # DataStore: lazy, cached access
├── config.py             # Scoped configuration
├── warnings.py           # Structured small-sample warnings
└── sample_data.py        # custody_data(), market_basket()
```

## Code Conventions

- Python ≥ 3.10. Use `X | Y` union types, `match`/`case` where appropriate.
- `from __future__ import annotations` at the top of each module.
- Type hints on all public methods.
- Doctests for any function where setup is ≤3 lines and the output conveys
  useful information. Use `int()` or `float()` wrappers to avoid
  `np.int64(...)` display issues.
- Warnings attach to result objects (lists of strings), not stderr.
- Never silently discard data or reshape tables — raise `ValueError` with
  guidance toward the correct function.
- New measures go through `MeasureRegistry.register()` with `requires_2x2`
  declared.

## Testing

- Tests live in `tests/`. Run: `python -m pytest tests/ -q`
- Doctests: `python -m pytest --doctest-modules ba/ -q`
- Ruff: `python -m ruff check ba/`

## Key Design Docs

- [design.md](misc/docs/design.md) — Full design specification
- [roadmap.md](misc/docs/roadmap.md) — Implementation status
- [research.md](misc/docs/research.md) — Research takeaways
- [spyn_modernization.md](misc/docs/spyn_modernization.md) — spyn upstream work

## Skills

- **ba-contributor**: Read when modifying ba's code — adding features, fixing
  bugs, extending the API.
- **ba-user**: Read when helping someone *use* ba as a library — writing
  analysis scripts, interpreting results, choosing priors.
