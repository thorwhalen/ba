---
name: ba-contributor
description: >-
  Guide for contributing code to the ba (Bayesian Association) package.
  Use this skill whenever you're modifying ba source code, adding new measures
  or analysis methods, extending the QCA/ARM/Bayesian layers, writing tests,
  fixing bugs, or working on any file under ba/ba/. Also use when asked to
  implement items from the roadmap, add a new metric to the registry, create
  a new submodule, or refactor existing ba code.
---

# Contributing to ba

## Before You Code

1. Read the project CLAUDE.md for the module map and architecture overview.
2. Check [roadmap.md](../../misc/docs/roadmap.md) to see what's done and what
   remains.
3. Run the existing tests to make sure you're starting from green:
   ```
   python -m pytest tests/ --doctest-modules ba/ -q
   ```

## Adding a New Measure

The measure registry (`ba/core/metrics.py`) is the extension point for all
association/interestingness measures. To add one:

1. Write a function `_my_measure(ct) -> float` that takes a `ContingencyTable`
   and returns a numeric value. Handle edge cases (zero denominators) by
   returning `None` or `0.0`.

2. Register it in the global registry at the bottom of `metrics.py`:
   ```python
   registry.register(
       "my_measure",
       _my_measure,
       requires_2x2=False,  # True if only valid for 2×2 tables
       description="Brief formula or explanation",
   )
   ```

3. Add a test in `tests/test_metrics.py` under `TestMeasureValues` that checks
   against a hand-computed value using the standard fixture
   (a=10, b=5, c=3, d=12, n=30).

4. If the measure is well-known, add a minimal doctest to the function.

## Adding a New Submodule

Follow the existing patterns:

- Subpackage with `__init__.py` that re-exports the public API.
- Each function accepts a `ContingencyTable` or `pd.DataFrame` as primary
  input.
- Return dataclasses or DataFrames, not raw dicts.
- Attach warnings to result objects — don't print to stderr.
- Add the subpackage import to `ba/__init__.py` if it's a top-level paradigm.

## The ContingencyTable Hierarchy

`ContingencyTable` (r×c) is the base. `ContingencyTable2x2` adds binary-only
properties (odds_ratio, phi, yules_q, etc.) and is constructed via:

- `ContingencyTable.from_counts(a, b, c, d)` — direct cell counts
- `ContingencyTable.from_dataframe(df, row_var, col_var)` — auto-detects 2×2
- `ct.as_2x2()` — validates and converts (raises if not 2×2)

When writing code that needs binary-specific features, call `ct.as_2x2()`
which provides a clear error with guidance if the table isn't 2×2.

`ContingencyTable2x2` uses a custom `__init__` (not `__post_init__`) because
dataclass inheritance with required-then-default fields is tricky. If you
modify the 2×2 class, keep this in mind.

## The Pot Bridge

`ba/core/pot.py` connects spyn's algebraic world to ba's metric world:

- `from_contingency(ct) → Pot` — labels are coerced to numeric via
  `_try_numeric`
- `to_contingency(pot, row_var, col_var) → ContingencyTable` — preserves
  data-order of labels (doesn't rely on alphabetical sort)
- `ContingencyTable.to_pot()` — convenience method
- `DataStore.pot(*var_names)` — from raw data

When modifying the bridge, ensure the round-trip
`ct → pot → ct` preserves counts exactly. Test this.

## Bayesian Layer Conventions

- Use scipy for conjugate inference (Beta-Binomial, Dirichlet-Multinomial).
  Never require PyMC for basic operations.
- Monte Carlo: use `np.random.default_rng()` (not `np.random.beta` directly)
  for reproducibility.
- Always compute and expose `data_weight = n / (n + ESS_prior)`.
- Priors resolve through `ba.bayesian.priors.resolve_prior()` which accepts
  strings ('jeffreys', 'uniform', 'beta(a,b)'), tuples, or arrays.

## QCA Layer Conventions

- QCA is the **binary-only** layer. Always validate that input is binary
  (0/1 or True/False) and raise `ValueError` with calibration guidance if not.
- The calibration gateway (`ba.qca.calibrate()`) is the explicit binarization
  step — never silently binarize inside other functions.

## Testing Conventions

- Tests go in `tests/test_<module>.py`.
- Use `pytest.fixture` for reusable test data.
- Standard 2×2 fixture: `a=10, b=5, c=3, d=12` (OR=8.0, n=30).
- Test against hand-computed values with `pytest.approx()`.
- For numpy scalar issues in doctests, wrap with `int()`, `float()`, or
  `bool()`.
- Run both unit tests and doctests before considering work complete.

## Common Gotchas

- `ContingencyTable2x2` row/col labels default to `('1', '0')` — string
  labels, not integers. The Pot bridge's `_try_numeric` handles conversion.
- `pandas.crosstab` sorts levels alphabetically. When label order matters
  (e.g., Pot bridge), use `dict.fromkeys` to preserve data order.
- `np.True_` is not `True` — use `==` not `is` in tests, or wrap with
  `bool()` in doctests.
