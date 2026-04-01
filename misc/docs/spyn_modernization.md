# spyn Modernization Plan

Instructions for modernizing the spyn package (located at
`/Users/thorwhalen/Dropbox/py/proj/t/spyn/`) to be appropriate for
Python 3.10+ and serve as a solid foundation for the `ba` package.

These are upstream contributions to spyn, not changes within ba.

---

## Context

spyn provides the `Pot` class — a probability/likelihood table with operator
overloading for probabilistic inference. It was written ~10-15 years ago and
refreshed ~6 years ago. The core design (DataFrame-backed potentials with `*`,
`/`, `[]` operators) is excellent and should be preserved. The implementation
needs modernization.

**Core file:** `spyn/ppi/pot.py` (~650 lines)

---

## Changes Required

### 1. Remove Wildcard Numpy Import

**File:** `spyn/ppi/pot.py`, line 6

```python
# BEFORE
from numpy import *

# AFTER
import numpy as np
```

Then fix all bare numpy references (`array(...)` → `np.array(...)`,
`nanmin(...)` → `np.nanmin(...)`, etc.) throughout the file.

### 2. Add Type Hints to Public API

Add PEP 484 type annotations to all public methods of `Pot`. Use Python 3.10+
union syntax (`X | Y` instead of `Union[X, Y]`).

Key signatures:

```python
def project_to(self, var_list: list[str] | str, *, assert_subset: bool = False) -> Pot: ...
def normalize(self, var_list: list[str] | str = ()) -> Pot: ...
def get_slice(self, intercept_dict: dict[str, Any]) -> Pot: ...
def __getitem__(self, item: dict | list | tuple | str | None) -> Pot: ...
def __mul__(self, other: Pot | float | int) -> Pot: ...
def __truediv__(self, other: Pot | list | str | dict) -> Pot: ...
def __add__(self, other: Pot | float | int) -> Pot: ...
def assimilate(self, pot: Pot) -> Pot: ...
def pval_of(self, var_val_dict: dict[str, Any], default: float = 0.0) -> float: ...
```

Also add `from __future__ import annotations` at the top of each file for
cleaner forward references.

### 3. Replace `lazyprop` with `functools.cached_property`

**File:** `spyn/util.py` defines a custom `lazyprop` descriptor.

Python 3.8+ has `functools.cached_property` which does the same thing.

```python
# BEFORE (in pot.py)
from spyn.util import lazyprop

class Pot:
    @lazyprop
    def vars(self):
        return [c for c in self.tb.columns if c != 'pval']

# AFTER
from functools import cached_property

class Pot:
    @cached_property
    def vars(self) -> list[str]:
        return [c for c in self.tb.columns if c != 'pval']
```

Note: `cached_property` stores values in the instance `__dict__`, same as
`lazyprop`. Both become stale if `.tb` is mutated — but since we're also fixing
mutation (item 4 below), this is acceptable.

If `lazyprop` is used elsewhere in the codebase, keep it in `util.py` but
deprecate it. Prefer `cached_property` for new code.

### 4. Fix In-Place Mutation in `order_vars`

`order_vars()` currently mutates `self` AND returns `self`. This is confusing
and breaks functional expectations.

```python
# BEFORE
def order_vars(self, var_list=None, sort_pts=True):
    # ... modifies self.tb in place ...
    return self

# AFTER
def order_vars(self, var_list: list[str] | None = None, *, sort_pts: bool = True) -> Pot:
    """Return a new Pot with reordered variables."""
    new_tb = self.tb.copy()
    # ... reorder new_tb ...
    return Pot(new_tb)
```

Same for `sort_pts()` — return new Pot, don't mutate.

### 5. Remove Custom `OrderedSet`

**File:** `spyn/utils/ordered_set.py`

This is a manually-implemented doubly-linked-list `OrderedSet` from an
ActiveState recipe (2009). Since Python 3.7, `dict` preserves insertion order.

Replace all usages with `dict.fromkeys(items)` for ordered deduplication:

```python
# BEFORE
from spyn.utils.ordered_set import OrderedSet
unique = OrderedSet(items)

# AFTER
unique = list(dict.fromkeys(items))
```

Then delete `ordered_set.py`.

### 6. Remove or Fix the Deprecated `|` Operator

`__or__` currently prints a deprecation warning and delegates to `normalize()`.
Either:
- Remove it entirely (breaking change, but it's already deprecated), or
- Raise `DeprecationWarning` properly via `warnings.warn()` instead of `print()`

```python
# BEFORE
def __or__(self, y):
    print("Operator | is deprecated for normalization. Use / instead.")
    return self.normalize(y)

# AFTER (option A: remove)
# Delete __or__ entirely

# AFTER (option B: proper warning)
def __or__(self, y):
    import warnings
    warnings.warn("Pot.__or__ (|) is deprecated. Use / for normalization.",
                  DeprecationWarning, stacklevel=2)
    return self.normalize(y)
```

### 7. Clean Up Debug Print Statements

Line 93 area has a commented-out debug print:
```python
# print('adifasdkfjalsdkjflaksjdlfkjasldkf')
```

Remove all such artifacts.

### 8. Consider `match`/`case` for Polymorphic `__getitem__`

`__getitem__` currently uses a chain of `isinstance` checks. Python 3.10+
structural pattern matching is cleaner:

```python
# BEFORE
def __getitem__(self, item):
    if isinstance(item, dict):
        return self.get_slice(item)
    elif isinstance(item, (list, tuple)):
        return self.project_to(item)
    elif isinstance(item, str):
        return self.project_to([item])
    elif item is None:
        return self.project_to([])
    ...

# AFTER
def __getitem__(self, item: dict | list | tuple | str | None) -> Pot:
    match item:
        case dict():
            return self.get_slice(item)
        case list() | tuple():
            return self.project_to(list(item))
        case str():
            return self.project_to([item])
        case None:
            return self.project_to([])
        case _:
            raise TypeError(f"Unsupported subscript type: {type(item)}")
```

Apply similarly to `__truediv__`.

### 9. Fix Magic Index Assignment

```python
# BEFORE (line 43 area)
self.tb.index = [''] * len(self.tb)

# AFTER
self.tb = self.tb.reset_index(drop=True)
```

The blank-string index is opaque and confusing in debugging. A standard
integer index is clearer.

### 10. Add a Test Suite

spyn has no test files. Add `tests/test_pot.py` with pytest covering:

- Construction from various inputs (DataFrame, dict, Pot, float)
- Projection / marginalization (`[]` with list, string, None)
- Slicing (`[]` with dict)
- Normalization (`/ []`, `/ 'var'`, `/ ['var1', 'var2']`)
- Factor product (`*`)
- Division (`/` with Pot)
- Addition (`+`)
- Assimilate / unassimilate (Bayesian update round-trip)
- Class methods (binary_pot, zero_potential, from_points_to_count, etc.)
- Edge cases: empty Pot, single-row Pot, division by zero handling
- Verify that modified operations (order_vars, sort_pts) return new Pot

### 11. Add Docstrings

Most methods lack documentation. Add Google-style docstrings (per spyn's
pyproject.toml ruff config) to all public methods, at minimum:

- One-line summary
- Args section with types
- Returns section
- Brief example where helpful

---

## Order of Operations

Recommended sequence to minimize breakage:

1. **Tests first** (#10) — write tests against current behavior so regressions
   are caught.
2. **Wildcard import** (#1) — most mechanical, highest impact on code hygiene.
3. **Debug cleanup** (#7) — trivial.
4. **OrderedSet removal** (#5) — straightforward replacement.
5. **lazyprop → cached_property** (#3) — drop-in replacement.
6. **Index fix** (#9) — cosmetic but improves debugging.
7. **Deprecation fix** (#6) — small.
8. **Mutation fix** (#4) — behavioral change, tests must cover this.
9. **Type hints** (#2) — large but non-breaking.
10. **match/case** (#8) — optional polish, requires Python 3.10+.
11. **Docstrings** (#11) — ongoing.

---

## What NOT to Change

- **The operator algebra** (`*`, `/`, `[]`, `>>`) — this is spyn's core value.
  Preserve semantics exactly.
- **DataFrame-backed storage** — the Pot ↔ DataFrame model is correct.
- **The `pval` column convention** — changing this would break all downstream
  code.
- **The `Pot` class name** — it's established in documentation and user code.
