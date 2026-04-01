"""DataStore: lazy, cached access to contingency tables and potentials.

>>> import pandas as pd
>>> from ba.store import DataStore
>>> df = pd.DataFrame({
...     'treatment': [1,1,1,0,0,0],
...     'outcome':   [1,1,0,0,0,1],
...     'age_group': [1,0,1,0,1,0],
... })
>>> store = DataStore(df)
>>> store.vars.treatment
'treatment'
>>> ct = store.contingency('treatment', 'outcome')
>>> ct.n
6
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import pandas as pd

from ba.core.contingency import ContingencyTable


class VarNamespace:
    """Attribute namespace for column names with metadata.

    >>> ns = VarNamespace(pd.DataFrame({'age': [1,2,3], 'name': ['a', 'b', 'a']}))
    >>> ns.age
    'age'
    >>> ns.is_binary('age')
    False
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._names = list(df.columns)

    def __getattr__(self, name: str) -> str:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._names:
            return name
        raise AttributeError(
            f"No variable '{name}'. Available: {self._names}"
        )

    def __dir__(self):
        return self._names

    def is_binary(self, name: str) -> bool:
        """Check if a variable has exactly 2 unique values."""
        return self._df[name].nunique() <= 2

    def n_levels(self, name: str) -> int:
        """Number of unique non-null values."""
        return int(self._df[name].nunique())

    def all(self) -> list[str]:
        """All variable names."""
        return list(self._names)

    def binary(self) -> list[str]:
        """Variables with exactly 2 unique values."""
        return [n for n in self._names if self.is_binary(n)]

    def __repr__(self) -> str:
        return f"VarNamespace({self._names})"


class DataStore:
    """Lazy, cached access to contingency tables from a DataFrame.

    >>> import pandas as pd
    >>> store = DataStore(pd.DataFrame({'X': [1,0,1,0], 'Y': [1,1,0,0]}))
    >>> ct = store.contingency('X', 'Y')
    >>> ct.n
    4
    >>> pairs = store.all_pairs()
    >>> len(pairs)
    1
    """

    def __init__(self, data: pd.DataFrame):
        self._data = data
        self._vars = VarNamespace(data)
        # Wrap _contingency_impl so it's per-instance
        self._ct_cache: dict[tuple[str, str], ContingencyTable] = {}

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def vars(self) -> VarNamespace:
        """Attribute namespace for column names."""
        return self._vars

    @property
    def n(self) -> int:
        """Number of rows in the dataset."""
        return len(self._data)

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns)

    def contingency(self, row_var: str, col_var: str) -> ContingencyTable:
        """Contingency table for two variables. Cached.

        >>> import pandas as pd
        >>> store = DataStore(pd.DataFrame({'A': [1,1,0,0], 'B': [1,0,1,0]}))
        >>> ct1 = store.contingency('A', 'B')
        >>> ct2 = store.contingency('A', 'B')
        >>> ct1 is ct2
        True
        """
        key = (row_var, col_var)
        if key not in self._ct_cache:
            self._ct_cache[key] = ContingencyTable.from_dataframe(
                self._data, row_var, col_var
            )
        return self._ct_cache[key]

    def pot(self, *var_names: str):
        """Joint count potential for the given variables. Requires spyn.

        Returns a spyn Pot representing the joint count distribution.

        >>> import pandas as pd
        >>> store = DataStore(pd.DataFrame({'A': [1,1,0,0], 'B': [1,0,1,0]}))
        >>> p = store.pot('A', 'B')
        >>> p.n
        4
        """
        from spyn.ppi.pot import Pot

        cols = list(var_names)
        return Pot.from_points_to_count(self._data[cols])

    def all_pairs(
        self,
        *,
        outcome: str | None = None,
        variables: list[str] | None = None,
    ) -> dict[tuple[str, str], ContingencyTable]:
        """All pairwise contingency tables.

        Args:
            outcome: If given, only pairs involving this column.
            variables: Subset of columns to consider. Default: all.

        >>> import pandas as pd
        >>> store = DataStore(pd.DataFrame({'A': [1,0], 'B': [1,0], 'C': [0,1]}))
        >>> pairs = store.all_pairs()
        >>> len(pairs)
        3
        """
        cols = variables or self.columns
        result = {}

        if outcome:
            if outcome not in cols:
                raise ValueError(f"Outcome '{outcome}' not in columns: {cols}")
            for col in cols:
                if col != outcome:
                    result[(col, outcome)] = self.contingency(col, outcome)
        else:
            for c1, c2 in combinations(cols, 2):
                result[(c1, c2)] = self.contingency(c1, c2)

        return result
