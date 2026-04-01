"""Configuration with scoped overrides.

>>> import ba
>>> ba.config['stats.ci_prob']
0.95
>>> with ba.config.context({'stats.ci_prob': 0.89}):
...     ba.config['stats.ci_prob']
0.89
>>> ba.config['stats.ci_prob']
0.95
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Iterator


_DEFAULTS = {
    "stats.ci_prob": 0.95,
    "stats.default_prior": "jeffreys",
    "stats.n_mc": 100_000,
    "rules.min_support": 0.05,
    "rules.min_confidence": 0.5,
    "qca.incl_cut": 0.8,
    "qca.n_cut": 1,
    "warnings.small_n_threshold": 30,
}


class Config:
    """Dict-like configuration with context-manager scoping.

    >>> cfg = Config()
    >>> cfg['stats.ci_prob']
    0.95
    >>> cfg['stats.ci_prob'] = 0.99
    >>> cfg['stats.ci_prob']
    0.99
    >>> cfg.reset()
    >>> cfg['stats.ci_prob']
    0.95
    """

    def __init__(self):
        self._values: dict[str, Any] = deepcopy(_DEFAULTS)
        self._stack: list[dict[str, Any]] = []

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._values[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    @contextmanager
    def context(self, overrides: dict[str, Any]) -> Iterator[None]:
        """Temporarily override configuration values.

        >>> cfg = Config()
        >>> with cfg.context({'stats.ci_prob': 0.50}):
        ...     cfg['stats.ci_prob']
        0.5
        >>> cfg['stats.ci_prob']
        0.95
        """
        self._stack.append(deepcopy(self._values))
        self._values.update(overrides)
        try:
            yield
        finally:
            self._values = self._stack.pop()

    def reset(self) -> None:
        """Reset all configuration to defaults."""
        self._values = deepcopy(_DEFAULTS)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._values)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in sorted(self._values.items()))
        return f"Config({items})"
