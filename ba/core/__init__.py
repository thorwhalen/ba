"""Core data model for ba: contingency tables, metrics, and Pot bridge."""

from ba.core.contingency import ContingencyTable, ContingencyTable2x2
from ba.core.metrics import MeasureRegistry, registry
from ba.core.pot import to_contingency, from_contingency
