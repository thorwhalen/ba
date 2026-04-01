"""Qualitative Comparative Analysis — binary-only analytical layer.

QCA requires binary data. Use ``calibrate()`` to binarize
categorical/numerical columns before entering the QCA pipeline.

Typical workflow::

    binary_df = ba.qca.calibrate(df, {'age': 30, 'illness': 'any_present'})
    tt = ba.qca.truth_table(binary_df, outcome='Y', conditions=['A', 'B', 'C'])
    solution = ba.qca.minimize(tt)
"""

from ba.qca.calibrate import calibrate
from ba.qca.truth_table import truth_table
from ba.qca.minimize import minimize, QCASolution
from ba.qca.necessity import necessity, sufficiency
