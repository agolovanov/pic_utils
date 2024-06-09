import pint

from pic_utils.units import strip_units
import numpy as np


def test_strip_units():
    ureg = pint.get_application_registry()
    values = [0.05, 5 * ureg.cm, 0.05 * ureg.m]

    results = [strip_units(v, 'm') for v in values]

    np.testing.assert_allclose(results, 0.05)
