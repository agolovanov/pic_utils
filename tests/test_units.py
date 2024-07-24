import pint

from pic_utils.units import strip_units, ensure_units
import numpy as np


def test_strip_units():
    ureg = pint.get_application_registry()
    values = [0.05, 5 * ureg.cm, 0.05 * ureg.m]

    results = [strip_units(v, 'm') for v in values]

    np.testing.assert_allclose(results, 0.05)


def test_ensure_units():
    ureg = pint.get_application_registry()
    values = [0.05 * ureg.m, 5.0, 5.0 * ureg.cm]

    results = [ensure_units(v, 'cm').m_as('cm') for v in values]
    np.testing.assert_allclose(results, 5.0)

    results = [ensure_units(v, ureg.cm).m_as('cm') for v in values]
    np.testing.assert_allclose(results, 5.0)
