import numpy as np
import pint

from pic_utils.units import ensure_units, strip_units, split_magnitude_units

ureg = pint.get_application_registry()


def test_strip_units():
    values = [0.05, 5 * ureg.cm, 0.05 * ureg.m]

    results = [strip_units(v, 'm') for v in values]

    np.testing.assert_allclose(results, 0.05)


def test_ensure_units():
    values = [0.05 * ureg.m, 5.0, 5.0 * ureg.cm]

    results = [ensure_units(v, 'cm').m_as('cm') for v in values]
    np.testing.assert_allclose(results, 5.0)

    results = [ensure_units(v, ureg.cm).m_as('cm') for v in values]
    np.testing.assert_allclose(results, 5.0)

    np.testing.assert_allclose(ensure_units(5.0, None), 5.0)


def test_split_magnitude_units():
    magn = 5.0
    unit = ureg.cm

    magn_new, unit_new = split_magnitude_units(magn * unit)
    np.testing.assert_allclose(magn_new, magn)
    assert unit_new == unit

    magn_new, unit_new = split_magnitude_units(magn)
    np.testing.assert_allclose(magn_new, magn)
    assert unit_new is None
