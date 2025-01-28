import numpy as np
import typing

if typing.TYPE_CHECKING:
    import pint


def assert_allclose_units(a: 'pint.Quantity', b: 'pint.Quantity', **kwargs):
    """Generalized version of np.testing.assert_allclose for pint.Quantity objects.

    Parameters
    ----------
    a : pint.Quantity
        the first quantity to compare
    b : pint.Quantity
        the second quantity to compare
    """
    from pic_utils.units import split_magnitude_units

    a_magn, a_unit = split_magnitude_units(a)
    b_magn, b_unit = split_magnitude_units(b)

    assert a_unit == b_unit
    np.testing.assert_allclose(a_magn, b_magn, **kwargs)
