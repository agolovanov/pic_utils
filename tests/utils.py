import numpy as np
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


def assert_dicts_equal(a: dict, b: dict):
    """Assert that two dictionaries are equal.

    Parameters
    ----------
    a : dict
        the first dictionary to compare
    b : dict
        the second dictionary to compare
    """
    assert len(a) == len(b)
    for key in a.keys():
        if isinstance(a[key], dict):
            assert_dicts_equal(a[key], b[key])
        elif isinstance(a[key], str):
            assert a[key] == b[key], f'Key {key} is not equal: {a[key]} != {b[key]}'
        elif a[key] is None:
            assert a[key] == b[key], f'Key {key} is not equal: {a[key]} != {b[key]}'
        elif isinstance(a[key], pint.Quantity):
            assert_allclose_units(a[key], b[key])
        else:
            np.testing.assert_allclose(a[key], b[key])
