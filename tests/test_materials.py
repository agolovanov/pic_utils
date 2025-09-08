import numpy as np
import pint
from hypothesis import given
from hypothesis import strategies as st
from utils import assert_allclose_units

from pic_utils.materials import Composition, calculate_element_shares, normalize_shares

ureg = pint.get_application_registry()


@given(st.lists(elements=st.floats(min_value=1e-5, max_value=1e5, allow_nan=False), min_size=1, max_size=20))
def test_nomalized_share_inplace(shares):
    shares_normalized_check = np.array(shares) / np.sum(shares)
    share_dict = dict(zip(range(len(shares)), shares))

    normalize_shares(share_dict, inplace=True)

    shares_normalized = np.array(list(share_dict.values()))

    shares_sum = shares_normalized.sum()
    np.testing.assert_almost_equal(shares_sum, 1.0)

    np.testing.assert_allclose(shares_normalized_check, shares_normalized)


def test_calculate_element_shares():
    target = normalize_shares(calculate_element_shares({'helium': 66, 'nitrogen': 33}))

    expected = {'He': 0.5, 'N': 0.5}

    assert len(target) == len(expected)

    for el in target:
        assert el.symbol in expected

        np.testing.assert_almost_equal(target[el], expected[el.symbol])


def test_composition():
    number_density = 1e18
    target = Composition({'helium': 2, 'nitrogen': 1}, number_density=number_density)

    np.testing.assert_almost_equal(target.get_number_density('He'), (2 / 3) * number_density)
    np.testing.assert_almost_equal(target.get_number_density('N'), (2 / 3) * number_density)
    np.testing.assert_almost_equal(target.get_number_density('e'), 0)

    np.testing.assert_approx_equal(target.get_full_ionization_density(), 6 * number_density)


def test_mass_density():
    number_density = 1e18 * ureg.cm**-3
    target = Composition({'helium': 2, 'nitrogen': 1}, number_density=number_density)

    mass_density = target.get_mass_density()
    mass_density_helium = target.get_mass_density('He')
    mass_density_nitrogen = target.get_mass_density('N')

    expected_helium = (4.002602 * ureg('dalton') * (2 / 3) * number_density).to('kg/m^3')
    expeted_nitrogen = (2 * 14.0067 * ureg('dalton') * (1 / 3) * number_density).to('kg/m^3')
    expected = expected_helium + expeted_nitrogen

    assert_allclose_units(mass_density_helium, expected_helium, rtol=1e-4)
    assert_allclose_units(mass_density_nitrogen, expeted_nitrogen, rtol=1e-4)
    assert_allclose_units(mass_density, expected, rtol=1e-4)

    new_target = Composition({'helium': 2, 'nitrogen': 1}, mass_density=mass_density)
    assert_allclose_units(new_target.number_density.to(number_density.units), number_density)
