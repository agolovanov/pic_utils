import numpy as np
import pint
from hypothesis import given
from hypothesis import strategies as st

from pic_utils.bunch import calculate_bunch_stats, energy_to_gamma, gamma_to_energy, generate_gaussian_bunch


@given(
    gamma=st.floats(allow_nan=False, min_value=1.0, max_value=1e6),
)
def test_gamma_to_energy(gamma):
    np.testing.assert_allclose(gamma, energy_to_gamma(gamma_to_energy(gamma)))


@given(
    particle_number=st.integers(min_value=int(1e4), max_value=int(1e5)),
    energy=st.floats(allow_nan=False, min_value=1, max_value=1e7),
    charge=st.floats(allow_nan=False, min_value=1, max_value=1e7),
    sigma_x=st.floats(allow_nan=False, min_value=1e-6, max_value=1e6),
    sigma_y=st.floats(allow_nan=False, min_value=1e-6, max_value=1e6),
)
def test_generate_gaussian_bunch(particle_number, energy, charge, sigma_x, sigma_y):
    ureg = pint.get_application_registry()

    energy = energy * ureg.MeV
    charge = charge * ureg.C

    data = generate_gaussian_bunch(particle_number, energy, sigma_x=sigma_x, sigma_y=sigma_y, charge=charge)

    stats = calculate_bunch_stats(data)

    assert stats['propagation_axis'] == 'z'

    np.testing.assert_allclose(sigma_x, stats['sigma_x'].m_as('m'), rtol=0.05)
    np.testing.assert_allclose(sigma_y, stats['sigma_y'].m_as('m'), rtol=0.05)
    np.testing.assert_allclose(energy.m_as('MeV'), stats['mean_energy'].m_as('MeV'))
    np.testing.assert_allclose(charge.m_as('C'), stats['total_charge'].m_as('C'))
