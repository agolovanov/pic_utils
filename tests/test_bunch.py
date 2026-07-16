import numpy as np
import pandas as pd
import pint
from hypothesis import given
from hypothesis import strategies as st

from pic_utils.bunch import (
    calculate_bunch_stats,
    calculate_subset_transverse_distributions,
    energy_to_gamma,
    gamma_to_energy,
    generate_gaussian_bunch,
    calculate_spectrum,
    print_bunch_stats,
)


@given(
    gamma=st.floats(allow_nan=False, min_value=1.0, max_value=1e6),
)
def test_gamma_to_energy(gamma):
    np.testing.assert_allclose(gamma, energy_to_gamma(gamma_to_energy(gamma)))


@given(
    particle_number=st.integers(min_value=int(1e4), max_value=int(1e5)),
    energy=st.floats(allow_nan=False, min_value=1, max_value=1e7),
    charge=st.floats(allow_nan=False, min_value=1, max_value=1e7),
    sigma_x=st.floats(allow_nan=False, min_value=0.0, max_value=1e6),
    sigma_y=st.floats(allow_nan=False, min_value=0.0, max_value=1e6),
)
def test_generate_gaussian_bunch(particle_number, energy, charge, sigma_x, sigma_y):
    ureg = pint.get_application_registry()

    energy = energy * ureg.MeV
    charge = charge * ureg.C

    data = generate_gaussian_bunch(particle_number, energy, sigma_x=sigma_x, sigma_y=sigma_y, charge=charge)

    stats = calculate_bunch_stats(data)

    assert stats['propagation_axis'] == 'z'

    np.testing.assert_allclose(sigma_x, stats['sigma_x'].m_as('m'), rtol=0.05, atol=1e-6)
    np.testing.assert_allclose(sigma_y, stats['sigma_y'].m_as('m'), rtol=0.05, atol=1e-6)
    np.testing.assert_allclose(energy.m_as('MeV'), stats['mean_energy'].m_as('MeV'))
    np.testing.assert_allclose(charge.m_as('C'), stats['total_charge'].m_as('C'))


@given(
    particle_number=st.integers(min_value=int(1e4), max_value=int(1e5)),
    energy=st.floats(allow_nan=False, min_value=1, max_value=1e7),
    sigma_x=st.floats(allow_nan=False, min_value=0, max_value=1e6),
    sigma_y=st.floats(allow_nan=False, min_value=0, max_value=1e6),
    rel_energy_spread=st.floats(allow_nan=False, min_value=0.0, max_value=0.05),
    divergence_x=st.floats(allow_nan=False, min_value=0.0, max_value=0.05),
    divergence_y=st.floats(allow_nan=False, min_value=0.0, max_value=0.05),
)
def test_generate_gaussian_bunch_spread(
    particle_number, energy, sigma_x, sigma_y, rel_energy_spread, divergence_x, divergence_y
):
    ureg = pint.get_application_registry()

    energy = energy * ureg.MeV
    energy_spread = energy * rel_energy_spread
    divergence_x = divergence_x * ureg.rad
    divergence_y = divergence_y * ureg.rad

    data = generate_gaussian_bunch(
        particle_number,
        energy,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        energy_spread=energy_spread,
        divergence_x=divergence_x,
        divergence_y=divergence_y,
    )

    stats = calculate_bunch_stats(data)

    assert stats['propagation_axis'] == 'z'

    np.testing.assert_allclose(sigma_x, stats['sigma_x'].m_as('m'), atol=1e-6, rtol=0.05)
    np.testing.assert_allclose(sigma_y, stats['sigma_y'].m_as('m'), atol=1e-6, rtol=0.05)
    np.testing.assert_allclose(energy.m_as('MeV'), stats['mean_energy'].m_as('MeV'), rtol=1e-3)
    np.testing.assert_allclose(energy_spread.m_as('MeV'), stats['energy_spread'].m_as('MeV'), atol=1e-6, rtol=0.05)
    np.testing.assert_allclose(divergence_x.m_as('rad'), stats['prime_sigma_x'].m_as('rad'), atol=1e-6, rtol=0.05)
    np.testing.assert_allclose(divergence_y.m_as('rad'), stats['prime_sigma_y'].m_as('rad'), atol=1e-6, rtol=0.05)


def test_calculate_spectrum_grid():
    x = np.random.normal(0.0, 3, 3000)
    w = np.ones(x.size)

    values, grid = calculate_spectrum(x, w, min_value=-3.0, max_value=3.0)
    values_new, grid_new = calculate_spectrum(x, w, grid=grid)

    assert len(grid) == len(grid_new)
    np.testing.assert_allclose(grid, grid_new)
    np.testing.assert_allclose(values, values_new)


def test_empty_slice_extrema_are_nan():
    particles = pd.DataFrame(
        {
            'x': [1e-6],
            'y': [2e-6],
            'z': [0.0],
            'ux': [0.1],
            'uy': [0.2],
            'uz': [10.0],
            'w': [1.0],
            'energy': [5.0],
        }
    )

    _, stats = calculate_subset_transverse_distributions(particles, 'x', 3, value='energy', value_range=(0.0, 9.0))

    np.testing.assert_array_equal(np.isnan(stats['min_x'].magnitude), [True, False, True])
    np.testing.assert_array_equal(np.isnan(stats['max_x'].magnitude), [True, False, True])
    np.testing.assert_array_equal(np.isnan(stats['pmin_x']), [True, False, True])
    np.testing.assert_array_equal(np.isnan(stats['pmax_x']), [True, False, True])
