from hypothesis import given, strategies as st
import pint
import numpy as np

ureg = pint.UnitRegistry()


def test_plasma_units_pint():
    from pic_utils.plasma import PlasmaUnits

    pu = PlasmaUnits(1e18 * ureg.cm ** -3)

    np.testing.assert_allclose(pu.wavelength.to('um').magnitude, 33, rtol=0.05)

    pu = PlasmaUnits(1e18 * ureg.cm ** -3, units={'length': 'um'})
    np.testing.assert_allclose(pu.wavelength.magnitude, 33, rtol=0.05)

    densities = np.logspace(10, 25, 100)
    for density in densities:
        pu = PlasmaUnits(density * ureg.cm ** -3)
        np.testing.assert_allclose((pu.wavelength * pu.wavenumber).to('').magnitude, 2 * np.pi)


@given(density=st.floats(min_value=1e12, max_value=1e25, allow_nan=False))
def test_plasma_units_init(density):
    from pic_utils.plasma import PlasmaUnits

    density = density * ureg['cm^-3']

    pu_density = PlasmaUnits(density)
    pu_wavelength = PlasmaUnits(pu_density.wavelength)
    pu_frequency = PlasmaUnits(pu_density.frequency)

    for pu_other in [pu_wavelength, pu_frequency]:
        np.testing.assert_allclose(pu_density.density.magnitude, pu_other.density.magnitude)
        np.testing.assert_allclose(pu_density.wavelength.magnitude, pu_other.wavelength.magnitude)
        np.testing.assert_allclose(pu_density.wavenumber.magnitude, pu_other.wavenumber.magnitude)
        np.testing.assert_allclose(pu_density.frequency.magnitude, pu_other.frequency.magnitude)
        np.testing.assert_allclose(pu_density.E.magnitude, pu_other.E.magnitude)
        np.testing.assert_allclose(pu_density.B.magnitude, pu_other.B.magnitude)
        np.testing.assert_allclose(pu_density.charge_density.magnitude, pu_other.charge_density.magnitude)


def test_plasma_units_import():
    """
    Test that module-level import also works
    """
    from pic_utils import PlasmaUnits  # noqa: F401
