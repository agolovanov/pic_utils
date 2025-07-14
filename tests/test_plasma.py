import numpy as np
import pint
from hypothesis import given
from hypothesis import strategies as st

ureg = pint.get_application_registry()


def test_plasma_units_pint():
    from pic_utils.plasma import PlasmaUnits

    pu = PlasmaUnits(1e18 * ureg.cm**-3)

    np.testing.assert_allclose(pu.wavelength.to('um').magnitude, 33, rtol=0.05)

    pu = PlasmaUnits(1e18 * ureg.cm**-3, units={'length': 'um'})
    np.testing.assert_allclose(pu.wavelength.magnitude, 33, rtol=0.05)

    densities = np.logspace(10, 25, 100)
    for density in densities:
        pu = PlasmaUnits(density * ureg.cm**-3)
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


def test_plasma_units_conversion():
    from pic_utils.plasma import PlasmaUnits
    from utils import assert_allclose_units

    e = ureg['elementary_charge']

    pu = PlasmaUnits(1e18 * ureg.cm**-3)

    np.testing.assert_allclose(pu.convert_to_unitless(pu.wavelength), 2 * np.pi)
    np.testing.assert_allclose(pu.convert_to_unitless(1 / pu.frequency), 1.0)
    np.testing.assert_allclose(pu.convert_to_unitless(e), 1.0)
    np.testing.assert_allclose(pu.convert_to_unitless(pu.energy), 1.0)
    np.testing.assert_allclose(pu.convert_to_unitless(pu.density), 1.0)
    np.testing.assert_allclose(pu.convert_to_unitless(e * pu.density), 1.0)
    np.testing.assert_allclose(pu.convert_to_unitless(pu.E), 1.0)
    np.testing.assert_allclose(pu.convert_to_unitless(pu.B), 1.0)
    np.testing.assert_allclose(pu.convert_to_unitless(pu.potential), 1.0)

    def test_value(value, unit_str):
        assert_allclose_units(pu.convert_to_units(pu.convert_to_unitless(value), unit_str), value)

    test_value(1.0 * ureg.m, 'm')
    test_value(1.0 * ureg.s, 's')
    test_value(1.0 * ureg.C, 'C')
    test_value(1.0 * ureg.J, 'J')
    test_value(1.0 * ureg.cm**-3, '1/cm^3')
    test_value(1.0 * ureg['nC/cm^3'], 'nC/cm^3')
    test_value(1.0 * ureg['V/m'], 'V/m')
    test_value(1.0 * ureg['T'], 'T')
    test_value(1.0 * ureg['V'], 'V')
