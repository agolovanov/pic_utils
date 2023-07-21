def test_plasma_units_pint():
    from pic_utils.plasma import PlasmaUnits
    import pint
    import numpy as np

    ureg = pint.UnitRegistry()
    pu = PlasmaUnits(1e18 * ureg.cm ** -3)

    np.testing.assert_allclose(pu.wavelength.to('um').magnitude, 33, rtol=0.05)

    pu = PlasmaUnits(1e18 * ureg.cm ** -3, units={'length': 'um'})
    np.testing.assert_allclose(pu.wavelength.magnitude, 33, rtol=0.05)

    densities = np.logspace(10, 25, 100)
    for density in densities:
        pu = PlasmaUnits(density * ureg.cm ** -3)
        np.testing.assert_allclose((pu.wavelength * pu.wavenumber).to('').magnitude, 2 * np.pi)


def test_plasma_units_import():
    """
    Test that module-level import also works
    """
    from pic_utils import PlasmaUnits  # noqa: F401
