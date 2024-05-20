import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st


@settings(deadline=None)
@given(
    gamma=st.floats(allow_nan=False, min_value=1.0, max_value=1e6),
)
def test_gamma_to_energy(gamma):
    import pint

    from pic_utils.bunch import energy_to_gamma, gamma_to_energy

    ureg = pint.UnitRegistry()

    np.testing.assert_allclose(gamma, energy_to_gamma(gamma_to_energy(gamma, ureg)))
