from hypothesis import given, settings, strategies as st
import numpy as np


@settings(deadline=None)
@given(
    gamma=st.floats(allow_nan=False, min_value=1.0, max_value=1e6),
)
def test_gamma_to_energy(gamma):
    import pint
    from pic_utils.bunch import gamma_to_energy, energy_to_gamma

    ureg = pint.UnitRegistry()

    np.testing.assert_allclose(gamma, energy_to_gamma(gamma_to_energy(gamma, ureg)))
