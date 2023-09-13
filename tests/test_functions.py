import numpy as np
from hypothesis import given, strategies as st


@given(
    x0=st.floats(allow_nan=False, min_value=-1e5, max_value=1e5),
    sigma=st.floats(min_value=1e-5, max_value=1e5, allow_nan=False),
    xlow=st.floats(min_value=2.0, max_value=10.0, allow_nan=False),
    xhigh=st.floats(min_value=2.0, max_value=10.0, allow_nan=False),
    number_of_points=st.integers(min_value=100, max_value=10000),
)
def test_fwhm_gaussian(x0, sigma, xlow, xhigh, number_of_points):
    from pic_utils.functions import fwhm

    number_of_points = (int)(number_of_points * (xhigh + xlow))

    x = np.linspace(x0 - xlow * sigma, x0 + xhigh * sigma, number_of_points)
    f = np.exp(- (x - x0) ** 2 / sigma ** 2)

    fwhm_expected = sigma * 2 * np.sqrt(np.log(2))

    assert np.isclose(fwhm(f, x), fwhm_expected, rtol=1e-4)


@given(
    x0=st.floats(allow_nan=False, min_value=-1e5, max_value=1e5),
    sigma=st.floats(min_value=1e-5, max_value=1e5, allow_nan=False),
    xlow=st.floats(min_value=2.0, max_value=10.0, allow_nan=False),
    xhigh=st.floats(min_value=2.0, max_value=10.0, allow_nan=False),
    level=st.floats(min_value=0.05, max_value=0.95, allow_nan=False),
    number_of_points=st.integers(min_value=500, max_value=2000),
)
def test_width_gaussian(x0, sigma, xlow, xhigh, level, number_of_points):
    from pic_utils.functions import full_width_at_level

    number_of_points = (int)(number_of_points * (xhigh + xlow))

    x = np.linspace(x0 - xlow * sigma, x0 + xhigh * sigma, number_of_points)
    f = np.exp(- (x - x0) ** 2 / sigma ** 2)

    width_expected = sigma * 2 * np.sqrt(-np.log(level))

    assert np.isclose(full_width_at_level(f, x, level=level, interpolate=False), width_expected, rtol=3e-2)
    assert np.isclose(full_width_at_level(f, x, level=level, interpolate=True), width_expected, rtol=1e-4)
