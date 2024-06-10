import numpy as np
import pint
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, array_shapes

from pic_utils.functions import full_width_at_level, full_width_at_level_radial, fwhm, fwhm_radial, running_mean

ureg = pint.get_application_registry()


@given(
    x0=st.floats(allow_nan=False, min_value=-1e5, max_value=1e5),
    sigma=st.floats(min_value=1e-5, max_value=1e5, allow_nan=False),
    xlow=st.floats(min_value=2.0, max_value=10.0, allow_nan=False),
    xhigh=st.floats(min_value=2.0, max_value=10.0, allow_nan=False),
    number_of_points=st.integers(min_value=100, max_value=10000),
)
def test_fwhm_gaussian(x0, sigma, xlow, xhigh, number_of_points):
    number_of_points = (int)(number_of_points * (xhigh + xlow))

    x = np.linspace(x0 - xlow * sigma, x0 + xhigh * sigma, number_of_points)
    f = np.exp(-((x - x0) ** 2) / sigma**2)

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
    number_of_points = (int)(number_of_points * (xhigh + xlow))

    x = np.linspace(x0 - xlow * sigma, x0 + xhigh * sigma, number_of_points)
    f = np.exp(-((x - x0) ** 2) / sigma**2)

    width_expected = sigma * 2 * np.sqrt(-np.log(level))

    assert np.isclose(full_width_at_level(f, x, level=level, interpolate=False), width_expected, rtol=3e-2)
    assert np.isclose(full_width_at_level(f, x, level=level, interpolate=True), width_expected, rtol=1e-4)


@given(
    sigma=st.floats(min_value=1e-5, max_value=1e5, allow_nan=False),
    rmax=st.floats(min_value=2.0, max_value=10.0, allow_nan=False),
    number_of_points=st.integers(min_value=100, max_value=10000),
)
def test_fwhm_radial_gaussian(sigma, rmax, number_of_points):
    number_of_points = (int)(number_of_points * rmax)

    r = np.linspace(0, rmax * sigma, number_of_points)
    f = np.exp(-(r**2) / sigma**2)

    fwhm_expected = sigma * 2 * np.sqrt(np.log(2))

    assert np.isclose(fwhm_radial(f, r), fwhm_expected, rtol=1e-4)


@given(
    sigma=st.floats(min_value=1e-5, max_value=1e5, allow_nan=False),
    rmax=st.floats(min_value=2.0, max_value=10.0, allow_nan=False),
    level=st.floats(min_value=0.05, max_value=0.95, allow_nan=False),
    number_of_points=st.integers(min_value=100, max_value=10000),
)
def test_width_radial_gaussian(sigma, rmax, level, number_of_points):
    number_of_points = (int)(number_of_points * rmax)

    r = np.linspace(0, rmax * sigma, number_of_points)
    f = np.exp(-(r**2) / sigma**2)

    width_expected = sigma * 2 * np.sqrt(-np.log(level))

    assert np.isclose(full_width_at_level_radial(f, r, level=level, interpolate=False), width_expected, rtol=5e-2)
    assert np.isclose(full_width_at_level_radial(f, r, level=level, interpolate=True), width_expected, rtol=5e-3)


@given(
    array=arrays(
        dtype=float,
        shape=array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=1000),
        elements=st.floats(min_value=1e-5, max_value=1e5, allow_nan=False),
    )
)
def test_running_mean_same(array):
    np.testing.assert_allclose(array, running_mean(array, 1))

    array = array * ureg.m
    mean_array = running_mean(array, 1)

    assert array.units == mean_array.units

    np.testing.assert_allclose(array.magnitude, mean_array.magnitude)


@given(
    size=st.integers(min_value=1, max_value=5000),
    window_size=st.integers(min_value=1, max_value=60),
)
def test_running_mean_ones(size, window_size):
    if window_size > size:
        return
    else:
        array = np.ones(size)
        mean_array = running_mean(array, window_size)

        np.testing.assert_allclose(array, mean_array)
