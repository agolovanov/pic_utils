import numpy as np
import pint

from utils import assert_allclose_units

from pic_utils.spectral import fft, fft2

ureg = pint.get_application_registry()


def test_fft_sine():
    x = np.linspace(0, 4 * np.pi, 300)[:-1]
    dx = x[1] - x[0]

    f = np.sin(x) + 0.2 * np.sin(2 * x)
    f_der = (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)

    k, f_img = fft(f, x)

    f_der_img = fft(f_der)

    f_der_img_spectral = 1j * np.sin(k * dx) / dx * f_img

    np.testing.assert_allclose(f_der_img_spectral, f_der_img, atol=1e-5)

    f_der_inv = fft(f_der_img, inverse=True)
    np.testing.assert_allclose(f_der_inv, f_der, atol=1e-5)


def test_fft_padding():
    x = np.linspace(0, 2 * np.pi, 300)[:-1]
    f = np.sin(x) + 0.2 * np.sin(2 * x)

    padding = 3.0
    k, f_img = fft(f, x, padding_factor=padding)
    x_inv, f_inv = fft(f_img, k, padding_factor=padding, inverse=True)

    np.testing.assert_allclose(x_inv, x, atol=1e-5)
    np.testing.assert_allclose(f_inv, f, atol=1e-5)


def test_fft2_padding():
    x = np.linspace(0, 2 * np.pi, 300)[:-1]
    y = np.linspace(0, 4 * np.pi, 401)[:-1]

    xx, yy = np.meshgrid(x, y)

    f = np.sin(xx) + 0.2 * np.sin(2 * yy)

    padding = 3.0
    kx, ky, f_img = fft2(f, x, y, padding_factor=padding)
    x_inv, y_inv, f_inv = fft2(f_img, kx, ky, padding_factor=padding, inverse=True)

    np.testing.assert_allclose(x_inv, x, atol=1e-5)
    np.testing.assert_allclose(y_inv, y, atol=1e-5)
    np.testing.assert_allclose(f_inv, f, atol=1e-5)


def test_fft2_inverse():
    x = np.linspace(-10, 5, 50) * ureg.m
    y = np.linspace(-6, 5, 51) * ureg.m

    f = np.random.rand(51, 50) * ureg['kg/m^2']

    kx, ky, f_fft = fft2(f, x, y)

    assert len(kx) == len(x)
    assert len(ky) == len(y)
    assert f_fft.shape == f.shape

    x_inv, y_inv, f_inv = fft2(f_fft, kx, ky, inverse=True)

    # correction to the origin
    x_inv += x[0] - x_inv[0]
    y_inv += y[0] - y_inv[0]

    assert_allclose_units(x_inv, x, atol=1e-5, rtol=0)
    assert_allclose_units(y_inv, y, atol=1e-5, rtol=0)
    assert_allclose_units(np.real(f_inv.magnitude) * f_inv.units, f)
    np.testing.assert_allclose(np.imag(f_inv.magnitude), np.zeros_like(f_inv.magnitude), atol=1e-5, rtol=0)

    f_inv = fft2(f_fft, inverse=True)
    assert_allclose_units(np.real(f_inv.magnitude) * f_inv.units, f)
    np.testing.assert_allclose(np.imag(f_inv.magnitude), np.zeros_like(f_inv.magnitude), atol=1e-5, rtol=0)
