import numpy as np
from scipy import fft


def fft2(f: np.ndarray, x: np.ndarray, y: np.ndarray, padding_factor: float = 1.0):
    """Perform 2D FFT over the data.

    Parameters
    ----------
    f : np.ndarray
        data array
    x : np.ndarray
        the x axis, should be of the same size a f.shape[1]
    y : np.ndarray
        the y axis, should be of the same size a f.shape[0]
    padding_factor: float
        increases the size of FFT transform by a factor by zero-padding the initial array
    """
    from .units import split_magnitude_units, ensure_units

    f, f_units = split_magnitude_units(f)
    x, x_units = split_magnitude_units(x)
    y, y_units = split_magnitude_units(y)

    kx_units = x_units**-1 if x_units is not None else None
    ky_units = y_units**-1 if x_units is not None else None

    x_size = int(padding_factor * f.shape[1])
    y_size = int(padding_factor * f.shape[0])

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    f_fft = ensure_units(fft.fftshift(fft.fft2(f, (x_size, y_size))), f_units)
    kx = ensure_units(2 * np.pi * fft.fftshift(fft.fftfreq(x_size, dx)), kx_units)
    ky = ensure_units(2 * np.pi * fft.fftshift(fft.fftfreq(y_size, dy)), ky_units)

    return kx, ky, f_fft
