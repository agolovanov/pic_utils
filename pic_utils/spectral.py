import numpy as np
from scipy import fft as _fft


def fftfreq(x: np.ndarray, k_size: int = None):
    from .units import split_magnitude_units, ensure_units

    if k_size is None:
        k_size = len(x)

    x, x_units = split_magnitude_units(x)
    dx = x[1] - x[0]

    k_units = x_units**-1 if x_units is not None else None
    k = ensure_units(2 * np.pi * _fft.fftshift(_fft.fftfreq(k_size, dx)), k_units)
    return k


def fft(f: np.ndarray, x: np.ndarray = None, *, padding_factor: float = 1.0, inverse: bool = False):
    """Perform 1D FFT over the data.

    Parameters
    ----------
    f : np.ndarray
        data array
    x : np.ndarray, optional
        the x axis, should be of the same size a f
    padding_factor: float, default: 1
        increases the size of FFT transform by a factor by zero-padding the initial array

        For the inverse transform, reduces the size of the output array by this factor.
    inverse: bool, optional
        if True, performs inverse FFT, by default False

    Returns
    -------
    tuple | np.ndarray
        tuple of k, f_fft where k is the k-space axis and f_fft is the FFT of the input data

        If x is not provided, only FFT is returned.
    """
    from .units import split_magnitude_units, ensure_units

    if f.ndim != 1:
        raise ValueError(f'f should be a 1D array, given dimensionality {f.ndim}')

    if x is not None:
        if x.ndim != 1:
            raise ValueError(f'x should be a 1-D array, given dimensionality {x.ndim}')
        if len(x) != len(f):
            raise ValueError(f'The length of x {len(x)} does not match the length of the data {len(f)}')

    if padding_factor < 1.0:
        raise ValueError('padding_factor should be greater than or equal to 1.0')

    f, f_units = split_magnitude_units(f)

    if inverse:
        # remove padding in the inverse transform instead
        k_size = int(round(f.shape[0] / padding_factor))
        f_fft = ensure_units(_fft.ifft(_fft.ifftshift(f)), f_units)[:k_size]
    else:
        k_size = int(round(padding_factor * f.shape[0]))
        f_fft = ensure_units(_fft.fftshift(_fft.fft(f, k_size)), f_units)

    if x is not None:
        if inverse:
            k = fftfreq(x)[:k_size]
            k -= k[0]
        else:
            k = fftfreq(x, k_size)

        return k, f_fft
    else:
        return f_fft


def fft2(
    f: np.ndarray, x: np.ndarray = None, y: np.ndarray = None, *, padding_factor: float = 1.0, inverse: bool = False
):
    """Perform 2D FFT over the data.

    Parameters
    ----------
    f : np.ndarray
        data array
    x : np.ndarray, optional
        the x axis, should be of the same size a f.shape[1]
    y : np.ndarray, optional
        the y axis, should be of the same size a f.shape[0]
    padding_factor: float, optional
        increases the size of FFT transform by a factor by zero-padding the initial array, by default 1.0

        For the inverse transform, reduces the size of the output array by this factor.
    inverse: bool, optional
        if True, performs inverse FFT, by default False

    Returns
    -------
    tuple | np.ndarray
        tuple of kx, ky, f_fft where kx and ky are the k-space axes and f_fft is the 2D FFT of the input data.

        If x and y are not provided, only the 2D FFT is returned.
    """
    from .units import split_magnitude_units, ensure_units

    if f.ndim != 2:
        raise ValueError(f'f should be a 2D array, given dimensionality {f.ndim}')

    if x is not None:
        if x.ndim != 1:
            raise ValueError(f'x should be a 1-D array, given dimensionality {x.ndim}')
        if len(x) != f.shape[1]:
            raise ValueError(f'The length of x {len(x)} does not match the length of the axis {f.shape[1]}')
    if y is not None:
        if y.ndim != 1:
            raise ValueError(f'y should be a 1-D array, given dimensionality {y.ndim}')
        if len(y) != f.shape[0]:
            raise ValueError(f'The length of x {len(y)} does not match the length of the axis {f.shape[0]}')

    if padding_factor < 1.0:
        raise ValueError('padding_factor should be greater than or equal to 1.0')

    f, f_units = split_magnitude_units(f)

    if inverse:
        kx_size = int(round(f.shape[1] / padding_factor))
        ky_size = int(round(f.shape[0] / padding_factor))

        f_fft = ensure_units(_fft.ifft2(_fft.ifftshift(f)), f_units)[:ky_size, :kx_size]
    else:
        kx_size = int(round(padding_factor * f.shape[1]))
        ky_size = int(round(padding_factor * f.shape[0]))

        f_fft = ensure_units(_fft.fftshift(_fft.fft2(f, (ky_size, kx_size))), f_units)

    if x is not None and y is not None:
        if inverse:
            kx = fftfreq(x)[:kx_size]
            ky = fftfreq(y)[:ky_size]
            kx -= kx[0]
            ky -= ky[0]
        else:
            kx = fftfreq(x, kx_size)
            ky = fftfreq(y, ky_size)

        return kx, ky, f_fft
    else:
        return f_fft
