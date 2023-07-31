import numpy as np


def full_width_at_level(f, x, level, bounds=False):
    """Calculates the width of function f(x) around the maximum at the specified level.


    Parameters
    ----------
    f : np.array or similar
        function values
    x : np.array or similar
        function coordinates
    level : float
        the level at which to calculate the width
    bounds : bool (default False)

    Returns
    -------
    float or tuple of (float, float)
        the width or the bounds of the function at the certain level
    """
    xhigh = x[np.argmax(f):][np.argmax(f[np.argmax(f):] < level * np.max(f))]
    xlow = x[np.argmax(f)::-1][np.argmax(f[np.argmax(f)::-1] < level * np.max(f))]
    if bounds:
        return xlow, xhigh
    else:
        return xhigh - xlow


def fwhm(f, x):
    """Returns full width at half maximum (FWHM) of function f(x) given by two arrays

    Parameters
    ----------
    f : np.array or similar
        function values
    x : np.array or similar
        function coorindates

    Returns
    -------
    float
        FHWM of f(x)
    """
    return full_width_at_level(f, x, level=0.5)
