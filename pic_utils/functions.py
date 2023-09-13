import numpy as np


def full_width_at_level(f, x, level, bounds=False, interpolate=True):
    """Calculates the width of function f(x) around the maximum at the specified level.


    Parameters
    ----------
    f : np.array or similar
        function values
    x : np.array or similar
        function coordinates
    level : float
        the level at which to calculate the width
    interpolate : bool, default True
        if True, the value will be calculated using a linear interpolation of the function

        if False, first elements in x corresponding to values < level are returned
    bounds : bool (default False)
        if True, returns the tuple of two values (the upper and the lower bounds)

        if False, returns just the width which is the difference between these two values
    Returns
    -------
    float or tuple of (float, float)
        the width or the bounds of the function at the certain level
    """
    f_norm = f - level * np.max(f)
    index_max_f = np.argmax(f_norm)

    index_high = np.argmax(f_norm[index_max_f:] < 0)
    index_low = np.argmax(f_norm[index_max_f::-1] < 0)

    if interpolate:
        x1 = x[index_max_f:][index_high - 1]
        x2 = x[index_max_f:][index_high]
        f1 = f_norm[index_max_f:][index_high - 1]
        f2 = f_norm[index_max_f:][index_high]

        xhigh = (f2 * x1 - f1 * x2) / (f2 - f1)

        x1 = x[index_max_f::-1][index_low - 1]
        x2 = x[index_max_f::-1][index_low]
        f1 = f_norm[index_max_f::-1][index_low - 1]
        f2 = f_norm[index_max_f::-1][index_low]

        xlow = (f2 * x1 - f1 * x2) / (f2 - f1)
    else:
        xhigh = x[index_max_f:][index_high]
        xlow = x[index_max_f::-1][index_low]

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
