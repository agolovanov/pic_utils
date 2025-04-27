import numba
import numpy as np
import pint


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

    mask_high = f_norm[index_max_f:] < 0
    if np.max(mask_high) == 0:
        index_high = len(mask_high) - 1
    else:
        index_high = np.argmax(mask_high)

    mask_low = f_norm[index_max_f::-1] < 0
    if np.max(mask_low) == 0:
        index_low = len(mask_low) - 1
    else:
        index_low = np.argmax(mask_low)

    if interpolate:
        x1 = x[index_max_f:][index_high - 1]
        x2 = x[index_max_f:][index_high]
        f1 = f_norm[index_max_f:][index_high - 1]
        f2 = f_norm[index_max_f:][index_high]

        xhigh = (f2 * x1 - f1 * x2) / (f2 - f1)
        if xhigh > np.max(x):
            xhigh = np.max(x)

        x1 = x[index_max_f::-1][index_low - 1]
        x2 = x[index_max_f::-1][index_low]
        f1 = f_norm[index_max_f::-1][index_low - 1]
        f2 = f_norm[index_max_f::-1][index_low]

        xlow = (f2 * x1 - f1 * x2) / (f2 - f1)
        if xlow < np.min(x):
            xlow = np.min(x)
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


def full_width_at_level_radial(f, r, level, interpolate=True):
    """Calculates the width of function f(x) around the maximum at the specified level.


    Parameters
    ----------
    f : np.array or similar
        function values
    r : np.array or similar
        function coordinates
    level : float
        the level at which to calculate the width
    interpolate : bool, default True
        if True, the value will be calculated using a linear interpolation of the function

        if False, first elements in r corresponding to values < level are returned
    Returns
    -------
    float
        the width or the bounds of the function at the certain level
    """
    if np.min(r) < 0:
        raise ValueError('Negative values of the radial coordinates')

    f_norm = f - level * np.max(f)
    index_max_f = np.argmax(f_norm)

    index = np.argmax(f_norm[index_max_f:] < 0)

    if interpolate:
        r1 = r[index_max_f:][index - 1]
        r2 = r[index_max_f:][index]
        f1 = f_norm[index_max_f:][index - 1]
        f2 = f_norm[index_max_f:][index]

        r_width = (f2 * r1 - f1 * r2) / (f2 - f1)
    else:
        r_width = r[index_max_f:][index]

    return 2 * r_width


def fwhm_radial(f, r):
    """Returns full width at half maximum (FWHM) of function f(r) given by two arrays

    Parameters
    ----------
    f : np.array or similar
        function values
    r : np.array or similar
        function coorindates

    Returns
    -------
    float
        FHWM of f(r)
    """
    return full_width_at_level_radial(f, r, level=0.5)


def mean(distribution, weights, *, total_weight=None):
    if total_weight is None:
        total_weight = weights.sum()
    return (distribution * weights).sum() / total_weight


def calculate_spread(distribution, weights, *, mean_value=None, total_weight=None):
    if total_weight is None:
        total_weight = weights.sum()
    if mean_value is None:
        mean_value = mean(distribution, weights, total_weight=total_weight)

    return np.sqrt(mean((distribution - mean_value) ** 2, weights, total_weight=total_weight))


def mean_spread(distribution, weights, *, total_weight=None):
    mean_value = mean(distribution, weights, total_weight=total_weight)
    spread = calculate_spread(distribution, weights, mean_value=mean_value, total_weight=total_weight)
    return mean_value, spread


def density_1d(values, weights, grid):
    import pint

    if isinstance(grid, pint.Quantity):
        values = values.m_as(grid.units)
        grid = grid.magnitude

    values = np.array(values)
    weights = np.array(weights)

    if values.size != weights.size:
        raise ValueError(f'The sizes of values ({values.size}) and weights ({weights.size}) are not the same')

    return _density_1d_compiled(values, weights, np.min(grid), np.max(grid), len(grid))


@numba.njit
def _density_1d_compiled(values, weights, grid_start, grid_end, grid_size):
    grid_step = (grid_end - grid_start) / (grid_size - 1)
    grid_values = np.zeros(grid_size, dtype=values.dtype)
    v_size = len(values)
    for i in range(v_size):
        v_rel = values[i] - grid_start
        n_cell = int(v_rel // grid_step)
        if n_cell >= -1 and n_cell < grid_size:
            a = v_rel / grid_step - n_cell
            b = 1.0 - a
            w = weights[i] / grid_step
            if n_cell != -1:
                grid_values[n_cell] += b * w
            if n_cell != grid_size - 1:
                grid_values[n_cell + 1] += a * w
    return grid_values


@numba.njit
def _density_2d_compiled(
    grid_values,
    values_x,
    values_y,
    weights,
    grid_start_x,
    grid_end_x,
    grid_size_x,
    grid_start_y,
    grid_end_y,
    grid_size_y,
):
    grid_step_x = (grid_end_x - grid_start_x) / (grid_size_x - 1)
    grid_step_y = (grid_end_y - grid_start_y) / (grid_size_y - 1)
    v_size = len(values_x)
    for i in range(v_size):
        vx_rel = values_x[i] - grid_start_x
        vy_rel = values_y[i] - grid_start_y
        nx = int(vx_rel // grid_step_x)
        ny = int(vy_rel // grid_step_y)

        if nx >= -1 and nx < grid_size_x and ny >= -1 and ny < grid_size_y:
            ax = vx_rel / grid_step_x - nx
            ay = vy_rel / grid_step_y - ny
            bx = 1.0 - ax
            by = 1.0 - ay
            w = weights[i] / grid_step_x / grid_step_y
            if nx != -1 and ny != -1:
                grid_values[nx, ny] += bx * by * w
            if nx != grid_size_x - 1 and ny != -1:
                grid_values[nx + 1, ny] += ax * by * w
            if nx != -1 and ny != grid_size_y - 1:
                grid_values[nx, ny + 1] += bx * ay * w
            if nx != grid_size_x - 1 and ny != grid_size_y - 1:
                grid_values[nx + 1, ny + 1] += ax * ay * w


def density_2d(values_x, values_y, weights, grid_x, grid_y):
    import pint

    if isinstance(grid_x, pint.Quantity):
        values_x = values_x.m_as(grid_x.units)
        grid_x = grid_x.magnitude
    if isinstance(grid_y, pint.Quantity):
        values_y = values_y.m_as(grid_y.units)
        grid_y = grid_y.magnitude

    values_x = np.array(values_x)
    values_y = np.array(values_y)
    weights = np.array(weights)
    x_min = np.min(grid_x)
    x_max = np.max(grid_x)
    grid_size_x = grid_x.size
    y_min = np.min(grid_y)
    y_max = np.max(grid_y)
    grid_size_y = grid_y.size

    if values_x.size != values_y.size:
        raise ValueError(f'The sizes of values_x ({values_x.size}) and values_y ({values_y.size}) are not the same')
    if values_x.size != weights.size:
        raise ValueError(f'The sizes of values_x ({values_x.size}) and weights ({weights.size}) are not the same')

    grid_values = np.zeros((grid_size_x, grid_size_y), dtype=values_x.dtype)

    _density_2d_compiled(grid_values, values_x, values_y, weights, x_min, x_max, grid_size_x, y_min, y_max, grid_size_y)

    return grid_values


def running_mean(values, window_size: int, *, axis: int = 0):
    """Calculates the running mean of the array using the Blackman window function (np.blackman)

    Parameters
    ----------
    values : array
        the inpute array
    window_size : int
        the size of the window for the running
    axis : int, optional
        the direction along which to apply the mean, by default 0

    Returns
    -------
    array
        the mean array with the same shape size as values

    Raises
    ------
    ValueError
        if the window size is larger than the size of the array
    """
    if window_size > values.shape[axis]:
        raise ValueError(f'The window size {window_size} is larger than the size of axis {axis}: {values.shape[axis]}')
    window = np.blackman(window_size)

    if isinstance(values, pint.Quantity):
        values_nodim = values.magnitude
    else:
        values_nodim = values

    def convolution_func(arr):
        return np.convolve(arr, window, mode='same')

    normalization = np.apply_along_axis(convolution_func, axis=axis, arr=np.ones(values_nodim.shape))
    res = np.apply_along_axis(convolution_func, axis=axis, arr=values_nodim) / normalization

    if isinstance(values, pint.Quantity):
        res = res * values.units

    return res


def interpolate_2d(f, x, y, **kwargs):
    """Creates an interpolation function for a 2D function f(x, y) using scipy.interpolate.RegularGridInterpolator

    Parameters
    ----------
    x : np.ndarray
        the 1D array of x coordinates
    y : np.ndarray
        the 1D array of y coordinates
    f : np.ndarray
        the 2D function values. The first axis should correspond to x.
    **kwargs
        additional arguments for RegularGridInterpolator

    Returns
    -------
    callable
        interpolated function of two variables f(x, y)
    """
    from scipy.interpolate import RegularGridInterpolator

    if np.ndim(x) != 1 or np.ndim(y) != 1:
        raise ValueError('The coordinates x and y should be 1D arrays')
    if np.ndim(f) != 2:
        raise ValueError('The values of the function f should be a 2D array')
    if f.shape != (len(x), len(y)):
        raise ValueError('The shape of the function values should be (len(x), len(y))')

    interpolator = RegularGridInterpolator((x, y), f, **kwargs)

    def interpolated_func(x, y):
        return interpolator((x, y))

    return interpolated_func
