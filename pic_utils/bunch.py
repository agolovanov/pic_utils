import numpy as np
from .functions import mean, mean_spread, calculate_spread  # noqa: F401

import pic_utils.geometry


def gamma(ux, uy, uz):
    """Calculates the Lorentz factor corresponding to the momentum
    """
    return np.sqrt(1 + ux ** 2 + uy ** 2 + uz ** 2)


def limit_particle_number(data, max_particle_number):
    """Ensures that the number of particles is less than a certain amount.
    Randomly selects and increases the weights of particles if the number is exceeded.

    Parameters
    ----------
    data : pd.DataFrame
        a dataframe of pandas data
    max_particle_number : int
        the maximum number of particles

    Returns
    -------
    pd.DataFrame
        new dataframe with particles no more than required.
    """
    every = (data.shape[0] // max_particle_number) + 1
    if every > 1:
        data = data.iloc[::every].copy()
        data['w'] *= every
    return data


def transverse_distributions(data, axis, ureg, *, total_weight=None):
    weights = data['w']
    if total_weight is None:
        total_weight = weights.sum()

    if axis == 'x':
        x = data['x']
        p = data['ux']
    elif axis == 'y':
        x = data['y']
        p = data['uy']
    else:
        raise ValueError('axis can only be x or y')

    res = {}

    pzmean = mean(data['uz'], weights, total_weight=total_weight)
    res['pzmean'] = pzmean

    xmean = mean(x, weights, total_weight=total_weight)
    pmean = mean(p, weights, total_weight=total_weight)
    xrel = x - xmean
    prel = p - pmean

    x2_mean = mean(xrel ** 2, weights, total_weight=total_weight)
    p2_mean = mean(prel ** 2, weights, total_weight=total_weight)
    xp_mean = mean(xrel * prel, weights, total_weight=total_weight)
    emittance_norm = np.sqrt(x2_mean * p2_mean - xp_mean ** 2)
    emittance = emittance_norm / pzmean

    res['xmean'] = (xmean * ureg.m).to('um')
    res['xsigma'] = (np.sqrt(x2_mean) * ureg.m).to('um')
    res['pmean'] = pmean
    res['psigma'] = np.sqrt(p2_mean)
    res['emittance'] = (emittance * ureg.m).to('mm mrad')
    res['emittance_norm'] = (emittance_norm * ureg.m).to('mm mrad')

    x_prime = p / data['uz']

    x_prime_mean = mean(x_prime, weights, total_weight=total_weight)
    x_prime_rel = x_prime - x_prime_mean

    x_prime2_mean = mean(x_prime_rel ** 2, weights, total_weight=total_weight)
    x_x_prime_mean = mean(xrel * x_prime_rel, weights, total_weight=total_weight)
    emittance_tr = np.sqrt(x2_mean * x_prime2_mean - x_x_prime_mean ** 2)
    emittance_tr_norm = pzmean * emittance_tr

    res['x_prime_mean'] = (x_prime_mean * ureg['']).to('mrad')
    res['x_prime_sigma'] = (np.sqrt(x_prime2_mean) * ureg['']).to('mrad')
    res['emittance_tr'] = (emittance_tr * ureg.m).to('mm mrad')
    res['emittance_tr_norm'] = (emittance_tr_norm * ureg.m).to('mm mrad')
    return res


def spectrum(distribution, weights, *, total_weight=None, min_value=None, max_value=None, step=None, nbins=300):
    import pint

    if min_value is None:
        min_value = np.min(distribution)
    if max_value is None:
        max_value = np.max(distribution)
    if step is not None:
        nbins = int((max_value - min_value) // step) + 1
    if total_weight is None:
        total_weight = np.sum(weights)

    if isinstance(distribution, pint.Quantity):
        unit = distribution.units
        distribution = distribution.magnitude
        min_value = min_value.m_as(unit)
        max_value = max_value.m_as(unit)
    else:
        unit = None

    sp, values = np.histogram(distribution, weights=weights, range=(min_value, max_value), bins=nbins,
                              density=total_weight > 0)
    values = 0.5 * (values[1:] + values[:-1])
    if unit is not None:
        values = values * unit
        sp = sp / unit

    return sp * total_weight, values


def project_to_plane(data, plane: pic_utils.geometry.Plane, plane_coordinates=True):
    """Propagates electrons to a plane

    Parameters
    ----------
    data : dict or similar
        electron parameters
    plane : pic_utils.geometry.Plane
        an object representing a plane
    plane_coordinates : bool, optional
        whether to return the coordinates within the plane or the original x, y, z coordinates, by default True

    Returns
    -------
    tuple
        two arrays corresponding to the plane coordinates if plane_coordinates is True
        three arrays corresponding to x, y, z, otherwise
    """
    n_origin = np.dot(plane.origin, plane.norm)
    n_r = (plane.norm[0] * data['x'] + plane.norm[1] * data['y'] + plane.norm[2] * data['z'])
    n_v = (plane.norm[0] * data['ux'] + plane.norm[1] * data['uy'] + plane.norm[2] * data['uz'])
    tau = (n_origin - n_r) / n_v
    x = data['x'] + data['ux'] * tau
    y = data['y'] + data['uy'] * tau
    z = data['z'] + data['uz'] * tau
    if plane_coordinates:
        x_plane = x * plane.v1[0] + y * plane.v1[1] + z * plane.v1[2] - np.dot(plane.origin, plane.v1)
        y_plane = x * plane.v2[0] + y * plane.v2[1] + z * plane.v2[2] - np.dot(plane.origin, plane.v2)
        return x_plane, y_plane
    else:
        return x, y, z
