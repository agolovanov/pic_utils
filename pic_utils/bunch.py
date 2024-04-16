import numpy as np
import pint
from .functions import mean, mean_spread, calculate_spread  # noqa: F401

import pic_utils.geometry


def gamma(ux, uy, uz):
    """Calculates the Lorentz factor corresponding to the momentum
    """
    return np.sqrt(1 + ux ** 2 + uy ** 2 + uz ** 2)


def energy_to_gamma(energy: pint.Quantity, mass: pint.Quantity = None):
    """Converts particle energy to the Lorentz factor

    Parameters
    ----------
    energy : pint.Quantity
        particle energy
    mass : pint.Quantity, optional
        mass of the particle, by default electron mass

    Returns
    -------
    float
        Lorentz factor
    """
    ureg = energy._REGISTRY
    c = ureg['speed_of_light']
    if mass is None:
        mass = ureg['electron_mass']
    return (energy / mass / c ** 2).m_as('') + 1.0


def gamma_to_energy(gamma, pint_unit_registry: pint.UnitRegistry, mass: pint.Quantity = None,
                    units='MeV') -> pint.Quantity:
    """Converts particle Lorentz factor to its kinetic energy

    Parameters
    ----------
    gamma : float or array
        Lorentz factor of the particle
    pint_unit_registry : pint.UnitRegistry
        Unit registry to be used for conversion
    mass : pint.Quantity, optional
        particle mass, by default electron mass
    units : str, optional
        output unit fo the enegy, by default 'MeV'

    Returns
    -------
    pint.Quantity
        energy in specified units
    """
    c = pint_unit_registry['speed_of_light']
    if mass is None:
        mass = pint_unit_registry['electron_mass']
    return (mass * c ** 2).to(units) * (gamma - 1.0)


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


def propagate_through_magnet_relativistic(data, B0, length, *, axis='x'):
    """Propagates particles through a magnet with a rectangular field distribution assuming that particles are
    relativistic.

    Parameters
    ----------
    data : dict or similar
        the particle data containing keys like 'x', 'ux', 'gamma'
    B0 : pint.Quantity
        the field strength of the magnet
    length : pint.Quantity
        the length of the magnet
    axis : str, optional
        the direction to which the particles are deflected, either 'x' or 'y', by default 'x'
        Note: it's not the direction of the magnetic field strength.

    Returns
    -------
    dict or similar
        the modified data

    Raises
    ------
    ValueError
        if the axis is invalid
    """
    ureg = B0._REGISTRY

    e = ureg['elementary_charge'].to('esu', 'Gau')
    m = ureg['electron_mass']
    c = ureg['speed_of_light']

    B0 = B0.to('gauss', 'Gau')
    z0 = length.m_as('m')
    du = (e * B0 * length / m / c ** 2).m_as('')

    if axis == 'x':
        x, ux = data['x'], data['ux']
        y, uy = data['y'], data['uy']
    elif axis == 'y':
        x, ux = data['y'], data['uy']
        y, uy = data['x'], data['ux']
    else:
        raise ValueError(f'Axis {axis} is invalid, can only be "x" or "y"')

    x += (ux + 0.5 * du) * z0 / data['gamma']
    ux += du
    y += uy * z0 / data['gamma']
    data['z'] += z0
    uz_sqr = data['gamma'] ** 2 - data['ux'] ** 2 - data['uy'] ** 2
    uz_sqr[uz_sqr < 0.0] = 0.0
    data['uz'] = np.sqrt(uz_sqr)

    return data


def propagate_through_magnet(data, B0, length, transverse_max=None, *, axis='x'):
    """Propagates particles through a magnet with a rectangular field distribution.

    Parameters
    ----------
    data : dict or similar
        the particle data containing keys like 'x', 'ux', 'gamma'
    B0 : pint.Quantity
        the field strength of the magnet
    length : pint.Quantity
        the length of the magnet
    transverse_max : pint.Quantity, optional
        the maximum transverse coordinate of the particle in the `axis` direction, by default None
    axis : str, optional
        the direction to which the particles are deflected, either 'x' or 'y', by default 'x'
        Note: it's not the direction of the magnetic field strength.

    Returns
    -------
    dict or similar
        the modified data

    Raises
    ------
    ValueError
        if the axis is invalid
    """
    ureg = B0._REGISTRY

    e = ureg['elementary_charge'].to('esu', 'Gau')
    m = ureg['electron_mass']
    c = ureg['speed_of_light']

    B0 = B0.to('gauss', 'Gau')
    omega0_base = e * B0 / m / c
    k0_base = (omega0_base / c).m_as('1/m')
    omega0_base = omega0_base.m_as('1/s')
    length = length.m_as('m')
    # print(omega0_base, 2 * np.pi * (1 / k0_base) * 1000)

    z0 = np.mean(data['z'])

    if axis == 'x':
        x, ux = 'x', 'ux'
        y, uy = 'y', 'uy'
    elif axis == 'y':
        x, ux = 'y', 'uy'
        y, uy = 'x', 'ux'
    else:
        raise ValueError(f'Axis {axis} is invalid, can only be "x" or "y"')

    omega0 = omega0_base / data['gamma']

    x_center = data[x] + data['uz'] / k0_base
    z_center = z0 - data[ux] / k0_base

    u_perp = np.sqrt(data[ux] ** 2 + data['uz'] ** 2)
    theta_x = np.arctan2(data[ux], data['uz'])

    sin_exit = k0_base / u_perp * (z0 + length - z_center)
    sin_exit[sin_exit > 1.0] = 1.0
    sin_exit[sin_exit < -1.0] = -1.0

    if transverse_max is not None:
        transverse_max = transverse_max.m_as('m')
        if omega0_base > 0:
            cos_max = - k0_base / u_perp * (transverse_max - x_center)
            cos_max[np.abs(cos_max) > 1.0] = 1.0
            sin_max = np.sqrt(1 - cos_max ** 2)
            sin_exit = np.minimum(sin_max, sin_exit)
        else:
            cos_max = - k0_base / u_perp * (- transverse_max - x_center)
            cos_max[np.abs(cos_max) > 1.0] = 1.0
            sin_max = - np.sqrt(1 - cos_max ** 2)
            sin_exit = np.maximum(sin_max, sin_exit)
        data['cos_max'] = cos_max

    data['sin_exit'] = sin_exit

    cos_exit = np.sqrt(1 - sin_exit ** 2)

    t_exit = (np.arcsin(sin_exit) - theta_x) / omega0

    data[x] = x_center - u_perp / k0_base * cos_exit
    data[y] += data[uy] / data['gamma'] * c.m_as('m/s') * t_exit
    data['z'] = z_center + u_perp / k0_base * sin_exit

    data[ux] = u_perp * sin_exit
    data['uz'] = u_perp * cos_exit

    return data
