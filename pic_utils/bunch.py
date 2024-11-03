from collections.abc import Iterable

import numpy as np
import pandas as pd
import pint
from typing_extensions import Literal, Tuple

import pic_utils.geometry

from .functions import calculate_spread, mean, mean_spread  # noqa: F401

AxisStr = Literal['x', 'y', 'z']


def gamma(ux, uy, uz):
    """Calculates the Lorentz factor corresponding to the momentum"""
    return np.sqrt(1 + ux**2 + uy**2 + uz**2)


def energy_to_gamma(energy: pint.Quantity, *, mass: pint.Quantity = None):
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
    return (energy / mass / c**2).m_as('') + 1.0


def gamma_to_energy(gamma, *, mass: pint.Quantity = None, units='MeV') -> pint.Quantity:
    """Converts particle Lorentz factor to its kinetic energy

    Parameters
    ----------
    gamma : float or array
        Lorentz factor of the particle
    mass : pint.Quantity, optional
        particle mass, by default electron mass
    units : str, optional
        output unit fo the enegy, by default 'MeV'

    Returns
    -------
    pint.Quantity
        energy in specified units
    """
    ureg = pint.get_application_registry()

    c = ureg['speed_of_light']
    if mass is None:
        mass = ureg['electron_mass']

    if isinstance(gamma, pd.Series):
        gamma = gamma.to_numpy()
    return (mass * c**2).to(units) * (gamma - 1.0)


def initialize_energy(data, *, mass: pint.Quantity = None, units='MeV'):
    """Add the 'gamma' and 'energy' field in the bunch data based on momenta 'ux', 'uy', 'uz'.

    Parameters
    ----------
    data : dict-like
        a data-frame of particle data with keys which can be updated
    mass : pint.Quantity, optional
        mass of the particle, by default None (electron mass)
    units : str, optional
        units to initialize the energy in, by default 'MeV'
    """
    data['gamma'] = gamma(data['ux'], data['uy'], data['uz'])
    data['energy'] = gamma_to_energy(data['gamma'], mass=mass, units=units).m_as(units)


def limit_particle_number(data: pd.DataFrame, max_particle_number: int):
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


def transverse_distributions(
    data: dict,
    axis: AxisStr | Iterable[AxisStr],
    *,
    total_weight: float | None = None,
    suffix: str | None = None,
    propagation_axis: AxisStr = 'z',
) -> dict:
    ureg = pint.get_application_registry()

    res = {}

    if axis == 'x' or axis == 'y' or axis == 'z':
        if axis == propagation_axis:
            raise ValueError(f'Axis {axis} coincides with the propagation axis {propagation_axis}')
        x = data[f'{axis}']
        p = data[f'u{axis}']
    elif isinstance(axis, Iterable) and len(axis) > 1:
        for ax in axis:
            if suffix is not None:
                suffix = suffix + f'_{ax}'
            res.update(
                transverse_distributions(
                    data, ax, total_weight=total_weight, suffix=suffix, propagation_axis=propagation_axis
                )
            )
        return res
    else:
        raise ValueError(f'axis can only be x, y, z or a sequence of those, {axis} is wrong')

    weights = data['w']
    if total_weight is None:
        total_weight = weights.sum()

    if suffix is None:
        suffix = f'_{axis}'

    u_long = data[f'u{propagation_axis}']
    u_long_mean = mean(u_long, weights, total_weight=total_weight)

    xmean = mean(x, weights, total_weight=total_weight)
    pmean = mean(p, weights, total_weight=total_weight)
    xrel = x - xmean
    prel = p - pmean

    x2_mean = mean(xrel**2, weights, total_weight=total_weight)
    p2_mean = mean(prel**2, weights, total_weight=total_weight)
    xp_mean = mean(xrel * prel, weights, total_weight=total_weight)
    emittance_norm = np.sqrt(x2_mean * p2_mean - xp_mean**2)
    emittance = emittance_norm / u_long_mean

    res[f'mean{suffix}'] = (xmean * ureg.m).to('um')
    res[f'sigma{suffix}'] = (np.sqrt(x2_mean) * ureg.m).to('um')
    res[f'pmean{suffix}'] = pmean
    res[f'psigma{suffix}'] = np.sqrt(p2_mean)
    res[f'emittance{suffix}'] = (emittance * ureg.m).to('mm mrad')
    res[f'emittance_norm{suffix}'] = (emittance_norm * ureg.m).to('mm mrad')

    x_prime = p / u_long

    x_prime_mean = mean(x_prime, weights, total_weight=total_weight)
    x_prime_rel = x_prime - x_prime_mean

    x_prime2_mean = mean(x_prime_rel**2, weights, total_weight=total_weight)
    x_x_prime_mean = mean(xrel * x_prime_rel, weights, total_weight=total_weight)
    emittance_tr = np.sqrt(x2_mean * x_prime2_mean - x_x_prime_mean**2)
    emittance_tr_norm = u_long_mean * emittance_tr

    res[f'prime_mean{suffix}'] = (x_prime_mean * ureg['']).to('mrad')
    res[f'prime_sigma{suffix}'] = (np.sqrt(x_prime2_mean) * ureg['']).to('mrad')
    res[f'emittance_tr{suffix}'] = (emittance_tr * ureg.m).to('mm mrad')
    res[f'emittance_tr_norm{suffix}'] = (emittance_tr_norm * ureg.m).to('mm mrad')

    return res


def calculate_spectrum(
    distribution, weights, *, total_weight=None, min_value=None, max_value=None, step=None, nbins=300, grid=None
):
    import pint

    if grid is not None:
        nbins = len(grid)
        step = grid[1] - grid[0]
        min_value = np.min(grid) - 0.5 * step
        max_value = np.max(grid) + 0.5 * step
    else:
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

    sp, values = np.histogram(
        distribution, weights=weights, range=(min_value, max_value), bins=nbins, density=total_weight > 0
    )
    values = 0.5 * (values[1:] + values[:-1])
    if unit is not None:
        values = values * unit
        sp = sp / unit

    return sp * total_weight, values


def spectrum(
    distribution, weights, *, total_weight=None, min_value=None, max_value=None, step=None, nbins=300, grid=None
):
    from warnings import warn

    warn('Use calculate_spectrum instead', DeprecationWarning)
    return calculate_spectrum(
        distribution,
        weights,
        total_weight=total_weight,
        min_value=min_value,
        max_value=max_value,
        step=step,
        nbins=nbins,
        grid=grid,
    )


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
    n_r = plane.norm[0] * data['x'] + plane.norm[1] * data['y'] + plane.norm[2] * data['z']
    n_v = plane.norm[0] * data['ux'] + plane.norm[1] * data['uy'] + plane.norm[2] * data['uz']
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
    du = (e * B0 * length / m / c**2).m_as('')

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
            cos_max = -k0_base / u_perp * (transverse_max - x_center)
            cos_max[np.abs(cos_max) > 1.0] = 1.0
            sin_max = np.sqrt(1 - cos_max**2)
            sin_exit = np.minimum(sin_max, sin_exit)
        else:
            cos_max = -k0_base / u_perp * (-transverse_max - x_center)
            cos_max[np.abs(cos_max) > 1.0] = 1.0
            sin_max = -np.sqrt(1 - cos_max**2)
            sin_exit = np.maximum(sin_max, sin_exit)
        data['cos_max'] = cos_max

    data['sin_exit'] = sin_exit

    cos_exit = np.sqrt(1 - sin_exit**2)

    t_exit = (np.arcsin(sin_exit) - theta_x) / omega0

    data[x] = x_center - u_perp / k0_base * cos_exit
    data[y] += data[uy] / data['gamma'] * c.m_as('m/s') * t_exit
    data['z'] = z_center + u_perp / k0_base * sin_exit

    data[ux] = u_perp * sin_exit
    data['uz'] = u_perp * cos_exit

    return data


def propagate_through_thin_quadrupole(data, focal_length, design_energy):
    """Calculates the result of particle interaction with a thin quadrupole

    Parameters
    ----------
    data : dict or similar
        the particle data containing keys like 'x', 'ux', 'gamma'
    focal_length : pint.Quantity
        the focal length of the quadrupole. Positive values correspond to focussing in x, negative in y.
    design_energy : pint.Quantity
        the particle energy for which the focal length was calculated
    """
    gamma_design = energy_to_gamma(design_energy)

    focal_length = focal_length.m_as('m')

    data['ux'] -= gamma_design / focal_length * data['x']
    data['uy'] += gamma_design / focal_length * data['y']


def filter_by_pinhole(
    data: pd.DataFrame, radius: pint.Quantity, plane: pic_utils.geometry.Plane | float = None
) -> pd.DataFrame:
    """Filters the bunch by a pinhole.

    Parameters
    ----------
    data : pd.Dataframe
        the particle data containing keys like 'x', 'ux', 'gamma'
    radius : pint.Quantity
        the radius of the pinhole
    plane : pic_utils.geometry.Plane | float, optional
        if given, the bucnh will first be projected to the pinhole plane (only perpendicular to z is supported), by default None

    Returns
    -------
    pd.DataFrame
        filtered particle data
    """
    radius = radius.m_as('m')

    if plane is not None:
        if not isinstance(plane, pic_utils.geometry.Plane):
            plane = pic_utils.geometry.coordinate_plane('z', plane)
        if not plane.check_normal([0, 0, 1]):
            raise ValueError(f'Only planes perpendicular to z are supported, the plane has a normal [{plane.norm}]')

        data['x'], data['y'], data['z'] = project_to_plane(data, plane, plane_coordinates=False)

    r = np.sqrt(data['x'] ** 2 + data['y'] ** 2)
    return data[r < radius].copy(deep=True)


def format_mean_spread(mean, spread):
    try:
        if mean.units != spread.units:
            raise ValueError(f'Units {mean.units} and {spread.units} are not the same')
        return f'{mean.magnitude:.3g} ± {spread.magnitude:.3g} {mean.units:~}'
    except:
        return f'{mean:.3g} ± {spread:.3g}'


def calculate_bunch_stats(particles: pd.DataFrame, propagation_axis: AxisStr | None = None):
    ureg = pint.get_application_registry()

    e = ureg['elementary_charge']
    c = ureg['speed_of_light']

    energies = particles['energy'].to_numpy() * ureg.MeV
    weights = particles['w'].to_numpy()

    total_weight = np.sum(weights)
    total_charge = (total_weight * e).to('pC')
    total_energy = np.sum(weights * energies).to('mJ')

    min_energy = np.min(energies)
    max_energy = np.max(energies)
    mean_energy, energy_spread = mean_spread(energies, weights, total_weight=total_weight)

    if propagation_axis is None:
        propagation_axis, mean_u = find_propagation_axis(particles, total_weight=total_weight, return_mean_momenta=True)
        ulong_mean = mean_u[f'u{propagation_axis}']
    else:
        ulong_mean = mean(particles[f'u{propagation_axis}'], weights, total_weight=total_weight)

    transverse_axes = get_transverse_axes(propagation_axis)

    stats = transverse_distributions(
        particles, transverse_axes, total_weight=total_weight, propagation_axis=propagation_axis
    )

    long_mean, long_sigma = mean_spread(particles[propagation_axis], weights, total_weight=total_weight)
    long_mean = (long_mean * ureg.m).to('um')
    long_sigma = (long_sigma * ureg.m).to('um')
    long_duration = (long_sigma / c).to('fs')

    stats.update(
        {
            'total_charge': total_charge,
            'total_energy': total_energy,
            'energies': energies,
            'weights': weights,
            'particle_number': len(particles.index),
            'min_energy': min_energy,
            'max_energy': max_energy,
            'mean_energy': mean_energy,
            'energy_spread': energy_spread,
            'propagation_axis': propagation_axis,
            'transverse_axes': transverse_axes,
            'long_mean': long_mean,
            'long_sigma': long_sigma,
            'long_duration': long_duration,
            'ulong_mean': ulong_mean,
        }
    )

    return stats


def print_bunch_stats(stats: dict | pd.DataFrame):
    if 'particle_number' not in stats:
        # the stats variable is a particle bunch
        stats = calculate_bunch_stats(stats)

    print(f'Propagation axis of the bunch: {stats["propagation_axis"]}')

    print(f'Number of particles: {stats["particle_number"]}')
    print(f'Charge {stats["total_charge"]:.3g#~}, energy {stats["total_energy"]:.3g#~}')
    print(
        f'Particle energy: min {stats["min_energy"]:.3g#~}, max {stats["max_energy"]:.3g#~}, '
        f'mean: {stats["mean_energy"]:.3g#~}, spread: {stats["energy_spread"]:.3g#~}'
    )

    ax1, ax2 = stats['transverse_axes']

    print(
        f'Coordinates: {ax1} = {format_mean_spread(stats[f"mean_{ax1}"], stats[f"sigma_{ax1}"])}, '
        f'{ax2} = {format_mean_spread(stats[f"mean_{ax2}"], stats[f"sigma_{ax2}"])}'
    )
    print(
        f'             {stats["propagation_axis"]} = {format_mean_spread(stats["long_mean"], stats["long_sigma"])} '
        f'(duration {stats["long_duration"]:.3g#~})'
    )

    print(
        f'Momenta: u{ax1} = {format_mean_spread(stats[f"pmean_{ax1}"], stats[f"psigma_{ax1}"])}, '
        f'u{ax2} = {format_mean_spread(stats[f"pmean_{ax2}"], stats[f"psigma_{ax2}"])}, '
        f'u{stats["propagation_axis"]} = {stats["ulong_mean"]:.3g}'
    )
    print(f'Pointing angle: {ax1} = {stats[f"prime_mean_{ax1}"]:.3g~}, {ax2} = {stats[f"prime_mean_{ax2}"]:.3g~}')
    print(f'Divergence: {ax1} = {stats[f"prime_sigma_{ax1}"]:.3g~}, {ax2} = {stats[f"prime_sigma_{ax2}"]:.3g~}')
    print(f'Emittance: {ax1} {stats[f"emittance_tr_{ax1}"]:.3g~} (tr), {stats[f"emittance_{ax1}"]:.3g~} (ph)')
    print(f'           {ax2} {stats[f"emittance_tr_{ax2}"]:.3g~} (tr), {stats[f"emittance_{ax2}"]:.3g~} (ph)')
    print(
        f'Norm. emittance: {ax1} {stats[f"emittance_tr_norm_{ax1}"]:.3g~} (tr), {stats[f"emittance_norm_{ax1}"]:.3g~} (ph)'
    )
    print(
        f'                 {ax2} {stats[f"emittance_tr_norm_{ax2}"]:.3g~} (tr), {stats[f"emittance_norm_{ax2}"]:.3g~} (ph)'
    )

    return stats


def find_propagation_axis(
    particles: dict, *, total_weight: float | None = None, return_mean_momenta: bool = False
) -> AxisStr | Tuple[AxisStr, dict]:
    """Finds the propagation axis of the bunch

    Parameters
    ----------
    particles : dict
        bunch data which must contain keys 'ux', 'uy', 'uz', 'w'
    total_weight : float | None, optional
        the total weight of the bunch, by default None
    return_mean_momenta : bool, optional
        whether to add the mean momenta to the output, by default False

    Returns
    -------
    AxisStr | Tuple[AxisStr, dict]
        either an axis or a tuple of an axis and a dict of mean 'ux', 'uy', 'uz' values
    """
    if total_weight is None:
        total_weight = particles['w'].sum()

    axes = ['x', 'y', 'z']
    momenta_axes = [f'u{ax}' for ax in axes]
    mean_u = {u: mean(particles[u], weights=particles['w'], total_weight=total_weight) for u in momenta_axes}

    res = axes[np.argmax(np.abs(list(mean_u.values())))]

    if return_mean_momenta:
        return res, mean_u
    else:
        return res


def get_transverse_axes(axis: AxisStr) -> Tuple[AxisStr, AxisStr]:
    """Gives the axes transverse to the given axis

    Parameters
    ----------
    axis : AxisStr
        the axis

    Returns
    -------
    tuple
        axes transverse to the given axis

    """
    if axis == 'x':
        return 'y', 'z'
    elif axis == 'y':
        return 'x', 'z'
    elif axis == 'z':
        return 'x', 'y'
    else:
        raise ValueError(f'Unknown axis {axis}')


def generate_gaussian_bunch(
    particle_number,
    energy: float | pint.Quantity,
    *,
    sigma_x: float | pint.Quantity = 0,
    sigma_y: float | pint.Quantity = 0,
    charge: float | pint.Quantity | None = None,
):
    from .units import ensure_units, strip_units

    ureg = pint.get_application_registry()

    sigma_x = strip_units(sigma_x, 'm')
    sigma_y = strip_units(sigma_y, 'm')
    energy = ensure_units(energy, 'MeV')

    if charge is not None:
        e = ureg['elementary_charge']
        particle_weight = (charge / e / particle_number).m_as('')
    else:
        particle_weight = 1.0

    data = pd.DataFrame(
        {
            'x': np.random.normal(0, sigma_x, particle_number),
            'y': np.random.normal(0, sigma_y, particle_number),
            'z': np.zeros(particle_number),
            'ux': np.zeros(particle_number),
            'uy': np.zeros(particle_number),
            'uz': np.full(particle_number, np.sqrt(energy_to_gamma(energy) ** 2 - 1)),
            'w': np.full(particle_number, particle_weight),
        }
    )
    initialize_energy(data)
    return data
