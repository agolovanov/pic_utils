from __future__ import annotations

import typing

from .units import conj, real

if typing.TYPE_CHECKING:
    import pint


def poynting_vector(
    e1: pint.Quantity,
    e2: pint.Quantity,
    b1: pint.Quantity,
    b2: pint.Quantity,
) -> pint.Quantity:
    """
    Calculate one Cartesian component of the electromagnetic Poynting vector.

    The component is evaluated as ``(E1 * B2 - B1 * E2) / mu0``. For example,
    passing ``E1=Ex``, ``E2=Ey``, ``B1=Bx`` and ``B2=By`` returns ``S_z``.

    Parameters
    ----------
    e1, e2 : pint.Quantity
        Electric-field components with units compatible with ``V/m``.
    b1, b2 : pint.Quantity
        Magnetic-field components with units compatible with tesla.

    Returns
    -------
    pint.Quantity
        Poynting-vector component in ``W/m^2``.
    """
    ureg = e1._REGISTRY
    mu0 = ureg('vacuum_permeability')
    return ((e1 * b2 - b1 * e2) / mu0).to('W/m^2')


def energy_density(
    ex: pint.Quantity,
    ey: pint.Quantity,
    ez: pint.Quantity,
    bx: pint.Quantity,
    by: pint.Quantity,
    bz: pint.Quantity,
) -> pint.Quantity:
    """
    Calculate the electromagnetic energy density from Cartesian field components.

    Parameters
    ----------
    ex, ey, ez : pint.Quantity
        Electric-field components with units compatible with ``V/m``.
    bx, by, bz : pint.Quantity
        Magnetic-field components with units compatible with tesla.

    Returns
    -------
    pint.Quantity
        Electromagnetic energy density in ``J/m^3``.
    """
    ureg = ex._REGISTRY
    mu0 = ureg('vacuum_permeability')
    eps0 = ureg('vacuum_permittivity')
    return (0.5 * (eps0 * (ex * ex + ey * ey + ez * ez) + (bx * bx + by * by + bz * bz) / mu0)).to('J/m^3')


def mode_poynting_vector(
    mode: int,
    er: pint.Quantity,
    et: pint.Quantity,
    br: pint.Quantity,
    bt: pint.Quantity,
) -> pint.Quantity:
    """
    Calculate the longitudinal Poynting-vector contribution of a cylindrical mode.

    For nonzero modes, the fields are interpreted as complex azimuthal-mode
    amplitudes and the result includes the factor of one half from azimuthal
    averaging. Mode zero is evaluated directly from its real field components.

    Parameters
    ----------
    mode : int
        Azimuthal mode number.
    er, et : pint.Quantity
        Radial and azimuthal electric-field mode amplitudes with units
        compatible with ``V/m``.
    br, bt : pint.Quantity
        Radial and azimuthal magnetic-field mode amplitudes with units
        compatible with tesla.

    Returns
    -------
    pint.Quantity
        Azimuthally averaged longitudinal Poynting-vector contribution in
        ``W/m^2``.
    """
    ureg = er._REGISTRY
    mu0 = ureg('vacuum_permeability')
    if mode == 0:
        power_density = real(er * bt - et * br) / mu0
    else:
        power_density = 0.5 * real(er * conj(bt) - et * conj(br)) / mu0
    return power_density.to('W/m^2')


def mode_energy_density(
    mode: int,
    er: pint.Quantity,
    et: pint.Quantity,
    ez: pint.Quantity,
    br: pint.Quantity,
    bt: pint.Quantity,
    bz: pint.Quantity,
) -> pint.Quantity:
    """
    Calculate the electromagnetic energy-density contribution of a cylindrical mode.

    For nonzero modes, the fields are interpreted as complex azimuthal-mode
    amplitudes and the squared magnitudes include the factor of one half from
    azimuthal averaging. Mode zero is evaluated directly from its real field
    components.

    Parameters
    ----------
    mode : int
        Azimuthal mode number.
    er, et, ez : pint.Quantity
        Radial, azimuthal and longitudinal electric-field mode amplitudes with
        units compatible with ``V/m``.
    br, bt, bz : pint.Quantity
        Radial, azimuthal and longitudinal magnetic-field mode amplitudes with
        units compatible with tesla.

    Returns
    -------
    pint.Quantity
        Azimuthally averaged electromagnetic energy-density contribution in
        ``J/m^3``.
    """
    ureg = er._REGISTRY
    mu0 = ureg('vacuum_permeability')
    eps0 = ureg('vacuum_permittivity')
    if mode == 0:
        e2 = real(er * er + et * et + ez * ez)
        b2 = real(br * br + bt * bt + bz * bz)
    else:
        e2 = 0.5 * real(er * conj(er) + et * conj(et) + ez * conj(ez))
        b2 = 0.5 * real(br * conj(br) + bt * conj(bt) + bz * conj(bz))
    return (0.5 * (eps0 * e2 + b2 / mu0)).to('J/m^3')
