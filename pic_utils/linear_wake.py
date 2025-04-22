import typing

import numpy as np
import pint

if typing.TYPE_CHECKING:
    from . import PlasmaUnits


ureg = pint.get_application_registry()


def field_to_ponderomotive_potential(
    field: np.ndarray | pint.Quantity,
    window_size: int = 20,
    unitless: bool | None = None,
    laser_units: 'PlasmaUnits' = None,
    longitudinal_axis: int = 1,
) -> np.ndarray | pint.Quantity:
    """Converts the transverse electric field distribution to the ponderomotive potential.

    Parameters
    ----------
    field : np.ndarray | pint.Quantity
        The transverse electric field distribution.
    window_size : int, optional
        The window size for averaging over laser oscillations, by default 20
    unitless : bool | None, optional
        If True or False, the result is returned as a unitless array or a Quantity, respectively. If None, it is determined by the input, by default None
    laser_units : PlasmaUnits, optional
        Laser units units for normalization (if needed), by default None
    longitudinal_axis : int, optional
        The axis along which the averaging is performed, by default 1

    Returns
    -------
    np.ndarray | pint.Quantity
        the resulting potential distribution

    """
    from .functions import running_mean

    if isinstance(field, pint.Quantity):
        if unitless is None:
            unitless = False

        if laser_units is None:
            raise ValueError('The laser_units argument must be provided if the field is a Quantity.')

        field = laser_units.convert_to_unitless(field)
    else:
        if unitless is None:
            unitless = True

    potential = 0.5 * running_mean(field**2, axis=longitudinal_axis, window_size=window_size)

    if unitless:
        return potential
    else:
        return laser_units.convert_to_units(potential, laser_units.energy)


def calculate_laser_linear_wake(
    ponderomotive_potential: np.ndarray | pint.Quantity,
    z: np.ndarray,
    r: np.ndarray,
    *,
    currents: bool = False,
    energy: bool = False,
    plasma_density: 'PlasmaUnits | pint.Quantity | None' = None,
    unitless: bool = False,
) -> dict:
    """Calculate the linear wakefield from the ponderomotive potential.

    At the moment, assumes r,z coordinates with z being the wakefield axis.

    Parameters
    ----------
    ponderomotive_potential : np.ndarray | pint.Quantity
        the ponderomotive potential either in plasma units or as a Quantity
    z : np.ndarray
        the longitudinal coordinate (along which the wake is calculated)
    r : np.ndarray
        the transverse coordinate
    currents : bool, optional
        if True, the wakefield currents are calculated, by default False
    energy : bool, optional
        if True, the wakefield energy is calculated, by default False
    plasma_density : PlasmaUnits | pint.Quantity
        a value representing the plasma density. Can be an instance of PlasmaUnits or a Quantity fully determining the density (density, frequency, wavelength), by default None
    unitless : bool, optional
        if True, the results are returned in plasma units, by default False

    Returns
    -------
    dict
        a dictionary with the following keys:
        - 'psi' (wakefield potential)
        - 'ez' (longitudinal electric field)
        - 'er' (transverse electric field)
        - 'drho' (electron charge density variation)
        - 'jz' and 'jr' (if currents is True, the longitudinal and transverse current densities)
        - 'w_tilde_em', 'w_tilde_e', 'w_tilde' (if energy is True, the electromagnetic and electric energy densities)
        - 'Psi_em', 'Psi_e', 'Psi' (if energy is True, the electromagnetic and electric energy densities integrated over the transverse coordinate)
    """
    from scipy.integrate import cumulative_trapezoid

    if isinstance(plasma_density, pint.Quantity):
        plasma_density = PlasmaUnits(plasma_density)

    pu = plasma_density

    if isinstance(ponderomotive_potential, pint.Quantity):
        ponderomotive_potential = pu.convert_to_unitless(ponderomotive_potential)

    if isinstance(z, pint.Quantity):
        z = pu.convert_to_unitless(z)

    if isinstance(r, pint.Quantity):
        r = pu.convert_to_unitless(r)

    xi = np.max(z) - z
    dz = z[1] - z[0]
    dr = r[1] - r[0]

    integrand = ponderomotive_potential * np.exp(-1j * xi)
    base_integral = np.exp(1j * xi) * cumulative_trapezoid(integrand[:, ::-1], dx=dz, initial=0)[:, ::-1]

    base_integral_dtrans = np.gradient(base_integral, dr, axis=0)
    base_integral_dtrans_2 = np.gradient(r[:, None] * base_integral_dtrans, dr, axis=0) / r[:, None]

    result = {}

    result['psi'] = np.imag(base_integral)
    result['ez'] = np.real(base_integral)
    result['er'] = -np.imag(base_integral_dtrans)
    result['drho'] = -ponderomotive_potential + result['psi'] - np.imag(base_integral_dtrans_2)

    if currents or energy:
        jz = -ponderomotive_potential + result['psi']
        jr = np.real(base_integral_dtrans)
        if currents:
            result['jz'] = jz
            result['jr'] = jr

    if energy:
        result['w_tilde_em'] = 0.5 * (result['er'] ** 2 + result['ez'] ** 2)
        result['w_tilde_e'] = ponderomotive_potential + 0.5 * (jz**2 + jr**2)
        result['w_tilde'] = result['w_tilde_em'] + result['w_tilde_e']
        result['Psi_em'] = np.pi * np.trapezoid(np.abs(r)[:, None] * result['w_tilde_em'], dx=dr, axis=0)
        result['Psi_e'] = np.pi * np.trapezoid(np.abs(r)[:, None] * result['w_tilde_e'], dx=dr, axis=0)
        result['Psi'] = np.pi * np.trapezoid(np.abs(r)[:, None] * result['w_tilde'], dx=dr, axis=0)

    if not unitless:
        result['psi'] = pu.convert_to_units(result['psi'], 'phi')
        result['ez'] = pu.convert_to_units(result['ez'], 'E')
        result['er'] = pu.convert_to_units(result['er'], 'E')
        result['drho'] = pu.convert_to_units(result['drho'], 'charge_density')

    return result
