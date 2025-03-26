import numpy as np


def solve_trajectory_axisymmetric(
    r0: float,
    z_array: np.ndarray,
    *,
    ez=lambda z, r: 0.0,
    er=lambda z, r: 0.0,
    bphi=lambda z, r: 0.0,
    psi=lambda z, r: 0.0,
    a2avg=lambda z, r: 0.0,
    a2avg_rder=lambda z, r: 0.0,
    a2avg_zder=lambda z, r: 0.0,
) -> dict:
    """Calculate the trajectory of a particle in a quasistatic field in axisymmetric geometry.

    Plasma units are assumed.

    Parameters
    ----------
    r0 : float
        the initial radial position of the particle
    z_array : np.ndarray

    ez : callable, optional
        E_z(z, r), by default a function returning 0.0
    er : callable, optional
        E_r(z, r), by default a function returning 0.0
    bphi : callable, optional
        B_phi(z, r), by default a function returning 0.0
    psi : callable, optional
        The wakefield potential psi(z, r), by default a function returning 0.0
    a2avg : callable, optional
        The average of the square of the laser vector potential <a^2>(z, r), by default a function returning 0.0
    a2avg_rder : callable, optional
        The derivative of <a^2> with respect to r, by default a function returning 0.0
    a2avg_zder : callable, optional
        The derivative of <a^2> with respect to z, by default a function returning 0.0

    Returns
    -------
    dict
        A dictionary with the following keys:
        - z, r (coordinates)
        - pz, pr (momenta)
        - gamma (average Lorentz factor)
        - t (time)
        - psi (wakefield potential on the trajectory)
        - H = gamma - pz - psi (Hamiltonian on the trajectory); normally should be conserved
    """
    from scipy.integrate import solve_ivp, cumulative_trapezoid

    def gamma_func(z, r, pz, pr):
        return np.sqrt(1 + pz**2 + pr**2 + a2avg(z, r))

    def dt_dz(gamma, pz):
        return -gamma / (gamma - pz)

    def dpr_dz(z, r, pz, pr):
        gamma = gamma_func(z, r, pz, pr)
        return dt_dz(gamma, pz) * (-er(z, r) + pz / gamma * bphi(z, r) - 0.5 * a2avg_rder(z, r) / gamma)

    def dpz_dz(z, r, pz, pr):
        gamma = gamma_func(z, r, pz, pr)
        return dt_dz(gamma, pz) * (-ez(z, r) - pr / gamma * bphi(z, r) - 0.5 * a2avg_zder(z, r) / gamma)

    def dr_dz(z, r, pz, pr):
        gamma = gamma_func(z, r, pz, pr)
        return dt_dz(gamma, pz) * pr / gamma

    Y0 = [r0, 0, 0]

    def integrand(z, Y):
        return [dr_dz(z, *Y), dpz_dz(z, *Y), dpr_dz(z, *Y)]

    integrated_res = solve_ivp(
        integrand, (np.max(z_array), np.min(z_array)), Y0, dense_output=True, t_eval=z_array[::-1]
    )

    result = {}
    result['z'] = integrated_res.t
    result['r'] = integrated_res.y[0]
    result['pz'] = integrated_res.y[1]
    result['pr'] = integrated_res.y[2]
    point = (result['z'], result['r'], result['pz'], result['pr'])
    result['gamma'] = gamma_func(*point)
    result['t'] = cumulative_trapezoid(dt_dz(result['gamma'], result['pz']), result['z'], initial=0)
    result['psi'] = psi(result['z'], result['r'])
    if np.ndim(result['psi']) == 0:
        result['psi'] = np.full_like(result['z'], result['psi'])
    result['H'] = result['gamma'] - result['pz'] - result['psi']

    return result
