def poynting_vector(e1, e2, b1, b2):
    ureg = e1._REGISTRY
    mu0 = ureg['vacuum_permeability']
    return ((e1 * b2 - b1 * e2) / mu0).to('W/m^2')


def energy_density(ex, ey, ez, bx, by, bz):
    ureg = ex._REGISTRY
    mu0 = ureg['vacuum_permeability']
    eps0 = ureg['vacuum_permittivity']
    return (0.5 * (eps0 * (ex * ex + ey * ey + ez * ez) + (bx * bx + by * by + bz * bz) / mu0)).to('J/m^3')