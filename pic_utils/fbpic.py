import fbpic.main
from . import materials


def add_plasma_profile(simulation: fbpic.main.Simulation, profile, composition: materials.Composition,
                       particles_per_cell, add_ions=True, ionizable_ions=True, different_ionized_species=True,
                       **kwargs):
    """Creates a plasma profile for an FBPIC simulation with electrons and ionizable ions.

    Parameters
    ----------
    simulation : fbpic.main.Simulation
        The Simulation object where to initialize profile
    profile : function of (r, z) or (x, y, z)
        The density profile function
    composition : materials.Composition
        The material composition. The number density has to be pint.Quantity.
    particles_per_cell : tuple, optional
        Particles per cell in the z, r, and theta directions
    add_ions : bool, optional
        Whether to initialize ions, by default True
    ionizable_ions : bool, optional
        Whether to make ions ionizable, by default True.
    different_ionized_species : bool, optional
        Whether to assign different ionization levels different electron species, by default True.
        When it is True, electrons will be named according to their source and ionization level,
        e.g. 'electrons_initial', 'electrons_nitrogen_5'. When it is False, all electrons will be named 'electrons'.
    kwargs
        Will be passed to simulation.add_particle_species

    Returns
    -------
    dict
        Dictionary of string - species pairs, where the strings contain the names of the species, e.g. 'helium',
        'electrons_helium_1'.
    """
    import pint

    if not isinstance(composition.number_density, pint.Quantity):
        raise ValueError("Currently, only pint.Quantity type of density is expected contain dimensions")

    ppc_nz, ppc_nr, ppc_nt = particles_per_cell
    sim_kwargs = {'dens_func': profile, 'p_nz': ppc_nz, 'p_nr': ppc_nr, 'p_nt': ppc_nt}
    sim_kwargs.update(kwargs)

    ureg = composition.number_density._REGISTRY

    e = ureg['elementary_charge'].m_as('C')
    m_e = ureg['electron_mass'].m_as('kg')
    m_u = ureg['dalton'].m_as('kg')

    n_e_initial = composition.get_number_density('e').m_as('1/m^3')

    species = {}

    if n_e_initial > 0:
        electrons = simulation.add_new_species(q=-e, m=m_e, n=n_e_initial, **sim_kwargs)
        if add_ions and ionizable_ions and different_ionized_species:
            species['electrons_initial'] = electrons
        else:
            species['electrons'] = electrons

    if add_ions:
        for el in composition.element_shares:
            initial_level = composition.ionization_levels[el]
            mass = el.relative_atomic_mass * m_u
            density = composition.get_number_density(el).m_as('1/m^3')
            if density <= 0:
                continue

            atoms = simulation.add_new_species(q=e * initial_level, m=mass, n=density, **sim_kwargs)
            species[el.name] = atoms

            if ionizable_ions:
                if different_ionized_species:
                    atom_electrons = {
                        i: simulation.add_new_species(q=-e, m=m_e) for i in range(initial_level, el.atomic_number)
                    }
                    for i, electrons in atom_electrons.items():
                        species[f'electrons_{el.name}_{i+1}'] = electrons

                    atoms.make_ionizable(el.symbol, target_species=atom_electrons, level_start=initial_level)
                else:
                    if 'electrons' in species:
                        electrons = species['electrons']
                    else:
                        electrons = simulation.add_new_species(q=-e, m=m_e)
                        species['electrons'] = electrons
                    atoms.make_ionizable(el.symbol, target_species=electrons, level_start=initial_level)

    return species
