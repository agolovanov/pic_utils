from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pint

if TYPE_CHECKING:
    import fbpic.main

    from . import materials


def __get_particles_per_cell(particles_per_cell: tuple | dict, species_name: str, species_symbol: str) -> tuple:
    if isinstance(particles_per_cell, dict):
        if species_name in particles_per_cell:
            ppc = particles_per_cell[species_name]
        elif species_symbol in particles_per_cell:
            ppc = particles_per_cell[species_symbol]
        else:
            raise ValueError(
                f'Particles per cell for species {species_name} ({species_symbol}) not found in the dictionary'
            )
    else:
        ppc = []
        for v in particles_per_cell:
            if isinstance(v, dict):
                if species_name in v:
                    ppc.append(v[species_name])
                elif species_symbol in v:
                    ppc.append(v[species_symbol])
                else:
                    raise ValueError(
                        f'Particles per cell for species {species_name} ({species_symbol}) not found in the dictionary'
                    )
            else:
                ppc.append(v)
        ppc = tuple(ppc)

    if len(ppc) != 3:
        raise ValueError('Particles per cell should be a tuple of 3 integers (ppc_z, ppc_r, ppc_theta), given {ppc}')

    return ppc


def add_plasma_profile(
    simulation: 'fbpic.main.Simulation',
    profile,
    composition: 'materials.Composition',
    particles_per_cell: 'tuple | dict',
    add_ions=True,
    ionizable_ions=True,
    different_ionized_species=True,
    prefix=None,
    **kwargs,
):
    """Creates a plasma profile for an FBPIC simulation with electrons and ionizable ions.

    Parameters
    ----------
    simulation : fbpic.main.Simulation
        The Simulation object where to initialize profile
    profile : function of (r, z) or (x, y, z)
        The density profile function
    composition : materials.Composition
        The material composition. The number density has to be pint.Quantity.
    particles_per_cell : tuple | dict
        Particles per cell in the z, r, and theta directions

        Can be a tuple of 3 integers (ppc_z, ppc_r, ppc_theta) to be used for all species,
        or a dictionary with species names as keys and tuples of 3 integers as values.
    add_ions : bool, optional
        Whether to initialize ions, by default True
    ionizable_ions : bool, optional
        Whether to make ions ionizable, by default True.
    different_ionized_species : bool, optional
        Whether to assign different ionization levels different electron species, by default True.
        When it is True, electrons will be named according to their source and ionization level,
        e.g. 'electrons_initial', 'electrons_nitrogen_5'. When it is False, all electrons will be named 'electrons'.
    prefix : str, optional
        A prefix to add to all species names, by default None
        E.g., if prefix='plasma', species will be named 'plasma_electrons', 'plasma_helium', etc.
    kwargs
        Will be passed to simulation.add_new_species

    Returns
    -------
    dict
        Dictionary of string - species pairs, where the strings contain the names of the species, e.g. 'helium',
        'electrons_helium_1'.
    """
    if not isinstance(composition.number_density, pint.Quantity):
        raise ValueError('Currently, only pint.Quantity type of density is expected contain dimensions')

    if prefix is None or prefix == '':
        prefix = ''
    else:
        prefix = f'{prefix}_'

    sim_kwargs = {'dens_func': profile}
    sim_kwargs.update(kwargs)

    ureg = pint.get_application_registry()

    e = ureg['elementary_charge'].m_as('C')
    m_e = ureg['electron_mass'].m_as('kg')
    m_u = ureg['dalton'].m_as('kg')

    n_e_initial = composition.get_number_density('e').m_as('1/m^3')

    species = {}

    electrons_element_name = 'electrons'
    electrons_element_symbol = 'e'
    electrons_name = f'{prefix}{electrons_element_name}'

    if n_e_initial > 0:
        ppc_z, ppc_r, ppc_theta = __get_particles_per_cell(
            particles_per_cell, electrons_element_name, electrons_element_symbol
        )
        electrons = simulation.add_new_species(
            q=-e, m=m_e, n=n_e_initial, p_nz=ppc_z, p_nr=ppc_r, p_nt=ppc_theta, **sim_kwargs
        )
        if add_ions and ionizable_ions and different_ionized_species:
            species[f'{electrons_name}_initial'] = electrons
        else:
            species[electrons_name] = electrons

    if add_ions:
        for el in composition.element_shares:
            initial_level = composition.ionization_levels[el]
            mass = el.relative_atomic_mass * m_u
            density = composition.get_number_density(el).m_as('1/m^3')
            if density <= 0:
                continue

            ppc_z, ppc_r, ppc_theta = __get_particles_per_cell(particles_per_cell, el.name, el.symbol)
            atoms = simulation.add_new_species(
                q=e * initial_level, m=mass, n=density, p_nz=ppc_z, p_nr=ppc_r, p_nt=ppc_theta, **sim_kwargs
            )
            species[prefix + el.name] = atoms

            if ionizable_ions:
                if different_ionized_species:
                    atom_electrons = {
                        i: simulation.add_new_species(q=-e, m=m_e) for i in range(initial_level, el.atomic_number)
                    }
                    for i, electrons in atom_electrons.items():
                        species[f'{prefix}electrons_{el.name}_{i + 1}'] = electrons

                    atoms.make_ionizable(el.symbol, target_species=atom_electrons, level_start=initial_level)
                else:
                    if electrons_name in species:
                        electrons = species[electrons_name]
                    else:
                        electrons = simulation.add_new_species(q=-e, m=m_e)
                        species[electrons_name] = electrons
                    atoms.make_ionizable(el.symbol, target_species=electrons, level_start=initial_level)

    return species


def setup_mpi(order: int = 32, *, print_details: bool = True) -> dict:
    import os

    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
    nprocs = MPI.COMM_WORLD.Get_size()

    if rank != 0:
        import sys

        f = open(os.devnull, 'w')
        sys.stdout = f

    order = order if nprocs > 1 else -1

    if print_details:
        print(f'Number of MPI processes: {nprocs}')
        print(f'Order {order}')

    return {'rank': rank, 'nprocs': nprocs, 'order': order}


def setup_output_folders(base_folder: str | Path, remove_subfolders: bool = False, use_boost: bool = True) -> dict:
    """Create output folders for FBPIC simulation results

    Parameters
    ----------
    base_folder : str | Path
        the main simulation results folder
    remove_subfolders : bool, optional
        whether to delete the existing subfolders with its content, by default False
    use_boost : bool, optional
        whether (will return the boost subfolder), by default True

    Returns
    -------
    dict
        dictionary with keys 'base', 'lab', and (optionally) 'boost' with the subfolder paths
    """
    import os
    import shutil

    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()

    base_folder = Path(base_folder)
    if rank == 0:
        os.makedirs(base_folder, exist_ok=True)

    folder_lab = base_folder / 'lab_diags'

    print(f'Folder: {base_folder}')
    print(f'Lab subfolder: {folder_lab}')
    folders = {'base': base_folder, 'lab': folder_lab}
    subfolders = [folder_lab]
    if use_boost:
        folder_boost = base_folder / 'boost_diags'
        folders['boost'] = folder_boost
        print(f'Boost subfolder: {folder_boost}')
        subfolders.append(folder_boost)

    if remove_subfolders and rank == 0:
        for folder in subfolders:
            if folder.exists():
                print(f'Folder {folder} exists, removing...')
                shutil.rmtree(folder)

    return folders


def setup_simulation_parameters(
    gamma_boost,
    *,
    dz: pint.Quantity,
    dr: pint.Quantity,
    v_window: pint.Quantity,
    zmin: pint.Quantity,
    zmax: pint.Quantity,
    interaction_length: pint.Quantity,
    output_interval: pint.Quantity,
    boost_write_period: int = 100,
):
    from fbpic.lpa_utils import boosted_frame

    ureg = pint.get_application_registry()
    c = ureg['speed_of_light']

    if output_interval.check({'[length]': 1}):  # given in length
        output_interval = output_interval / v_window

    if gamma_boost > 1.0:
        use_boost = True
        boost = boosted_frame.BoostConverter(gamma_boost)
        print(f'Lorentz boost gamma = {gamma_boost}')

        v_comoving = -np.sqrt(1.0 - 1.0 / boost.gamma0**2)
        print(f'Adding Galilean comoving frame with v/c={v_comoving:.3g}')
        v_comoving = (c * v_comoving).m_as('m/s')
    else:
        use_boost = False
        boost = None
        gamma_boost = None
        v_comoving = None
        print('Running non-boosted simulations')

    # The simulation timesteplab_simulation_steps / simulations_steps
    dt_lab = (dz / c).to('s')

    if use_boost:
        (dt_boost,) = boost.copropag_length([dt_lab])
        dt_max = (dr / c).to('s')
        if dt_max < dt_boost:
            print(f'The timestep has to be no bigger than {dt_max.to("fs"):.3g#~}, {dt_boost:.3g#~} would be too big')
            dt_boost = dt_max
        else:
            print(f'The timestep has to be no bigger than {dt_max.to("fs"):.3g#~}, actual {dt_boost:.3g#~}')
    else:
        print(f'Timestep: {dt_lab:.3g#~}')

    interaction_time = interaction_length / v_window

    diag_period = int((output_interval // dt_lab).m_as(''))  # Period of the diagnostics in number of timesteps
    n_timesteps = int((interaction_time / dt_lab).m_as(''))
    n_diag_timesteps = n_timesteps // diag_period + 1
    n_timesteps = (n_diag_timesteps - 1) * diag_period + 1
    interaction_time = n_timesteps * dt_lab
    interaction_length = (v_window * interaction_time).to('m')

    print(f'Simulation length: {interaction_length:.3g#~}; time: {interaction_time:.3g#~}')

    diag_dt_lab = (diag_period * dt_lab).to('s')
    diag_dt_boost = None

    if use_boost:
        interaction_time_boost = (
            boost.interaction_time(interaction_length.m_as('m'), (zmax - zmin).m_as('m'), v_window.m_as('m/s'))
            * ureg['s']
        )
        print(f'Interaction time in the boosted frame: {interaction_time_boost:.3g#~}')

        diag_dt_boost = (interaction_time_boost / (n_diag_timesteps - 1)).to('s')

    print(f'Output timestep = {diag_dt_lab:.3g#~} ({(diag_dt_lab * v_window).to("m"):.3g#~})')

    if use_boost:
        print(f'Output timestep in the boosted frame = {diag_dt_boost:.3g#~}')
    print(f'There should be {n_diag_timesteps} diagnostic iterations')
    if use_boost:
        print(f'Lab diagnostics will be saved to disc every {boost_write_period} iterations')

    v_window_magn = v_window.m_as('m/s')
    v_window_boost = v_window_magn
    if use_boost:
        (v_window_boost,) = boost.velocity([v_window_boost])

    lab_simulation_steps = int((interaction_time // dt_lab).m_as(''))
    if use_boost:
        simulations_steps = int((interaction_time_boost // dt_boost).m_as(''))
        print(
            f'Simulation will have {simulations_steps} timesteps '
            f'({lab_simulation_steps / simulations_steps:.3g}x less than without boost)'
        )
    else:
        simulations_steps = lab_simulation_steps
        print(f'Simulation will have {simulations_steps} timesteps')

    res = {
        'boost': use_boost,
        'gamma': gamma_boost,
        'dt': dt_lab.m_as('s'),
        'zmin': zmin.m_as('m'),
        'zmax': zmax.m_as('m'),
        'v_window': v_window_boost,
        'v_comoving': v_comoving,
        'lab_diag_dt': diag_dt_lab.m_as('s'),
        'diag_period': diag_period,
        'lab_diag_timesteps': n_diag_timesteps,
        'simulation_steps': simulations_steps,
    }

    if use_boost:
        res.update(
            {
                'boost_diag_dt': diag_dt_boost.m_as('s'),
                'v_window_lab': v_window_magn,
                'boost_diag_timesteps': n_diag_timesteps,
                'boost_write_period': boost_write_period,
            }
        )

    return res


def setup_diagnostics(
    simulation: 'fbpic.main.Simulation',
    simulation_parameters: dict,
    *,
    lab_dir=None,
    boost_dir=None,
    fields_lab: list = None,
    particle_species_lab: dict = None,
    density_species_lab: dict = None,
    particle_select_lab=None,
):
    from fbpic.openpmd_diag import (
        BackTransformedFieldDiagnostic,
        BackTransformedParticleDiagnostic,
        FieldDiagnostic,
        ParticleChargeDensityDiagnostic,
        ParticleDiagnostic,
    )

    diags = []

    zmin = simulation_parameters['zmin']
    zmax = simulation_parameters['zmax']
    use_boost = simulation_parameters['boost']
    lab_dt = simulation_parameters['lab_diag_dt']
    lab_timesteps = simulation_parameters['lab_diag_timesteps']
    gamma = simulation_parameters['gamma']

    comm = simulation.comm
    fld = simulation.fld

    if use_boost:
        v_lab = simulation_parameters['v_window_lab']
        period = simulation_parameters['boost_write_period']

        if fields_lab is not None:
            diags.append(
                BackTransformedFieldDiagnostic(
                    zmin_lab=zmin,
                    zmax_lab=zmax,
                    v_lab=v_lab,
                    dt_snapshots_lab=lab_dt,
                    Ntot_snapshots_lab=lab_timesteps,
                    period=period,
                    gamma_boost=gamma,
                    fldobject=fld,
                    comm=comm,
                    write_dir=lab_dir,
                    fieldtypes=fields_lab,
                )
            )
            print(f'Field diagnostics (lab frame) for: {", ".join(fields_lab)}')

        if density_species_lab is not None:
            print('Particle density diagnostic is not available for the lab frame in boosted frame simulations')

        if particle_species_lab is not None:
            diags.append(
                BackTransformedParticleDiagnostic(
                    zmin_lab=zmin,
                    zmax_lab=zmax,
                    v_lab=v_lab,
                    dt_snapshots_lab=lab_dt,
                    comm=comm,
                    Ntot_snapshots_lab=lab_timesteps,
                    period=period,
                    gamma_boost=gamma,
                    species=particle_species_lab,
                    write_dir=lab_dir,
                    fldobject=fld,
                    select=particle_select_lab,
                )
            )
            print(f'Particle diagnostics (lab frame) for: {", ".join(particle_species_lab.keys())}')
    else:
        diag_period = simulation_parameters['diag_period']

        if fields_lab is not None:
            diags.append(FieldDiagnostic(period=diag_period, fldobject=fld, comm=comm, write_dir=lab_dir))
            print(f'Field diagnostics for: {", ".join(fields_lab)}')

        if density_species_lab:
            diags.append(
                ParticleChargeDensityDiagnostic(
                    period=diag_period, sim=simulation, species=density_species_lab, write_dir=lab_dir
                )
            )
            print(f'Density diagnostics for: {", ".join(density_species_lab.keys())}')

        if particle_species_lab:
            diags.append(
                ParticleDiagnostic(
                    period=diag_period,
                    species=particle_species_lab,
                    comm=comm,
                    write_dir=lab_dir,
                    select=particle_select_lab,
                )
            )
            print(f'Particle diagnostics for: {", ".join(particle_species_lab.keys())}')

    simulation.diags = diags


def prepare_script_folder(script: str | Path, script_folder: str | Path, parameters: dict | None = None) -> str:
    """Prepares a separate folder for an FBPIC script.

    Copies the script to the folder and creates an input JSON file based on the dictionary of parameters inside this
    folder.

    Parameters
    ----------
    script : str | Path
        the path to the script or script template
    script_folder : str | Path
        the folder to be used to store the script
    parameters : dict | None, optional
        The list of input parameters to be passed to the script, by default None

        The parameters will be saved as a JSON file and passed to the script as the first command-line argument

    Returns
    -------
    str
        The final script to be run by python followed by command-line arguments passed to the script
    """
    from shutil import copy

    import pic_utils.json

    script = Path(script)
    script_folder = Path(script_folder).resolve()

    script_folder.mkdir(parents=True, exist_ok=True)
    new_script_path = script_folder / script.name
    copy(script, new_script_path)

    if parameters is not None:
        input_parameters_path = script_folder / 'input.json'
        pic_utils.json.save(parameters, input_parameters_path)
        return f'"{new_script_path.resolve()}" "{input_parameters_path.resolve()}"'
    else:
        return f'"{new_script_path.resolve()}"'


def print_particle_species(data: 'fbpic.main.Particles | fbpic.main.Simulation'):
    import fbpic.main

    if isinstance(data, fbpic.main.Simulation):
        for i, species in enumerate(data.ptcl):
            print(f'Species #{i}:')
            print_particle_species(species)
            print('-' * 10)
        return

    ureg = pint.get_application_registry()
    e = ureg['elementary_charge'].m_as('C')
    m_e = ureg['electron_mass'].m_as('kg')
    m_u = ureg['dalton'].m_as('kg')

    print(f'Charge: {data.q / e:.3g} e')
    if data.m >= 0.1 * m_u:
        print(f'Mass: {data.m / m_u:.3g} Da')
    else:
        print(f'Mass: {data.m / m_e:.3g} m_e')
    print(f'Particle shape: {data.particle_shape}')
    print(f'Number of particles: {data.Ntot}')
    if data.ionizer is not None:
        element = data.ionizer.element
        level_start = data.ionizer.level_start
        level_max = data.ionizer.level_max
        print(f'Ionizable species as element [{element}] from level {level_start} to {level_max}')

    if data.injector is not None:
        injector = data.injector
        Npr = injector.Npr
        Nptheta = injector.Nptheta
        dz_particles = injector.dz_particles
        rmin = injector.rmin
        rmax = injector.rmax
        n = injector.n
        print(
            f'Continuous injection with Npr={Npr}, Nptheta={Nptheta}, dz_particles={dz_particles:.3g}, '
            f'n={n:.3g}, rmin={rmin:.3g}, rmax={rmax:.3g}'
        )
