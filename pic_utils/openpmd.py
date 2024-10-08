import numpy as _np
import pandas as _pd
import pint as _pint
from openpmd_viewer.addons import LpaDiagnostics as _Lpa

import typing

if typing.TYPE_CHECKING:
    from .plasma import PlasmaUnits


class OpenPMDWrapper:
    def __init__(self, folder, plasma_units: 'PlasmaUnits | None' = None) -> None:
        self.simulation = _Lpa(folder)
        self.ureg = _pint.get_application_registry()
        self.plasma_units = plasma_units

        self.c = self.ureg['speed_of_light']

    def read_field(
        self,
        iteration,
        field,
        component=None,
        *,
        geometry='xz',
        grid=False,
        mode='all',
        only_positive_r=False,
        plasma_units: 'PlasmaUnits | str | None' = 'auto',
    ):
        if geometry == 'xz':
            theta = 0.0
        elif geometry == 'yz':
            theta = _np.pi / 2
        elif geometry == '3d':
            if only_positive_r:
                raise ValueError(f'Value {only_positive_r=} is not allowed when {geometry=}')
            theta = None
        else:
            raise ValueError(f'Geometry {geometry} is not available, only xz, yz, 3d')

        if plasma_units == 'auto':
            plasma_units = self.plasma_units

        f, f_info = self.simulation.get_field(field, component, iteration=iteration, theta=theta, m=mode)

        if field == 'E':
            f = f * self.ureg['V/m']
        elif field == 'B':
            f = f * self.ureg['tesla']
        elif field == 'J':
            f = f * self.ureg['C/m^2/s']
        elif field.startswith('rho'):
            f = f * self.ureg['C/m^3']

        if plasma_units is not None:
            f = plasma_units.convert_to_unitless(f)

        if geometry == '3d':
            if grid:
                xx, yy, zz = _np.meshgrid(f_info.x, f_info.y, f_info.z, indexing='ij')
                zz = (zz * self.ureg.m).to('um')
                xx = (xx * self.ureg.m).to('um')
                yy = (yy * self.ureg.m).to('um')
                if plasma_units is not None:
                    xx = plasma_units.convert_to_unitless(xx)
                    yy = plasma_units.convert_to_unitless(yy)
                    zz = plasma_units.convert_to_unitless(zz)
                return zz, xx, yy, f
            else:
                return f
        else:
            r = f_info.r
            if only_positive_r:
                index = len(r) // 2
                f = f[index:, :]
                r = r[index:]
            if grid:
                zz, xx = _np.meshgrid(f_info.z, r)
                zz = (zz * self.ureg.m).to('um')
                xx = (xx * self.ureg.m).to('um')
                if plasma_units is not None:
                    xx = plasma_units.convert_to_unitless(xx)
                    zz = plasma_units.convert_to_unitless(zz)
                return zz, xx, f
            else:
                return f

    def read_poynting_vector(
        self, iteration, component, *, geometry='xz', grid=False, mode='all', only_positive_r=False
    ):
        from .electromagnetism import poynting_vector

        if component != 'z':
            raise NotImplementedError(f'Components other than z are not implemented yet, {component} not available')

        kwargs = {'geometry': geometry, 'mode': mode, 'only_positive_r': only_positive_r}

        ex = self.read_field(iteration, 'E', 'x', grid=grid, **kwargs)
        ey = self.read_field(iteration, 'E', 'y', grid=False, **kwargs)

        bx = self.read_field(iteration, 'B', 'x', grid=False, **kwargs)
        by = self.read_field(iteration, 'B', 'y', grid=False, **kwargs)

        if grid:
            xx, yy, ex = ex
            return xx, yy, poynting_vector(ex, ey, bx, by)
        else:
            return poynting_vector(ex, ey, bx, by)

    def read_particles(
        self,
        iteration,
        species,
        *,
        parameters=['x', 'y', 'z', 'ux', 'uy', 'uz', 'w'],
        select=None,
        initialize_energy=True,
        sort_by=None,
        max_particle_number=None,
    ):
        from .bunch import gamma, gamma_to_energy, limit_particle_number

        specie_data_arr = []

        if isinstance(species, str):
            species = [species]

        for i, s in enumerate(species):
            specie_data_list = self.simulation.get_particle(parameters, species=s, iteration=iteration, select=select)
            specie_data = _pd.DataFrame.from_dict(dict(zip(parameters, specie_data_list)))
            specie_data['species_id'] = i
            specie_data_arr.append(specie_data)

        if len(specie_data_arr) > 0:
            data = _pd.concat(specie_data_arr)
        else:
            dataframe_columns = parameters + ['species_id']
            data = _pd.DataFrame(columns=dataframe_columns)

        if initialize_energy:
            data['gamma'] = gamma(data['ux'], data['uy'], data['uz'])
            data['energy'] = gamma_to_energy(data['gamma']).m_as('MeV')

        if max_particle_number is not None:
            data = limit_particle_number(data, max_particle_number)

        if sort_by is not None:
            data.sort_values(sort_by, inplace=True)

        return data

    def iterations(self):
        return self.simulation.iterations

    def times(self, units='s'):
        return self.simulation.t * self.ureg[units]

    def species(self):
        species = self.simulation.avail_species
        return species if species is not None else []

    def density_species(self, prefix='rho_'):
        return [field[len(prefix) :] for field in self.simulation.avail_fields if field.startswith(prefix)]

    def get_laser_frequency(self, iteration, polarization='x', method='max'):
        """
        Calculates laser frequency for a specific iteration.
        Uses openpmd_reader LPA diagnostics.
        """
        return self.simulation.get_main_frequency(iteration=iteration, pol=polarization, method=method) / self.ureg.s

    def get_laser_wavelength(self, iteration, polarization='x', method='max'):
        """
        Calculates laser wavelength for a specific iteration.
        Uses openpmd_reader LPA diagnostics.
        """
        return (2 * _np.pi * self.c / self.get_laser_frequency(iteration, polarization, method)).to('um')
