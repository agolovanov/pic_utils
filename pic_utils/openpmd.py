import numpy as _np
import pandas as _pd
import pint as _pint
from openpmd_viewer.addons import LpaDiagnostics as _Lpa


class OpenPMDWrapper:
    def __init__(self, folder) -> None:
        self.simulation = _Lpa(folder)
        self.ureg = _pint.get_application_registry()

        self.c = self.ureg['speed_of_light']

    def read_field(
        self, iteration, field, component=None, *, geometry='xz', grid=False, mode='all', only_positive_r=False
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

        f, f_info = self.simulation.get_field(field, component, iteration=iteration, theta=theta, m=mode)

        if field == 'E':
            f = f * self.ureg['V/m']
        elif field == 'B':
            f = f * self.ureg['tesla']
        elif field.startswith('rho'):
            f = f * self.ureg['C/m^3']

        if geometry == '3d':
            if grid:
                xx, yy, zz = _np.meshgrid(f_info.x, f_info.y, f_info.z, indexing='ij')
                zz = (zz * self.ureg.m).to('um')
                xx = (xx * self.ureg.m).to('um')
                yy = (yy * self.ureg.m).to('um')
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

    def read_particles(self, iteration, species, parameters=['x', 'y', 'z', 'ux', 'uy', 'uz', 'w'], select=None):
        specie_data_arr = []

        if isinstance(species, str):
            species = [species]

        for i, s in enumerate(species):
            specie_data_list = self.simulation.get_particle(parameters, species=s, iteration=iteration, select=select)
            specie_data = _pd.DataFrame.from_dict(dict(zip(parameters, specie_data_list)))
            specie_data['species_id'] = i
            specie_data_arr.append(specie_data)

        if len(specie_data_arr) > 0:
            return _pd.concat(specie_data_arr)
        else:
            dataframe_columns = parameters + ['species_id']
            return _pd.DataFrame(columns=dataframe_columns)

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
