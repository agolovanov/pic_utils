import numpy as _np
from openpmd_viewer.addons import LpaDiagnostics as _Lpa
import pandas as _pd


class OpenPMDWrapper:
    def __init__(self, folder, pint_unit_registry) -> None:
        self.simulation = _Lpa(folder)
        self.ureg = pint_unit_registry

        self.c = self.ureg['speed_of_light']

    def read_field(self, iteration, field, component=None, geometry='xz', grid=False, mode='all',
                   only_positive_r=False):
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

    def read_particles(self, iteration, species, parameters=['x', 'y', 'z', 'ux', 'uy', 'uz', 'w'], select=None):
        specie_data_arr = []

        if isinstance(species, str):
            species = [species]

        for i, s in enumerate(species):
            specie_data_list = self.simulation.get_particle(parameters, species=s, iteration=iteration, select=select)
            specie_data = _pd.DataFrame.from_dict(dict(zip(parameters, specie_data_list)))
            specie_data['species_id'] = i
            specie_data_arr.append(specie_data)

        return _pd.concat(specie_data_arr)

    def iterations(self):
        return self.simulation.iterations

    def times(self, units="s"):
        return self.simulation.t * self.ureg[units]

    def species(self):
        return self.simulation.avail_species

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
