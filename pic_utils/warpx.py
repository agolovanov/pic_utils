import pandas as _pd
import numpy as _np


class FieldProbe2D:
    import pint
    """
    Class representing a 2D field probe.
    """
    def __init__(self, path, ureg: pint.UnitRegistry) -> None:
        """Creates a 2D field probe

        Parameters
        ----------
        path :
            path to the file
        ureg : pint.UnitRegistry
            unit registry for the units
        """
        self.ureg = ureg
        self.df = _pd.read_csv(path, sep=' ')

        column_names = {
            '[0]step()': 'step',
            '[1]time(s)': 'time',
            '[2]part_x_lev0-(m)': 'x',
            '[3]part_y_lev0-(m)': 'y',
            '[4]part_z_lev0-(m)': 'z',
            '[5]part_Ex_lev0-(V/m)': 'Ex',
            '[6]part_Ey_lev0-(V/m)': 'Ey',
            '[7]part_Ez_lev0-(V/m)': 'Ez',
            '[8]part_Bx_lev0-(T)': 'Bx',
            '[9]part_By_lev0-(T)': 'By',
            '[10]part_Bz_lev0-(T)': 'Bz',
            '[11]part_S_lev0-(W/m^2)': 'S',
        }

        self.df.rename(columns=column_names, inplace=True)

        self.steps = self.df['step'].unique()
        self.timesteps = self.df['time'].unique() * ureg.s

        df0 = self.df[self.df['step'] == self.steps[0]]
        r = df0[['x', 'y', 'z']].to_numpy() * ureg.m

        self.v1 = r[1] - r[0]
        self.d1 = _np.sqrt(_np.sum(self.v1 ** 2))

        v_diag = r[-1] - r[0]
        self.r_center = r[0] + 0.5 * v_diag
        shape_v1 = int(v_diag @ self.v1 / self.d1 ** 2) + 1
        r = r.reshape(shape_v1, -1, 3)

        self.r = r

        self.shape = r.shape[:2]
        self.v2 = r[1, 0] - r[0, 0]
        self.d2 = _np.sqrt(_np.sum(self.v2 ** 2))

        self.vn = _np.cross(self.v1, self.v2)

    def grid_spacing(self):
        """Grid spacing of the probe

        Returns
        -------
        tuple
            2 numbers of d1 and d2
        """
        return (self.d1, self.d2)

    def __str__(self) -> str:
        n1, n2 = self.shape

        return (f'2D field probe {n1}×{n2} at {self.r_center:g~}')

    def __repr__(self) -> str:
        return str(self)
    
    def field_unit(self, field):
        if field in ('Ex', 'Ey', 'Ez'):
            unit = self.ureg['V/m']
        elif field in ('Bx', 'By', 'Bz'):
            unit = self.ureg['T']
        elif field == 'S':
            unit = self.ureg['W/m^2']
        else:
            raise ValueError(f'Unknown field {field}')
        return unit

    def get_field(self, step, field, aggregate=False):
        """Gets the field at a certain step

        Parameters
        ----------
        step : int
            The step iteration
        field : str
            field string (possible values are Ex, Ey, Ez, Bx, By, Bz, S)
        aggregate : bool, optional
            whether to calculate the integral of the field over the probe, by default False

        Returns
        -------
        Array or number
            2D array or the aggregated value of the field
        """
        if step not in self.steps:
            raise ValueError(f'Step size {step} not available')

        df0 = self.df[self.df['step'] == step]
        field = df0[field].to_numpy().reshape(self.shape) * self.field_unit(field)

        if aggregate:
            return _np.sum(field) * self.d1 * self.d2
        else:
            return field

    def get_field_aggregate_series(self, field):
        """Calculates the integrated over all steps field series

        Parameters
        ----------
        field : str
            field string (possible values are Ex, Ey, Ez, Bx, By, Bz, S)
        """
        return self.df.groupby('step')[field].sum().to_numpy() * self.d1 * self.d2 * self.field_unit(field)
