import pint as _pint

default_units = {
    'length': 'm',
    'time': 's',
    'density': '1/m^3',
    'E': 'V/m',
    'B': 'T',
    'charge': 'C',
    'charge_cgs': 'esu',
    'mass': 'kg'
}


class PlasmaUnits:
    def __init__(self, density, unit_registry: _pint.UnitRegistry, units: dict = None) -> None:
        """Creates a system of plasma units.

        Parameters
        ----------
        density : pint.Quantity
            Base density for plasma units (can possibly be an array).
        unit_registry : pint.UnitRegistry
            Unit registry from `pint` which provides units.
        default_units : dict
            Dictionary which changes the default units to be used in the system.
            By default, uses pic_utils.plasma.default_units (which are SI units); possible keys can be looked up there.
        """
        import numpy as np
        ureg = unit_registry  # alias

        self.default_units = default_units.copy()
        if units is not None:
            self.default_units.update(units)
        du = self.default_units  # alias

        self.e = (1 * ureg.elementary_charge).to(du['charge'])
        self.e_cgs = self.e.to(du['charge_cgs'], 'Gau')
        self.m_e = (1 * ureg.electron_mass).to(du['mass'])
        self.c = (1 * ureg.speed_of_light).to(ureg(du['length']) / ureg(du['time']))

        self.density = density.to(du['density'])
        self.frequency = np.sqrt(4 * np.pi * self.e_cgs ** 2 * self.density / self.m_e).to(1 / ureg(du['time']))
        self.wavenumber = (self.frequency / self.c).to(1 / ureg(du['length']))
        self.wavelength = (2 * np.pi / self.wavenumber).to(du['length'])

    def __str__(self) -> str:
        return f'PlasmaUnits({self.density:g~})'

    def __repr__(self) -> str:
        return self.__str__()
