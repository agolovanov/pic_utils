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
    def __init__(self, density, units: dict = None) -> None:
        """Creates a system of plasma units.

        Parameters
        ----------
        density : pint.Quantity
            Base density for plasma units (can possibly be an array).
        default_units : dict
            Dictionary which changes the default units to be used in the system.
            By default, uses pic_utils.plasma.default_units (which are SI units); possible keys can be looked up there.
        """
        import numpy as np
        import pint
        
        if not isinstance(density, pint.Quantity):
            raise ValueError("Density should be of pint.Quantity type and contain dimensions")
        ureg = density._REGISTRY

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

        self.charge_density = self.density * self.e

        self.E = (self.m_e * self.c ** 2 * self.wavenumber / self.e_cgs).to(du['E'], 'Gau')
        self.B = self.E.to(du['B'], 'Gau')

    def __str__(self) -> str:
        return f'PlasmaUnits({self.density:g~})'

    def __repr__(self) -> str:
        return self.__str__()
