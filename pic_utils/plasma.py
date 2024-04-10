import pint

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
    def __init__(self, base_unit: pint.Quantity, units: dict = None) -> None:
        """Creates a system of plasma units.

        Parameters
        ----------
        base_unit : pint.Quantity
            Base unit for plasma units; can be density, wavelength, or frequency.
            Arrays are allowed.
        default_units : dict
            Dictionary which changes the default units to be used in the system.
            By default, uses pic_utils.plasma.default_units (which are SI units); possible keys can be looked up there.
        """
        import numpy as np

        if not isinstance(base_unit, pint.Quantity):
            raise ValueError("Base unit should be of pint.Quantity type and contain units")
        ureg = base_unit._REGISTRY

        self.default_units = default_units.copy()
        if units is not None:
            self.default_units.update(units)
        du = self.default_units  # alias

        self.e = (1 * ureg.elementary_charge).to(du['charge'])
        self.e_cgs = self.e.to(du['charge_cgs'], 'Gau')
        self.m_e = (1 * ureg.electron_mass).to(du['mass'])
        self.c = (1 * ureg.speed_of_light).to(ureg(du['length']) / ureg(du['time']))

        if base_unit.check({'[length]': -3}):  # density
            self.density = base_unit.to(du['density'])
            self.frequency = np.sqrt(4 * np.pi * self.e_cgs ** 2 * self.density / self.m_e).to(1 / ureg(du['time']))
            self.wavenumber = (self.frequency / self.c).to(1 / ureg(du['length']))
            self.wavelength = (2 * np.pi / self.wavenumber).to(du['length'])
        elif base_unit.check({'[length]': 1}):  # wavelength
            self.wavelength = base_unit.to(du['length'])
            self.wavenumber = (2 * np.pi / self.wavelength).to(1 / ureg(du['length']))
            self.frequency = (self.c * self.wavenumber).to(1 / ureg(du['time']))
            self.density = (self.m_e * self.frequency ** 2 / (4 * np.pi * self.e_cgs ** 2)).to(du['density'])
        elif base_unit.check({'[time]': -1}):  # frequency
            self.frequency = base_unit.to(1 / ureg(du['time']))
            self.wavenumber = (self.frequency / self.c).to(1 / ureg(du['length']))
            self.wavelength = (2 * np.pi / self.wavenumber).to(du['length'])
            self.density = (self.m_e * self.frequency ** 2 / (4 * np.pi * self.e_cgs ** 2)).to(du['density'])
        else:
            raise ValueError(f'Value {base_unit:.3g~} has wrong units {base_unit.units}; '
                             'only densities, wavelengths and frequencies are allowed')

        self.charge_density = self.density * self.e

        self.E = (self.m_e * self.c ** 2 * self.wavenumber / self.e_cgs).to(du['E'], 'Gau')
        self.B = self.E.to(du['B'], 'Gau')

    def __str__(self) -> str:
        return f'PlasmaUnits({self.density:g~})'

    def __repr__(self) -> str:
        return self.__str__()
