import pint
import typing

if typing.TYPE_CHECKING:
    import numpy as np

default_units = {
    'length': 'm',
    'time': 's',
    'density': '1/m^3',
    'E': 'V/m',
    'B': 'T',
    'charge': 'C',
    'charge_cgs': 'esu',
    'mass': 'kg',
    'energy': 'J',
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
            raise ValueError('Base unit should be of pint.Quantity type and contain units')
        ureg = base_unit._REGISTRY

        self.default_units = default_units.copy()
        if units is not None:
            self.default_units.update(units)
        du = self.default_units  # alias

        self.e = (1 * ureg.elementary_charge).to(du['charge'])
        self.e_cgs = self.e.to(du['charge_cgs'], 'Gau')
        self.m_e = (1 * ureg.electron_mass).to(du['mass'])
        self.c = (1 * ureg.speed_of_light).to(ureg(du['length']) / ureg(du['time']))
        self.energy = (self.m_e * self.c**2).to(du['energy'])

        if base_unit.check({'[length]': -3}):  # density
            self.density = base_unit.to(du['density'])
            self.frequency = np.sqrt(4 * np.pi * self.e_cgs**2 * self.density / self.m_e).to(1 / ureg(du['time']))
            self.wavenumber = (self.frequency / self.c).to(1 / ureg(du['length']))
            self.wavelength = (2 * np.pi / self.wavenumber).to(du['length'])
        elif base_unit.check({'[length]': 1}):  # wavelength
            self.wavelength = base_unit.to(du['length'])
            self.wavenumber = (2 * np.pi / self.wavelength).to(1 / ureg(du['length']))
            self.frequency = (self.c * self.wavenumber).to(1 / ureg(du['time']))
            self.density = (self.m_e * self.frequency**2 / (4 * np.pi * self.e_cgs**2)).to(du['density'])
        elif base_unit.check({'[time]': -1}):  # frequency
            self.frequency = base_unit.to(1 / ureg(du['time']))
            self.wavenumber = (self.frequency / self.c).to(1 / ureg(du['length']))
            self.wavelength = (2 * np.pi / self.wavenumber).to(du['length'])
            self.density = (self.m_e * self.frequency**2 / (4 * np.pi * self.e_cgs**2)).to(du['density'])
        else:
            raise ValueError(
                f'Value {base_unit:.3g~} has wrong units {base_unit.units}; '
                'only densities, wavelengths and frequencies are allowed'
            )

        self.charge_density = self.density * self.e

        self.E = (self.m_e * self.c**2 * self.wavenumber / self.e_cgs).to(du['E'], 'Gau')
        self.B = self.E.to(du['B'], 'Gau')

    def __str__(self) -> str:
        return f'PlasmaUnits({self.density:g~})'

    def __repr__(self) -> str:
        return self.__str__()

    def convert_to_unitless(self, value: pint.Quantity):
        """Convert the given value to a unitless value in the corresponding plasma units.

        Parameters
        ----------
        value : pint.Quantity
            the value to be converted

        Returns
        -------
        Any
            array

        ------
        ValueError
            in the case when the conversion is unsuccesful
        """
        if value.check('[length]'):
            return (self.wavenumber * value).m_as('')
        elif value.check('[time]'):
            return (self.frequency * value).m_as('')
        elif value.check('[charge]'):
            return (value / self.e).m_as('')
        elif value.check('1 / [volume]'):
            return (value / self.density).m_as('')
        elif value.check('[charge] / [volume]'):  # charge density
            return (value / self.density / self.e).m_as('')
        elif value.check('[velocity] * [charge] / [volume]'):  # charge current density
            return (value / self.density / self.e / self.c).m_as('')
        elif value.check('[electric_field]'):
            return (value / self.E).m_as('')
        elif value.check('[magnetic_field]'):
            return (value / self.B).m_as('')
        elif value.check('[energy]'):
            return (value / self.energy).m_as('')
        else:
            raise ValueError(f'The value has a unit [{value.units}] which cannot be converted')

    def convert_to_units(self, value: 'float | np.ndarray', unit: str | pint.Unit):
        """Converts the given dimensionless value to a value with the added unit.

        Parameters
        ----------
        value : np.array
            the dimensionless value to be converted
        unit : str | pint.Unit
            the unit to be added.
            Can be a key of the default_units dictionary (like 'length'), a string representation of a unit, or a pint.Unit object.

        Returns
        -------
        pint.Quantity
            the value with the added unit
        """
        if unit in self.default_units:
            unit = self.default_units[unit]

        if isinstance(unit, str):
            unit = pint.Unit(unit)

        unit_value = 1 * unit  # workaround for lack of check method in pint.Unit

        if unit_value.check('[length]'):
            return (value / self.wavenumber).to(unit)
        elif unit_value.check('[time]'):
            return (value / self.frequency).to(unit)
        elif unit_value.check('[charge]'):
            return (value * self.e).to(unit)
        elif unit_value.check('1 / [volume]'):
            return (value * self.density).to(unit)
        elif unit_value.check('[charge] / [volume]'):  # charge density
            return (value * self.density * self.e).to(unit)
        elif unit_value.check('[velocity] * [charge] / [volume]'):  # charge current density
            return (value * self.density * self.e * self.c).to(unit)
        elif unit_value.check('[electric_field]'):
            return (value * self.E).to(unit)
        elif unit_value.check('[magnetic_field]'):
            return (value * self.B).to(unit)
        elif unit_value.check('[energy]'):
            return (value * self.energy).to(unit)
