import pint


def strip_units(value, target_unit: pint.Unit | str):
    """Takes a value that may or may not have units
    and returns the magnitude in the specified target unit.
    If the value is a plain number, it is assumed to be in the target unit.

    Parameters
    ----------
    value :
        the value to be reduced
    target_unit : str or pint.Unit
        the desired unit

    Returns
    -------
    float or array
        the value without units
    """
    if isinstance(value, pint.Quantity):
        return value.m_as(target_unit)
    else:
        return value


def ensure_units(value, units: pint.Unit | str | None):
    """Takes the value that may or may not have units and returns the value with specified units.

    Parameters
    ----------
    value : pint.Quantity | float
        the value to be returned
    units : pint.Unit | str | None
        the desired unit. If None is given, the value will be returned without checks.

    Returns
    -------
    pint.Quantity
        value in the desired units
    """

    if units is None:
        return value

    if isinstance(value, pint.Quantity):
        return value.to(units)
    else:
        if not isinstance(units, pint.Unit):
            ureg = pint.get_application_registry()
            units = ureg[units]
        return value * units


def split_magnitude_units(value) -> tuple:
    """Returns magnitude and units of the value.
    If the value does not have units, retuns `None` in the place of units.

    Parameters
    ----------
    value :
        the value to be returned

    Returns
    -------
    tuple
        (magnitude, units) of the value. If the value does not have units, `units` will be `None`.
    """
    if isinstance(value, pint.Quantity):
        return value.magnitude, value.units
    else:
        return value, None
