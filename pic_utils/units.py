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