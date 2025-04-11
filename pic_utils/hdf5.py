import h5py
import pint
import numpy as np

import typing

if typing.TYPE_CHECKING:
    from pathlib import Path


def save_array(group: h5py.Group, name: str, array, overwrite=False, compressed=False):
    """Save an array to an HDF5 group.

    Supports saving pint.Quantity objects with units.

    Parameters
    ----------
    group : h5py.Group
        The group to which to save the array.
    name : str
        Name of the array in the group.
    array : np.ndarray or similar
        Array to save.
    overwrite : bool, optional
        Whether to overwrite existing data, by default False
    compressed : bool, optional
        Whether to compress the saved array, by default False

    """
    from .units import split_magnitude_units

    compression = 'gzip' if compressed else None

    if name not in group:
        print(f'{name} = {array}')
        if array is None:
            save_array(group, name, np.array([]), overwrite=overwrite, compressed=compressed)
            group[name].attrs['array_type'] = 'None'
        elif isinstance(array, pint.Quantity):
            array, units = split_magnitude_units(array)
            group.create_dataset(name, data=array, compression=compression)
            group[name].attrs['units'] = str(units)
            group[name].attrs['array_type'] = 'pint'
        else:
            group.create_dataset(name, data=array, compression=compression)
    elif overwrite:
        del group[name]
        save_array(group, name, array, overwrite=False)
    else:
        raise ValueError(f'{name} already exists')


def save_dict(group: 'h5py.Group | str | Path', data_dict: dict, overwrite=False, compressed=False):
    """Save a dictionary to an HDF5 group.

    Parameters
    ----------
    group : h5py.Group | str | Path
        The group to which to save the dictionary.
        If a string or Path, it will be opened as a file.
    data_dict : dict
        A dictionary to save.
    overwrite : bool, optional
        Whether to overwrite existing data, by default False
    compressed : bool, optional
        Whether to compress the saved data, by default False

    """
    if not isinstance(group, h5py.Group):
        group = h5py.File(group, 'a')

    for key, value in data_dict.items():
        if isinstance(value, dict):
            if key not in group:
                sub_group = group.create_group(key)
            else:
                sub_group = group[key]
                if not isinstance(sub_group, h5py.Group):
                    raise ValueError(f'{key} already exists and is not a group')

            save_dict(sub_group, value, overwrite=overwrite, compressed=compressed)
        else:
            save_array(group, key, value, overwrite=overwrite, compressed=compressed)


def load_array(group: h5py.Group, name: str):
    """Loads an array entry from an HDF5 group.

    Parameters
    ----------
    group : h5py.Group
        the group to load the array from
    name : str
        the name of the array

    Returns
    -------
    np.ndarray or pint.Quantity
        the loaded array, with units if available
    """
    if 'array_type' in group[name].attrs:
        if group[name].attrs['array_type'] == 'None':
            return None
    if 'units' in group[name].attrs:
        ureg = pint.get_application_registry()
        return group[name][()] * ureg[group[name].attrs['units']]
    else:
        value = group[name][()]
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        return value


def load_dict(group: 'h5py.Group | str | Path') -> dict:
    """Load a dictionary from an HDF5 group.

    Parameters
    ----------
    group : h5py.Group | str | Path
        The group to load the dictionary from.
        If a string or Path, it will be opened as a file.

    Returns
    -------
    dict
        dictionary with the loaded data
    """
    if not isinstance(group, h5py.Group):
        group = h5py.File(group, 'r')

    data_dict = {}
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            data_dict[key] = load_dict(group[key])
        else:
            data_dict[key] = load_array(group, key)
    return data_dict
