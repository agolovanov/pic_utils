import numpy as np
import wake_t


def bunch_to_wake_t(data, propagation_distance=None):
    """

    Parameters
    ----------
    data : dict or similar
        bunch in a dict-like format, has to contain fields x, y, z, ux, uy, uz, w (weight)
    propagation_distance : float, optional
        the current propagation distance of the bunch, by default the mean coordinate of the bunch

    Returns
    -------
    waket.ParticleBunch
        a particle bunch in the Wake-T format
    """
    if propagation_distance is None:
        z0 = np.mean(data['z'])
    else:
        z0 = propagation_distance.m_as('m')
    return wake_t.ParticleBunch(
        np.array(data['w']),
        np.array(data['x']),
        np.array(data['y']),
        np.array(data['z']) - z0,
        np.array(data['ux']),
        np.array(data['uy']),
        np.array(data['uz']),
        prop_distance=z0)


def wake_t_to_dict(bunch: wake_t.ParticleBunch):
    """Converts a Wake-T bunch to a dictionary

    Parameters
    ----------
    bunch : waket.ParticleBunch
        the bunch

    Returns
    -------
    dict
        dictionary containing x, y, z, ux, uy, uz, w keys with arrays
    """
    return {
        'x': np.array(bunch.x),
        'y': np.array(bunch.y),
        'z': np.array(bunch.xi) + bunch.prop_distance,
        'ux': np.array(bunch.px),
        'uy': np.array(bunch.py),
        'uz': np.array(bunch.pz),
        'w': np.array(bunch.w)
    }


def wake_t_to_pandas(bunch: wake_t.ParticleBunch):
    import pandas as pd
    return pd.DataFrame(wake_t_to_dict(bunch), index=np.arange(np.array(bunch.x).size))