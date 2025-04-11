from pic_utils import hdf5
import numpy as np
import pint

from utils import assert_dicts_equal


ureg = pint.get_application_registry()


def test_save_load_dict(tmp_path):
    d = {
        'a': np.linspace(0, 10, 100) ** 2,
        'b': np.linspace(0, 20, 100).reshape((5, -1)) ** 3 * ureg.kg / ureg.cm**-3,
        'c': {'c_sub': np.linspace(5, -3, 10)},
        'd': 10,
        'e': 9.7 * ureg.keV,
        'f': None,
        'g': np.ndarray([]),
        'h': 'TestString',
        'i': 'ТестоваяСтрока',
    }

    output = tmp_path / 'test.h5'

    hdf5.save_dict(output, d)
    d_read = hdf5.load_dict(output)

    assert_dicts_equal(d, d_read)

    d['a'] = np.linspace(-3, 12, 50)
    hdf5.save_dict(output, d, overwrite=True)
    d_read = hdf5.load_dict(output)

    assert_dicts_equal(d, d_read)
