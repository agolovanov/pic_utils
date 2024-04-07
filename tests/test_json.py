import numpy as np
import pint

from pic_utils.json import PintJSONEncoder, PintJSONDecoder


def test_encoder_decoder():
    ureg = pint.UnitRegistry()

    data = {
        'string': 'test_string',
        'array': [1, 2, 3, 4.5],
        'number': 3.5,
        'int_number': 2,
        'numpy_array': np.array([1.2, 2.5, 3.3, 4.12, 5.23]),
        'pint_value': 2 * ureg.mm,
        'pint_array': np.array([1.5, 2.5, 3.5]) * ureg.C,
    }

    encoder = PintJSONEncoder()
    decoder = PintJSONDecoder(ureg)

    decoded = decoder.decode(encoder.encode(data))

    for k in data:
        assert k in decoded

        if isinstance(data[k], pint.Quantity):
            np.testing.assert_allclose(data[k].magnitude, decoded[k].magnitude)
            assert data[k].units == decoded[k].units
        elif isinstance(data[k], np.ndarray):
            np.testing.assert_allclose(data[k], decoded[k])
        else:
            assert data[k] == decoded[k]
