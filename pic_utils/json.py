import json

import numpy as np
import pint


class PintJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {'numpy_data': obj.tolist()}
        if isinstance(obj, pint.Quantity):
            return {'pint_data': obj.magnitude, 'pint_units': str(obj.units)}

        return json.JSONEncoder.default(self, obj)


class PintJSONDecoder(json.JSONDecoder):
    def __init__(self):
        self.unit_registry = pint.get_application_registry()
        super().__init__(object_hook=lambda jdict: self.parse_dictionary(jdict))

    def parse_dictionary(self, jdict):
        if not isinstance(jdict, dict):
            return jdict
        if 'numpy_data' in jdict:
            return np.array(jdict['numpy_data'])
        if 'pint_data' in jdict:
            return self.parse_dictionary(jdict['pint_data']) * self.unit_registry[jdict['pint_units']]
        return jdict


def save(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, cls=PintJSONEncoder, indent=2, sort_keys=True, ensure_ascii=False)


def load(json_path):
    decoder = PintJSONDecoder()
    with open(json_path) as f:
        return json.load(f, object_hook=lambda o: decoder.parse_dictionary(o))
