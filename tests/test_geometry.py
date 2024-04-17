from hypothesis import given, strategies as st, settings
import numpy as np
from pic_utils.geometry import vector_norm


@given(st.floats(allow_nan=False, min_value=1e-5, max_value=1e5))
def test_vector_norm_orthogonal(norm):
    vectors = [
        np.array((norm, 0, 0)),
        np.array((-norm, 0, 0)),
        np.array((0, norm, 0)),
        np.array((0, -norm, 0)),
        np.array((0, 0, norm)),
        np.array((0, 0, -norm)),
        np.array((norm / np.sqrt(2), norm / np.sqrt(2), 0)),
        np.array((norm / np.sqrt(3), norm / np.sqrt(3), norm / np.sqrt(3))),
    ]

    for v in vectors:
        np.testing.assert_allclose(vector_norm(v), norm)
