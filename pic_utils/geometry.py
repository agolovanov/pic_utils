import numpy as np

class Plane:
    def __init__(self, origin, v1, v2):
        """Creates a plane representation

        Creates a plane given by r = origin + v1 * x + v2 * y

        v1 and v2 are assumed to be orthogonal, although not necessarily normalized

        Parameters
        ----------
        origin : array of size 3
        v1 : array of size 3
        v2 : array of size 3

        Raises
        ------
        ValueError
            if v1 and v2 are not orthogonal
        """
        self.origin = np.array(origin, dtype=float)
        self.v1 = np.array(v1, dtype=float)
        self.v2 = np.array(v2, dtype=float)

        # normalizing the vectors
        self.v1 /= vector_norm(self.v1)
        self.v2 /= vector_norm(self.v2)

        if np.abs(np.dot(self.v1, self.v2) > 1e-5):
            raise ValueError(f'Vectors {v1} and {v2} and not orthogonal')
        self.norm = np.cross(self.v1, self.v2)

    def check_normal(self, v):
        return are_parallel(v, self.norm)


def coordinate_plane(axis: str, coordinate=0.0):
    """Creates one of the coordinate planes perpendicular to the coordinate axes.

    Parameters
    ----------
    axis : 'x', 'y' or 'z'.
        The axis perpendicular to which to take a plane.
        The corresponding planes will be yz, zx, and xy
    coordinate : float, optional
        the coordinate of the plane along the axis, by default 0.0
    """
    if axis == 'x':
        return Plane((coordinate, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    elif axis == 'y':
        return Plane((0.0, coordinate, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    elif axis == 'z':
        return Plane((0.0, 0.0, coordinate), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def vector_norm(v):
    return np.linalg.norm(v)


def are_parallel(v1, v2):
    return np.allclose(np.abs(np.dot(v1, v2)), vector_norm(v1) * vector_norm(v2))