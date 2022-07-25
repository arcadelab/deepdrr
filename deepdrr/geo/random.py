from typing import List, Optional, overload
import numpy as np
from .core import Vector3D, vector


def _sample_spherical(d_phi: float, n: int) -> np.ndarray:
    """Sample n vectors within `phi` radians of [0, 0, 1]."""
    theta = np.random.uniform(0, 2 * np.pi, n)

    phi = np.arccos(np.random.uniform(np.cos(d_phi), 1, n))

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.stack([x, y, z], axis=1)


@overload
def spherical_unifrom(center: Vector3D, d_phi: float, n: int) -> List[Vector3D]:
    ...


@overload
def spherical_unifrom(center: Vector3D, d_phi: float, n: None) -> Vector3D:
    ...


def spherical_uniform(center=[0, 0, 1], d_phi=np.pi, n=None):
    """Sample unit vectors within `d_phi` radians of `v`."""
    v = vector(center).hat()
    points = _sample_spherical(d_phi, 1 if n is None else n)
    F = v.rotation(vector(0, 0, 1))
    if n is None:
        return F @ vector(points[0])
    else:
        return [F @ vector(p) for p in points]
