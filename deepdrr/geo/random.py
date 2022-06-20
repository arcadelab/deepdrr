from typing import List
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


def spherical_uniform(center: Vector3D = [0, 0, 1], d_phi: float = np.pi, n: int = 1) -> List[Vector3D]:
    """Sample unit vectors within `d_phi` radians of `v`."""
    v = vector(center).hat()
    points = _sample_spherical(d_phi, n)
    F = v.rotation(vector(0, 0, 1))
    return [F @ vector(p) for p in points]
