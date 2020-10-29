"""Define the 3D geometry primitives that the rest of DeepDRR would use.
"""

import numpy as np


class Point(object):
    pass

class Point2D(Point):
    def __init__(
        self,
        x: float,
        y: float,
    ):
        self.point = np.array([x, y, 1], dtype=np.float32)


class Point3D(Point):
    def __init__(
        self,
        x: float,
        y: float,
        z: float,
    ):
        self.point = np.array([x, y, z, 1], dtype=np.float32)