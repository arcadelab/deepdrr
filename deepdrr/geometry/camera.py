from __future__ import annotations

from typing import Union, Tuple, Iterable, List
from numpy.typing import ArrayLike

import numpy as np
import logging

from .projection import Projection


class Camera(object):
    """Contains all the info you need about the camera and its projections."""

    def __init__(
        self,
        intrinsic_matrix: ArrayLike,
        isocenter_distance: float,
    ) -> None:
        """Generate the camera object.

        Args:
            intrinsic_matrix (ArrayLike): the camera intrinsic matrix K.
            isocenter_distance (float): the isocenter is the point through which the central ray of the radiation beams passes.
        """
        self.intrinsic_matrix = intrinsic_matrix
        self.isocenter_distance = isocenter_distance

    @classmethod
    def from_intrinsic_matrix(cls, intrinsic_matrix: ArrayLike) -> Camera:
        return cls(intrinsic_matrix)

    @classmethod
    def from_parameters(
        cls, 
        sensor_size: Union[float, Tuple[float, float]],
        pixel_size: Union[float, Tuple[float, float]],
        source_to_detector_distance: float,
        isocenter_distance: float,
    ) -> Camera:
        """Generate the camera from more human-readable parameters.

        Args:
            sensor_size (Union[float, Tuple[float, float]]): the width and height of the sensor, or a single value for both.
            pixel_size (Union[float, Tuple[float, float]]): the width and height of a pixel, or a single value for both.
            source_to_detector_distance (float): distance from source to detector
            isocenter_distance (float): isocenter distance

        Returns:
            Camera: camera object
        """

        K = np.zeros(3, 3)
        K[0, 0] = source_to_detector_distance / pixel_size[0]
        K[1, 1] = source_to_detector_distance / pixel_size[1]
        K[0, 2] = sensor_size[0] / 2
        K[1, 2] = sensor_size[1] / 2
        K[2, 2] = 1.0
        return cls(K, isocenter_distance)

    @property
    def K(self):
        return self.intrinsic_matrix

    @staticmethod
    def make_rotation(
        

    ):

    def make_projections(
        self, 
        phis: List[float],
        thetas: List[float],
        rhos: Optional[List[float]] = None,
        offsets: Optional[List[float]] = None,
    ) -> List[Projection]:

        assert len(phis) == len(thetas), 'unequal lengths'

        num_projections = len(phis)
        logging.info(f"generating {num_projections} projections")

        if rhos is None:
            rho_list = [0 for _ in range(num_projections)]

        if offsets is None:
            offset_list = [np.zeros(3) for _ in range(num_projections)]
            
        prejections = []
        for phi, theta, rho, offset in zip(phis, thetas, rhos, offsets):
            R = generat_rotation_from_angles(phi, theta, rho)
            t = generate_translation(isocenter_distance, offset[0], offset[1], offset[2])
            matrices.append(ProjMatrix(R, K, t))
        return matrices