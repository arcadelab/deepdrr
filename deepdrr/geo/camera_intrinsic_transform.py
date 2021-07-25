from __future__ import annotations
from typing import Union, Tuple, Optional

import numpy as np

from .core import point, Point2D, Transform, FrameTransform
from .. import utils


class CameraIntrinsicTransform(FrameTransform):
    dim: int = 2
    input_dim: int = 2

    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data)
        assert self.data.shape == (3, 3), f"unrecognized shape: {self.data.shape}"

    @classmethod
    def from_parameters(
        cls,
        optical_center: Point2D,
        focal_length: Union[float, Tuple[float, float]] = 1,
        shear: float = 0,
        aspect_ratio: Optional[float] = None,
    ) -> CameraIntrinsicTransform:
        """The camera intrinsic matrix.

        The intrinsic matrix is fundamentally a FrameTransform in 2D, namely `index_from_camera2d`.
        It transforms to the index-space of the image (as mapped on the sensor)
        from the index-space centered on the principle ray.

        Note:
            Focal lengths are often measured in world units (e.g. millimeters.),
            but here they are in pixels.
            The conversion can be taken from the size of a pixel.

        Useful references include Szeliski's "Computer Vision"
        - https://ksimek.github.io/2013/08/13/intrinsic/

        Args:
            optical_center (Point2D): the index-space point where the isocenter (or pinhole) is centered.
            focal_length (Union[float, Tuple[float, float]]): the focal length in index units. Can be a tubple (f_x, f_y),
                or a scalar used for both, or a scalar modified by aspect_ratio, in index units.
            shear (float): the shear `s` of the camera.
            aspect_ratio (Optional[float], optional): the aspect ratio `a` (for use with one focal length). If not provided, aspect
                ratio is 1. Defaults to None.

        Returns:
            CameraIntrinsicTransform: The camera intrinsic matrix.

        """
        optical_center = point(optical_center)
        assert optical_center.dim == 2, "center point not in 2D"

        cx, cy = np.array(optical_center)

        if aspect_ratio is None:
            fx, fy = utils.tuplify(focal_length, 2)
        else:
            assert isinstance(
                focal_length, (float, int)
            ), "cannot use aspect ratio if both focal lengths provided"
            fx, fy = (focal_length, aspect_ratio * focal_length)

        data = np.array([[fx, shear, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)

        return cls(data)

    @classmethod
    def from_sizes(
        cls,
        sensor_size: Union[int, Tuple[int, int]],
        pixel_size: Union[float, Tuple[float, float]],
        source_to_detector_distance: float,
    ) -> CameraIntrinsicTransform:
        """Generate the camera from human-readable parameters.

        This is the recommended way to create the camera. Note that although pixel_size and source_to_detector distance are measured in world units,
        the camera intrinsic matrix contains no information about the world, as these are merely used to compute the focal length in pixels.

        Args:
            sensor_size (Union[float, Tuple[float, float]]): (width, height) of the sensor, or a single value for both, in pixels.
            pixel_size (Union[float, Tuple[float, float]]): (width, height) of a pixel, or a single value for both, in world units (e.g. mm).
            source_to_detector_distance (float): distance from source to detector in world units.

        Returns:

        """
        sensor_size = utils.tuplify(sensor_size, 2)
        pixel_size = utils.tuplify(pixel_size, 2)
        fx = source_to_detector_distance / pixel_size[0]
        fy = source_to_detector_distance / pixel_size[1]
        optical_center = point(sensor_size[0] / 2, sensor_size[1] / 2)
        return cls.from_parameters(optical_center=optical_center, focal_length=(fx, fy))

    @property
    def optical_center(self) -> Point2D:
        return Point2D(self.data[:, 2])

    @property
    def fx(self) -> float:
        return self.data[0, 0]

    @property
    def fy(self) -> float:
        return self.data[1, 1]

    @property
    def aspect_ratio(self) -> float:
        """Image aspect ratio."""
        return self.fy / self.fx

    @property
    def focal_length(self) -> float:
        """Focal length in pixels."""
        return self.fx

    @property
    def sensor_width(self) -> int:
        """Get the sensor width in pixels.

        Based on the convention of origin in top left, with x pointing to the right and y pointing down."""
        return int(np.ceil(2 * self.data[0, 2]))

    @property
    def sensor_height(self) -> int:
        """Get the sensor height in pixels.

        Based on the convention of origin in top left, with x pointing to the right and y pointing down."""
        return int(np.ceil(2 * self.data[1, 2]))

    @property
    def sensor_size(self) -> Tuple[int, int]:
        """Tuple with the (width, height) of the sense/image, in pixels."""
        return (self.sensor_width, self.sensor_height)
