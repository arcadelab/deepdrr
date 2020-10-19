from __future__ import annotations

from typing import Union, Tuple, Iterable, List
from numpy.typing import ArrayLike

import numpy as np


class Projection(object):
    def __init__(
        self,
        R: ArrayLike,
        K: ArrayLike,
        t: ArrayLike,
    ) -> None:
        """Make a projection matrix from camera parameters.

        Args:
            R (ArrayLike): rotation matrix of extrinsic parameters
            K (ArrayLike): camera intrinsic matrix
            t (ArrayLike): translation matrix of extrinsic parameters
        """
        self.R = np.array(R, dtype=np.float32)
        self.t = np.array(t, dtype=np.float32)
        self.K = np.array(K, dtype=np.float32)
        self.P = np.matmul(self.K, np.concatenate((self.R, np.expand_dims(self.t, 1)), axis=1))
        self.rtk_inv = np.matmul(np.transpose(self.R), np.linalg.inv(self.K))

    @classmethod
    def from_camera_parameters(
        cls,
        intrinsic: ArrayLike,
        extrinsic: Tuple[ArrayLike, ArrayLike],
    ) -> ProjMatrix:
        """Alternative to the init function, more readable.

        Args:
            intrinsic (ArrayLike): intrinsic camera matrix
            extrinsic (Tuple[ArrayLike, ArrayLike]): the tuple extrinsic parameters [R, T]

        Returns:
            ProjMatrix: a projection matrix object
        """
        R, t = extrinsic
        K = intrinsic
        return cls(R, K, t)

    def get_rtk_inv(self):
        return self.rtk_inv

    def get_projection(self):
        return self.P

    def get_camera_ceter(self):
        return -np.matmul(np.transpose(self.R), self.t)

    def get_principle_axis(self):
        axis = self.R[2, :] / self.K[2, 2]
        return axis

    def get_conanical_proj_matrix(self, voxel_size, volume_size, origin_shift):
        inv_voxel_scale = np.zeros([3, 3])
        inv_voxel_scale[0][0] = 1 / voxel_size[0]
        inv_voxel_scale[1][1] = 1 / voxel_size[1]
        inv_voxel_scale[2][2] = 1 / voxel_size[2]
        inv_ar = np.matmul(inv_voxel_scale, self.rtk_inv)

        source_point = np.zeros((3, 1), dtype=np.float32)
        camera_ceter = - self.get_camera_ceter()
        source_point[0] = -(-0.5 * (volume_size[0] - 1.0) + origin_shift[0] * inv_voxel_scale[0, 0] + inv_voxel_scale[0, 0] * camera_ceter[0])
        source_point[1] = -(-0.5 * (volume_size[1] - 1.0) + origin_shift[1] * inv_voxel_scale[1, 1] + inv_voxel_scale[1, 1] * camera_ceter[1])
        source_point[2] = -(-0.5 * (volume_size[2] - 1.0) + origin_shift[2] * inv_voxel_scale[2, 2] + inv_voxel_scale[2, 2] * camera_ceter[2])
        return inv_ar, source_point