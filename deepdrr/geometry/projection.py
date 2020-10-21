from __future__ import annotations

from typing import Union, Tuple, Iterable, List

import numpy as np
from pathlib import Path


class Projection(object):
    def __init__(
        self,
        R: np.ndarray,
        K: np.ndarray,
        t: np.ndarray,
    ) -> None:
        """Make a projection matrix from camera parameters.

        Args:
            R (np.ndarray): rotation matrix of extrinsic parameters
            K (np.ndarray): camera intrinsic matrix
            t (np.ndarray): translation matrix of extrinsic parameters
        """
        self.R = np.array(R, dtype=np.float32)
        self.t = np.array(t, dtype=np.float32)
        self.K = np.array(K, dtype=np.float32)
        self.P = np.matmul(self.K, np.concatenate((self.R, np.expand_dims(self.t, 1)), axis=1))
        self.rtk_inv = np.matmul(np.transpose(self.R), np.linalg.inv(self.K))

    @classmethod
    def from_camera_parameters(
        cls,
        intrinsic: np.ndarray,
        extrinsic: Tuple[np.ndarray, np.ndarray],
    ) -> ProjMatrix:
        """Alternative to the init function, more readable.

        Args:
            intrinsic (np.ndarray): intrinsic camera matrix
            extrinsic (Tuple[np.ndarray, np.ndarray]): the tuple extrinsic parameters [R, T]

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

    def get_camera_center(self):
        return -np.matmul(np.transpose(self.R), self.t)

    def get_principle_axis(self):
        axis = self.R[2, :] / self.K[2, 2]
        return axis

    def get_canonical_matrix(
        self, 
        voxel_size: np.ndarray, 
        volume_size: np.ndarray, 
        origin_shift: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the inverse transformation matrix and the source point for the projection.

        Args:
            voxel_size (np.ndarray): size of a voxel of the volume in [x, y, z]
            volume_size (np.ndarray): size of the volume in [x, y, z] (TODO: world space or volume index space?)
            origin_shift (np.ndarray): shift of the origin in world space.

        Returns:
            Tuple[np.ndarray, np.ndarray]: [description]
        """
        inv_ar = np.diag(1 / voxel_size) @ self.rtk_inv
        camera_center = -self.get_camera_center() # why is this negated if the function is too?
        source_point = (volume_size - 1) / 2 - origin_shift / voxel_size - camera_center / voxel_size
        return inv_ar, source_point

    
def load_projections(
    path: str,
    lim: int = 100000000,
) -> List[Projection]:
    """Load all the projections saved in the directory at `path`

    Args:
        path (str): path to the directory containing R.txt, T.txt, and K.txt.
        lim (int, optional): Limits number of projections to read. Defaults to 100000000.

    Returns:
        List[Projection]: list of the projections
    """
    root = Path(path)
    Rs = np.loadtxt(root / 'R.txt', max_rows=lim)[:, 0:9].reshape(-1, 3, 3)
    Ks = np.loadtxt(root / 'K.txt', max_rows=lim)[:, 0:3]
    ts = np.loadtxt(root / 'T.txt', max_rows=lim)[:, 0:9].reshape(-1, 3, 3)
    return [Projection(R, K, t) for R, K, t in zip(Rs, Ks, ts)]
