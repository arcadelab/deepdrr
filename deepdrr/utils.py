from typing import Optional, TypeVar, Any, Tuple, Union

import os
import numpy as np
import PIL.Image as Image
from datetime import datetime
from pathlib import Path
from numpy.lib.function_base import interp
from scipy.optimize import curve_fit
import pickle


def image_saver(images, prefix, path):
    for i in range(0, images.shape[0]):
        image_pil = Image.fromarray(images[i, :, :])
        image_pil.save(Path(path) / f"{prefix}{str(i).zfill(5)}.tiff")
    return True


def param_saver(thetas, phis, proj_mats, camera, origin, photons, spectrum, prefix, save_path):
    i0 = np.sum(spectrum[:, 0] * (spectrum[:, 1] / np.sum(spectrum[:, 1]))) / 1000
    data = {"date": datetime.now(), "thetas": thetas, "phis": phis, "proj_mats": proj_mats, "camera": camera, "origin": origin, "photons": photons, "spectrum": spectrum, "I0": i0}
    with open(Path(save_path) / f"{prefix}.pickle", 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return True


def one_hot(
    x: np.ndarray, 
    num_classes: Optional[int] = None,
    axis: int = -1,
) -> np.ndarray:
    """One-hot encode the vector x along the axis.

    Args:
        x (np.ndarray): n-dim array x.
        num_classes (Optional[int]): number of classes. Uses maximum label if not provided.
        axis (int): the axis to insert the labels along.

    Returns:
        np.ndarray: one-hot encoded labels with n + 1 axes.
    """
    if num_classes is None:
        num_classes = x.max()

    x = x[..., np.newaxis] == np.arange(num_classes + 1)
    if axis != -1:
        # copy x to actually move the axis, not just make a new view.
        x = np.moveaxis(x, -1, axis).copy()
        
    return x


T = TypeVar('T')


def tuplify(t: Union[Tuple[T,...], T], n: int) -> Tuple[T,...]:
    """ Create a tuple with `n` copies of `t`,  if `t` is not already a tuple of length `n`."""
    if isinstance(t, Tuple):
        assert len(t) == n
        return t
    else:
        return tuple(t for _ in range(n))


def make_detector_rotation(phi, theta, rho):
    # rotation around phi and theta
    sin_p = np.sin(phi)
    neg_cos_p = -np.cos(phi)
    z = 0
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    omc = 1 - cos_t

    # Rotation by theta about vector [sin(phi), -cos(phi), z].
    R = np.array([
        [
            sin_p * sin_p * omc + cos_t,
            sin_p * neg_cos_p * omc - z * sin_t, 
            sin_p * z * omc + neg_cos_p * sin_t,
        ],
        [
            sin_p * neg_cos_p * omc + z * sin_t,
            neg_cos_p * neg_cos_p * omc + cos_t,
            neg_cos_p * z * omc - sin_p * sin_t,
        ],
        [
            sin_p * z * omc - neg_cos_p * sin_t,
            neg_cos_p * z * omc + sin_p * sin_t,
            z * z * omc + cos_t,
        ]])
    # rotation around detector priniciple axis
    rho = -phi + np.pi * 0.5 + rho
    R_principle = np.array([[np.cos(rho), -np.sin(rho), 0],
                            [np.sin(rho), np.cos(rho), 0],
                            [0, 0, 1]])
    R = np.matmul(R_principle, R)

    return R