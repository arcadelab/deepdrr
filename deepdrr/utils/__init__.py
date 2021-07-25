import pickle
from pathlib import Path
from datetime import datetime
import PIL.Image as Image
import numpy as np
import os
import logging
from typing import Optional, TypeVar, Any, Tuple, Union, List

from . import data_utils, image_utils, test_utils

__all__ = ["param_saver", "one_hot", "tuplify", "listify",
           "radians", "generate_uniform_angles", "neglog",
           "try_import_pyvista", "try_import_vtk"]


logger = logging.getLogger(__name__)


def param_saver(
    thetas, phis, proj_mats, camera, origin, photons, spectrum, prefix, save_path
):
    """Save the paramaters.

    This function may be deprecated.

    Args:
        thetas ([type]): [description]
        phis ([type]): [description]
        proj_mats ([type]): [description]
        camera ([type]): [description]
        origin ([type]): [description]
        photons ([type]): [description]
        spectrum ([type]): [description]
        prefix ([type]): [description]
        save_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    i0 = np.sum(spectrum[:, 0] * (spectrum[:, 1] /
                np.sum(spectrum[:, 1]))) / 1000
    data = {
        "date": datetime.now(),
        "thetas": thetas,
        "phis": phis,
        "proj_mats": proj_mats,
        "camera": camera,
        "origin": origin,
        "photons": photons,
        "spectrum": spectrum,
        "I0": i0,
    }
    with open(Path(save_path) / f"{prefix}.pickle", "wb") as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return True


def one_hot(
    x: np.ndarray, num_classes: Optional[int] = None, axis: int = -1,
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


T = TypeVar("T")


def tuplify(t: Union[Tuple[T, ...], T], n: int = 1) -> Tuple[T, ...]:
    """ Create a tuple with `n` copies of `t`,  if `t` is not already a tuple of length `n`."""
    if isinstance(t, (tuple, list)):
        assert len(t) == n
        return tuple(t)
    else:
        return tuple(t for _ in range(n))


def listify(x: Union[List[T], T], n: int = 1) -> List[T]:
    if isinstance(x, list):
        return x
    else:
        return [x] * n


def radians(
    *ts: Union[float, np.ndarray], degrees: bool = True
) -> Union[float, List[float]]:
    """Convert to radians.

    Args:
        ts: the angle or array of angles.
        degrees (bool, optional): whether the inputs are in degrees. If False, this is a no-op. Defaults to True.

    Returns:
        Union[float, List[float]]: each argument, converted to radians.
    """
    if degrees:
        ts = [np.radians(t) for t in ts]
    return ts[0] if len(ts) == 1 else ts


def generate_uniform_angles(
    phi_range: Tuple[float, float, float], theta_range: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a uniform sampling of angles over the given ranges.

    If inputs are in degrees, so will the outputs be.

    Args:
        phi_range (Tuple[float, float, float]): range of angles phi in (min, max, step) form, in degrees.
        theta_range (Tuple[float, float, float]): range of angles theta in (min, max, step) form, in degrees.

    Returns:
        Tuple[np.ndarray, np.ndarray]: phis, thetas over uniform angles, in radians.
    """
    min_theta, max_theta, spacing_theta = theta_range
    min_phi, max_phi, spacing_phi = phi_range
    thetas = np.array(
        np.arange(min_theta, max_theta + spacing_theta / 2, step=spacing_theta)
    )
    num_thetas = len(thetas)
    phis = np.array(np.arange(min_phi, max_phi, step=spacing_phi))
    num_phis = len(phis)
    thetas = np.tile(thetas, num_phis)
    phis = phis.repeat(num_thetas, 0)
    return phis, thetas


def neglog(image: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """Take the negative log transform of an intensity image.

    Args:
        image (np.ndarray): a single 2D image, or N such images.
        epsilon (float, optional): positive offset from 0 before taking the logarithm.

    Returns:
        np.ndarray: the image or images after a negative log transform, scaled to [0, 1]
    """
    image = np.array(image)
    shape = image.shape
    if len(shape) == 2:
        image = image[np.newaxis, :, :]

    # shift image to avoid invalid values
    image += image.min(axis=(1, 2), keepdims=True) + epsilon

    # negative log transform
    image = -np.log(image)

    # linear interpolate to range [0, 1]
    image_min = image.min(axis=(1, 2), keepdims=True)
    image_max = image.max(axis=(1, 2), keepdims=True)
    if np.any(image_max == image_min):
        logger.warning(
            f"mapping constant image to 0. This probably indicates the projector is pointed away from the volume."
        )
        # TODO(killeen): for multiple images, only fill the bad ones
        image[:] = 0
        if image.shape[0] > 1:
            logger.error(
                "TODO: zeroed all images, even though only one might be bad.")
    else:
        image = (image - image_min) / (image_max - image_min)

    if np.any(np.isnan(image)):
        logger.warning(f"got NaN values from negative log transform.")

    if len(shape) == 2:
        return image[0]
    else:
        return image


def try_import_pyvista():
    try:
        import pyvista as pv

        pv_available = True
    except ImportError:
        pv = None
        pv_available = False

    return pv, pv_available


def try_import_vtk():
    try:
        import vtk
        from vtk.util import numpy_support as nps

        vtk_available = True
    except ImportError:
        vtk = None
        nps = None
        vtk_available = False

    return vtk, nps, vtk_available
