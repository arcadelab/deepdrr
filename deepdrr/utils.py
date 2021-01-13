from typing import Optional, TypeVar, Any, Tuple, Union, List

import logging
import os
import numpy as np
import PIL.Image as Image
from datetime import datetime
from pathlib import Path
from numpy.lib.function_base import interp
from scipy.optimize import curve_fit
import pickle


logger = logging.getLogger(__name__)


def image_saver(
    images: np.ndarray,
    prefix: str,
    path: str
) -> bool:
    """Save the images as tiff

    Args:
        images (np.ndarray): array of images
        prefix (str): prefix for each file name
        path (str): path to directory to save the files in

    Returns:
        bool: return code.
    """

    for i in range(0, images.shape[0]):
        image_pil = Image.fromarray(images[i, :, :])
        image_pil.save(Path(path) / f"{prefix}{str(i).zfill(5)}.tiff")
    return True


def param_saver(thetas, phis, proj_mats, camera, origin, photons, spectrum, prefix, save_path):
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


def radians(*ts: Union[float, np.ndarray], degrees: bool = True) -> Union[float, List[float]]:
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
    phi_range: Tuple[float, float, float],
    theta_range: Tuple[float, float, float],
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
    thetas = np.array(np.arange(min_theta, max_theta + spacing_theta / 2, step=spacing_theta))
    num_thetas = len(thetas)
    phis = np.array(np.arange(min_phi, max_phi, step=spacing_phi))
    num_phis = len(phis)
    thetas = np.tile(thetas, num_phis)
    phis = phis.repeat(num_thetas, 0)
    return phis, thetas


def neglog(image, I_0=1):
    """Negative log transform.

    Args:
        image (np.ndarray): the image, as output by projector. Assumes last two dimensions are height and width.
        I_0 (int, optional): I_0. Defaults to 1.

    Returns:
        np.ndarray: Image with neg_log transform applied.
    """
    if np.all(image == 0):
        logger.warning(f'image is all 0')
        return image
        
    min_nonzero_value = image[image > 0].min()
    return np.where(image == 0, min_nonzero_value, -np.log(image / I_0))