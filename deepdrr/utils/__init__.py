import pickle
from pathlib import Path
from datetime import datetime
import PIL.Image as Image
import numpy as np
import os
import logging
from typing import Optional, TypeVar, Any, Tuple, Union, List, overload, Dict
import math

from .data_utils import jsonable
from . import data_utils, image_utils, test_utils

__all__ = [
    "param_saver",
    "one_hot",
    "tuplify",
    "listify",
    "radians",
    "generate_uniform_angles",
    "neglog",
    "try_import_pyvista",
    "try_import_vtk",
    "jsonable",
    "mappable",
]


logger = logging.getLogger(__name__)

S = TypeVar("S")
T = TypeVar("T")


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
    i0 = np.sum(spectrum[:, 0] * (spectrum[:, 1] / np.sum(spectrum[:, 1]))) / 1000
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


T = TypeVar("T")


def tuplify(t: Union[Tuple[T, ...], T], n: int = 1) -> Tuple[T, ...]:
    """Create a tuple with `n` copies of `t`,  if `t` is not already a tuple of length `n`."""
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


@overload
def radians(t: float, degrees: bool) -> float:
    ...


@overload
def radians(t: np.ndarray, degrees: bool) -> np.ndarray:
    ...


@overload
def radians(ts: List[T], degrees: bool) -> List[T]:
    ...


@overload
def radians(ts: Dict[S, T], degrees: bool) -> Dict[S, T]:
    ...


@overload
def radians(*ts: T, degrees: bool) -> List[T]:
    ...


def radians(*args, degrees=True):
    """Convert to radians.

    Args:
        ts: the angle or array of angles.
        degrees (bool, optional): whether the inputs are in degrees. If False, this is a no-op. Defaults to True.

    Returns:
        Union[float, List[float]]: each argument, converted to radians.
    """
    if len(args) == 1:
        if isinstance(args[0], (float, int)):
            return math.radians(args[0]) if degrees else args[0]
        elif isinstance(args[0], dict):
            return {k: radians(v, degrees=degrees) for k, v in args[0].items()}
        elif isinstance(args[0], (list, tuple)):
            return [radians(t, degrees=degrees) for t in args[0]]
        elif isinstance(args[0], np.ndarray):
            return np.radians(args[0]) if degrees else args[0]
        else:
            raise TypeError(f"Cannot convert {type(args[0])} to radians.")
    elif isinstance(args[-1], bool):
        return radians(*args[:-1], degrees=args[-1])
    else:
        return [radians(t, degrees=degrees) for t in args]


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
        logger.debug(
            f"mapping constant image to 0. This probably indicates the projector is pointed away from the volume."
        )
        # TODO(killeen): for multiple images, only fill the bad ones
        image[:] = 0
        if image.shape[0] > 1:
            logger.error("TODO: zeroed all images, even though only one might be bad.")
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


def mappable(
    ndim: Union[int, List[int]] = 1, every: bool = False, method: bool = False
):
    """Decorator for funcs that take a n-D array x.

    Maps func across the last axis for arrays with ndim > 1. Assumes that the array `x` is the last
    positional argument to the function, allowing for methods of a class or functions that take
    other parameters first, unless `every` is `True`. The function may also take keyword arguments,
    of course, and these are passed unchanged.

    Args:
        ndim: ndim(s) of argument(s) that the function expects. Either an int (same for each) or a
            list of ints, one for each argument. 0 indicates a scalar.
        every (bool): whether every argument is mappable (of base dimension n) or just the last argument.
        method (bool): whether the function is a method of a class instance.

    """

    def decorator(func):
        """Decorator function.

        If func returns an array y with shape [M0, M1, ...], the decorated function
        returns array with shape [P1, ..., Pm, M0, M1, ...], where P1, ..., Pm are
        the preceding dimensions of the inputs to the decorated fucntion. If func
        returns a tuple of such arrays, this mapping is performed for each one. If
        func returns a singular object/scalar, then the returned array has shape [P1, ..., Pm].

        Args:
            func: a function that takes in a 1D array x of shape [N1, N2, ..., Nn].

        """

        def wrapper(*args, **kwargs):
            # Determine which args are params (e.g. the `self` for a method) and
            # which are mappable args (margs).
            margs: List[np.ndarray]
            if every and method:
                margs = args[1:]
                params = args[:1]
            elif every and not method:
                margs = args
                params = []
            else:
                margs = args[-1:]
                params = args[:-1]

            # check that the ndims for the inputs make sense. If no mapping required,
            # return the function output.
            ndims = listify(ndim, len(margs))
            if margs[0].ndim == ndims[0]:
                return func(*params, *margs, **kwargs)
            if margs[0].ndim < ndims[0]:
                raise RuntimeError(f"each arg must have rank >= {ndim}")

            # determine the shape of the inputs and prepare to iterate over them by
            # flattening redundant dimensions
            preshape = margs[0].shape[: -ndims[0]]
            inputs = []
            for x, n in zip(margs, ndims):
                assert (
                    x.shape[:-n] == preshape
                ), f"all args must have the same preceding dimensions, but got {x.shape[:-n]} and {preshape}"
                xs = x.reshape(-1, *x.shape[-n:])
                inputs.append(xs)

            # determine the shape of the outputs
            y = func(*params, *[xs[0] for xs in inputs], **kwargs)
            nested = False
            if type(y) == np.ndarray:
                y_shape = list(y.shape)
                ys = np.empty([inputs[0].shape[0]] + y_shape, y.dtype)
                ys[0] = y
            elif type(y) == tuple:
                all([type(elem) == np.ndarray for elem in y])
                nested = True
                elem_shapes = []
                for t, elem in enumerate(y):
                    if type(elem) == np.ndarray:
                        elem_shapes.append(list(elem.shape))
                    else:
                        elem_shapes.append([])
                ys = tuple(
                    np.empty(
                        [inputs[0].shape[0]] + elem_shapes[t], np.array(elem).dtype
                    )
                    for t, elem in enumerate(y)
                )
                for t, elem in enumerate(y):
                    ys[t][0] = elem
            else:
                y_shape = []
                ys = [None] * inputs[0].shape[0]
                ys[0] = y

            # Call the function for each of the elements in the inputs.
            for i in range(1, inputs[0].shape[0]):
                ret = func(*params, *[xs[i] for xs in inputs], **kwargs)
                if nested:
                    for t, elem in enumerate(ret):
                        ys[t][i] = elem
                else:
                    ys[i] = ret

            # reshape the outputs to match the preceding dimensions of the inputs
            if nested:
                out = [None] * len(ys)
                for t, elem in enumerate(ys):
                    out[t] = np.reshape(
                        ys[t], list(margs[0].shape[: -ndims[0]]) + elem_shapes[t]
                    )
                out = tuple(out)
            else:
                out = np.reshape(ys, list(margs[0].shape[: -ndims[0]]) + y_shape)
            return out

        return wrapper

    return decorator
