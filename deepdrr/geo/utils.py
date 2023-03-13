from typing import List, Union
import numpy as np
import logging
import traceback

log = logging.getLogger(__name__)


def _array(x: Union[List[np.ndarray], List[float]]) -> np.ndarray:
    # TODO: this is a little sketchy
    if len(x) == 1:
        return np.array(x[0])
    else:
        if isinstance(x[0], np.ndarray):
            log.warning(f"got unusual args for array: {x}")
            traceback.print_stack()
        return np.array(x)


def _to_homogeneous(x: np.ndarray, is_point: bool = True) -> np.ndarray:
    """Convert an array to homogeneous points or vectors.

    Args:
        x (np.ndarray): array with objects on the last axis.
        is_point (bool, optional): if True, the array represents a point, otherwise it represents a vector. Defaults to True.

    Returns:
        np.ndarray: array containing the homogeneous point/vector(s).
    """
    if is_point:
        return np.concatenate([x, np.ones_like(x[..., -1:])], axis=-1)
    else:
        return np.concatenate([x, np.zeros_like(x[..., -1:])], axis=-1)


def _from_homogeneous(x: np.ndarray, is_point: bool = True) -> np.ndarray:
    """Convert array containing homogeneous data to raw form.

    Args:
        x (np.ndarray): array containing homogenous
        is_point (bool, optional): whether the objects are points (true) or vectors (False). Defaults to True.

    Returns:
        np.ndarray: the raw data representing the point/vector(s).
    """
    if is_point:
        return (x / x[..., -1:])[..., :-1]
    else:
        assert np.all(np.isclose(x[..., -1], 0)), f"not a homogeneous vector: {x}"
        return x[..., :-1]
