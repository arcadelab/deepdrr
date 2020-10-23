from typing import Optional

import numpy as np
import PIL.Image as Image
from datetime import datetime
from pathlib import Path
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