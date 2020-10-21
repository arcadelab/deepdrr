import numpy as np
import PIL.Image as Image
from datetime import datetime
from os import path
import pickle


def image_saver(images, prefix, path):
    for i in range(0, images.shape[0]):
        image_pil = Image.fromarray(images[i, :, :])
        image_pil.save(path + "\\" + prefix + str(i).zfill(5) + ".tiff")
    return True


def param_saver(thetas, phis, proj_mats, camera, origin, photons, spectrum, prefix, save_path):
    i0 = np.sum(spectrum[:, 0] * (spectrum[:, 1] / np.sum(spectrum[:, 1]))) / 1000
    data = {"date": datetime.now(), "thetas": thetas, "phis": phis, "proj_mats": proj_mats, "camera": camera, "origin": origin, "photons": photons, "spectrum": spectrum, "I0": i0}
    with open(path.join(save_path, prefix + '.pickle'), 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return True


def one_hot(x: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """One-hot encode the vector x.

    Args:
        x (np.ndarray): n-dim array x.
        num_classes (Optional[int]): number of classes. Uses maximum label if not provided.

    Returns:
        np.ndarray: one-hot encoded labels with n + 1 axes.
    """
    if num_classes is None:
        num_classes = x.max()
        
    return x[..., np.newaxis] == np.arange(num_classes + 1)