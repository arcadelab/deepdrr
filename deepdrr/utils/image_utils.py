import numpy as np
from PIL import Image
from pathlib import Path


def save(path: str, image: np.ndarray) -> None:
    """Save the given image using PIL.

    Args:
        path (str): the path to write the image to. Also determines the type.
        image (np.ndarray): the image. If in float32, assumed to be a float image. Converted to uint8 before saving.
    """

    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def image_saver(images: np.ndarray, prefix: str, path: str) -> bool:
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
