from datetime import datetime
import logging
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from typing import Optional

from .. import geo

log = logging.getLogger(__name__)


def as_uint8(image: np.ndarray) -> np.ndarray:
    """Convert the image to uint8.

    Args:
        image (np.ndarray): the image to convert.

    Returns:
        np.ndarray: the converted image.
    """
    if image.dtype in [np.float16, np.float32, np.float64]:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    elif image.dtype == bool:
        image = image.astype(np.uint8) * 255
    elif image.dtype != np.uint8:
        logging.warning(f"Unknown image type {image.dtype}. Converting to uint8.")
        image = image.astype(np.uint8)
    return image


def save(path: Path, image: np.ndarray, mkdir: bool = True) -> Path:
    """Save the given image using PIL.

    Args:
        path (Path): the path to write the image to. Also determines the type.
        image (np.ndarray): the image, in [C, H, W] or [H, W, C] order. (If the former, transposes).
            If in float32, assumed to be a float image. Converted to uint8 before saving.
    """
    path = Path(path)
    if not path.parent.exists() and mkdir:
        path.parent.mkdir(parents=True)

    if len(image.shape) == 3 and image.shape[0] in [3, 4]:
        image = image.transpose(1, 2, 0)

    image = as_uint8(image)

    Image.fromarray(image).save(str(path))
    return path


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


def ensure_cdim(x: np.ndarray, c: int = 3) -> np.ndarray:
    if x.ndim == 2:
        x = x[:, :, np.newaxis]
    elif x.ndim == 3:
        pass
    else:
        raise ValueError(f"bad input ndim: {x.shape}")

    if x.shape[2] < c:
        return np.concatenate([x] * c, axis=2)
    elif x.shape[2] == c:
        return x
    else:
        raise ValueError


def draw_line(
    image: np.ndarray, line: geo.Line2D, color: tuple = (255, 0, 0), thickness: int = 2
) -> np.ndarray:
    """Draw a line on an image.

    Args:
        image (np.ndarray): the image to draw on.
        line (geo.Line2D): the line to draw.
        color (tuple, optional): the color to draw the line in. Defaults to (255, 0, 0).
        thickness (int, optional): the thickness of the line. Defaults to 2.


    """
    line = geo.line(line)
    s = geo.p(0, -(line.a * 0 + line.c) / line.b)
    t = geo.p(image.shape[1], -(line.a * image.shape[1] + line.c) / line.b)
    image = ensure_cdim(as_uint8(image)).copy()
    image = cv2.line(
        image, (int(s.x), int(s.y)), (int(t.x), int(t.y)), color, thickness
    )
    return image


def draw_segment(
    image: np.ndarray,
    segment: geo.Segment2D,
    color=[255, 0, 0],
    thickness: int = 2,
    radius: int = 5,
) -> np.ndarray:
    """Draw a segment on an image."""

    color = np.array(color)
    if np.any(color < 1):
        color = color * 255
    color = color.astype(int)[:3].tolist()

    image = ensure_cdim(as_uint8(image)).copy()
    image = cv2.line(
        image,
        (int(segment.p.x), int(segment.p.y)),
        (int(segment.q.x), int(segment.q.y)),
        color,
        thickness,
    )
    if radius > 0:
        image = cv2.circle(
            image, (int(segment.p.x), int(segment.p.y)), radius, color, -1
        )
        image = cv2.circle(
            image, (int(segment.q.x), int(segment.q.y)), radius, color, -1
        )
    return image


def draw_circles(
    image: np.ndarray,
    circles: np.ndarray,
    color: list[int] = [255, 0, 0],
    thickness: int = 2,
    radius: Optional[int] = None,
) -> np.ndarray:
    """Draw circles on an image.

    Args:
        image (np.ndarray): the image to draw on.
        circles (np.ndarray): the circles to draw. [N, 3] array of [x, y, r] coordinates.

    """
    color = np.array(color)
    if np.any(color < 1):
        color = color * 255
    color = color.astype(int)[:3].tolist()

    circles = np.array(circles)
    image = ensure_cdim(as_uint8(image)).copy()
    for circle in circles:
        if circles.shape[1] == 3:
            x, y, r = circle
        elif circles.shape[1] == 2:
            x, y = circle
            r = radius if radius is not None else 15
        else:
            raise ValueError(f"bad circles shape: {circles.shape}")
        if radius is not None:
            r = radius
        image = cv2.circle(image, (int(x), int(y)), int(r), color, thickness)
    return image
