from datetime import datetime
from typing import List
import logging
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from typing import Optional
import seaborn as sns

from . import heatmap_utils
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


def as_float32(image: np.ndarray) -> np.ndarray:
    """Convert the image to float32.

    Args:
        image (np.ndarray): the image to convert.

    Returns:
        np.ndarray: the converted image.
    """
    if image.dtype in [np.float16, np.float32, np.float64]:
        image = image.astype(np.float32)
    elif image.dtype == bool:
        image = image.astype(np.float32)
    elif image.dtype != np.uint8:
        logging.warning(f"Unknown image type {image.dtype}. Converting to float32.")
        image = image.astype(np.float32)
    else:
        image = image.astype(np.float32) / 255
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
        raise ValueError(f"bad input shape: {x.shape}")


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
    color: List[int] = [255, 0, 0],
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


def blend_heatmaps(
    image: np.ndarray,
    heatmaps: np.ndarray,
    alpha: float = 0.5,
    seed: Optional[int] = 0,
    palette: str = "Spectral",
) -> np.ndarray:
    """Visualize heatmaps on top of an image.

    Args:
        image (np.ndarray): (H, W, C) Image to visualize heatmaps on top of. If float, in range [0, 1], will be converted to uint8.
        heatmaps (np.ndarray): (H, W, num_heatmaps) Heatmaps to visualize. If float, in range [0, 1], will be converted to uint8.
        alpha (float, optional): Alpha value for the heatmaps. Defaults to 0.5.

    Returns:
        np.ndarray: Image with heatmaps visualized, as a uint8 image.

    """
    image = as_float32(image)
    if image.min() < 0 or image.max() > 1:
        log.error(
            f"image min: {image.min()}, max: {image.max()}. Expected in range [0, 1]"
        )
    num_heatmaps = heatmaps.shape[2]
    colors = np.array(sns.color_palette(palette, num_heatmaps))
    if seed is not None:
        np.random.seed(seed)
    colors = colors[np.random.permutation(colors.shape[0])]  # (num_heatmaps, 3)
    combined_heatmap = (
        heatmaps[:, :, :, None] * colors[None, None, :, :]
    )  # (H, W, num_heatmaps, 3)
    combined_heatmap = np.max(combined_heatmap, axis=2)  # (H, W, 3)
    hmin = np.min(combined_heatmap)
    hmax = np.max(combined_heatmap)
    combined_heatmap = (combined_heatmap - hmin) / (hmax - hmin + 1e-8)

    # Blend the heatmaps with the image using alpha blending
    image = image * (1 - alpha) + combined_heatmap * alpha
    return (image * 255).astype(np.uint8)


def draw_masks(
    image: np.ndarray,
    masks: np.ndarray,
    alpha: float = 0.3,
    palette: str = "Spectral",
    threshold: float = 0.5,
    seed: Optional[int] = 0,
) -> np.ndarray:
    """Draw contours of masks on an image.

    Args:
        image (np.ndarray): the image to draw on.
        masks (np.ndarray): the masks to draw. [H, W, num_masks] array of masks.
    """

    image = as_float32(image)
    masks = masks.transpose(2, 0, 1)
    colors = np.array(sns.color_palette(palette, masks.shape[0]))
    if seed is not None:
        np.random.seed(seed)
    colors = colors[np.random.permutation(colors.shape[0])]
    image *= 1 - alpha
    for i, mask in enumerate(masks):
        bool_mask = mask > threshold
        image[bool_mask] = colors[i] * alpha + image[bool_mask] * (1 - alpha)

        contours, _ = cv2.findContours(
            bool_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image = as_float32(
            cv2.drawContours(
                as_uint8(image), contours, -1, (255 * colors[i]).tolist(), 1
            )
        )
    return (image * 255).astype(np.uint8)
