"""Utils for dealing with DICOM images."""

from typing import Optional, Tuple, List
from pathlib import Path
from datetime import datetime
import numpy as np
import pydicom
import logging

log = logging.getLogger(__name__)


def get_time(path: str) -> Optional[datetime]:
    """Parse a dicom path to get the acquisition time.

    Does not actually open the image.

    """
    p = Path(path)
    
    try:
        t = datetime.strptime(p.stem.split("_")[1][:-3], r"%Y%m%d%H%M%S")
    except ValueError:
        log.debug(f"Could not parse time from {path}")
        return None
    except IndexError:
        log.debug(f"Could not parse time from {path}")
        return None

    return t


def find_dicom(
    image_dir: Path, device_time: datetime, max_difference: Optional[int] = 2
) -> Optional[Path]:
    """Get the image with the given acquisition time.

    # TODO: parallelize this to work for multiple provided device times, for efficiency.

    Args:
        image_dir: Path to the directory containing the images. All subtrees are searched, so make sure it's not large.
        device_time (str): Datetime object for the acquisition time.
        max_difference (int, optional): Maximum difference in seconds between the given timestamp and the closest image. Defaults to 2.

    Returns:
        Path: Path to the image.
    """

    # Get the image path
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise ValueError(f"{image_dir} is not a directory.")

    image_paths: List[Path] = []
    device_times: List[datetime] = []
    for p in image_dir.glob("**/*.dcm"):
        t = get_time(p)
        if t is None:
            continue

        image_paths.append(p)
        device_times.append(t)

    if len(image_paths) == 0 or len(device_times) == 0:
        return None

    assert len(image_paths) == len(device_times)
    image_diffs = [abs((device_time - t).total_seconds()) for t in device_times]
    image_path_idx = np.argmin(image_diffs)
    image_path = image_paths[image_path_idx]
    dt = device_times[image_path_idx]
    if max_difference is not None and image_diffs[image_path_idx] > max_difference:
        return None

    return image_path


def read_image(path: str) -> np.ndarray:
    """Load a DICOM image.

    Args:
        path: Path to the DICOM file.

    Returns:
        The image as a float32 numpy array, in range [0, 1].
    """
    image = pydicom.dcmread(path).pixel_array
    image = image.astype(np.float32)
    image = image / (2**16 - 1)
    return image
