from typing import List, Optional, Any, Tuple
import os
import logging
import numpy as np
from pathlib import Path
from torchvision.datasets.utils import download_url, extract_archive
import urllib
import subprocess
import json

log = logging.getLogger(__name__)


def deepdrr_data_dir() -> Path:
    """Get the data directory for DeepDRR.

    The data directory is determined by the environment variable `DEEPDRR_DATA_DIR` if it exists.
    Otherwise, it is `~/datasets/DeepDRR`. If the directory does not exist, it is created.

    Returns:
        Path: The data directory.
    """
    if os.environ.get("DEEPDRR_DATA_DIR") is not None:
        root = Path(os.environ.get("DEEPDRR_DATA_DIR")).expanduser()
    else:
        root = Path.home() / "datasets" / "DeepDRR_DATA"

    if not root.exists():
        root.mkdir(parents=True)

    return root


def download(
    url: str,
    filename: Optional[str] = None,
    root: Optional[str] = None,
    md5: Optional[str] = None,
    extract_name: Optional[str] = None,
) -> Path:
    """Download a data file and place it in root.

    Args:
        url (str): The download link.
        filename (str, optional): The name the save the file under. If None, uses the name from the URL. Defaults to None.
        root (str, optional): The directory to place downloaded data in. Can be overriden by setting the environment variable DEEPDRR_DATA_DIR. Defaults to "~/datasets/DeepDRR_Data".
        md5 (str, optional): MD5 checksum of the download. Defaults to None.
        extract_name: If not None, extract the downloaded file to `root / extract_name`.

    Returns:
        Path: The path of the downloaded file, or the extracted directory.
    """
    if root is None:
        root = deepdrr_data_dir()
    else:
        root = Path(root)

    if filename is None:
        filename = os.path.basename(url)

    try:
        download_url(url, root, filename=filename, md5=md5)
    except urllib.error.HTTPError:
        log.warning(f"Pretty download failed. Attempting with wget...")
        subprocess.call(["wget", "-O", str(root / filename), url])
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Download failed. Try installing wget. This is probably because you are on windows."
        )
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

    path = root / filename
    if extract_name is not None:
        extract_archive(path, root, remove_finished=True)
        path = root / extract_name

    return path


def jsonable(obj: Any):
    """Convert obj to a JSON-ready container or object.
    Args:
        obj ([type]):
    """
    if obj is None:
        return "null"
    elif isinstance(obj, (str, float, int, complex)):
        return obj
    elif isinstance(obj, Path):
        return str(obj.resolve())
    elif isinstance(obj, (list, tuple)):
        return type(obj)(map(jsonable, obj))
    elif isinstance(obj, dict):
        return dict(jsonable(list(obj.items())))
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "__array__"):
        return np.array(obj).tolist()
    else:
        raise ValueError(f"Unknown type for JSON: {type(obj)}")


def save_json(path: str, obj: Any):
    obj = jsonable(obj)
    with open(path, "w") as file:
        json.dump(obj, file, indent=4, sort_keys=True)


def load_json(path: str) -> Any:
    with open(path, "r") as file:
        out = json.load(file)
    return out


def save_fcsv(
    path: str,
    points: np.ndarray,
    names: Optional[List[str]] = None,
    coordinate_system: str = "LPS",
):
    """Save a fcsv file.

    Args:
        path (str): The path to save the file to.
        points (np.ndarray): The points to save. Shape: (N, 3)
        names (List[str]): The names of the points. Shape: (N,)
    """
    if names is None:
        names = ["" for i in range(len(points))]
    assert points.shape[0] == len(names)
    assert coordinate_system in ["LPS", "RAS"]
    assert points.shape[1] == 3

    with open(path, "w") as file:
        file.write("# Markups fiducial file version = 5.0\n")
        file.write(f"# CoordinateSystem = {coordinate_system}\n")
        file.write(
            f"# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
        )
        lines = []
        for i, (point, name) in enumerate(zip(points, names)):
            line = f"{i},{point[0]},{point[1]},{point[2]},0,0,0,1,1,1,1,{name},,\n"
            lines.append(line)

        file.writelines(lines)


def load_fcsv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a fcsv file.

    Args:
        path (str): The path to the fcsv file.

    Returns:
        np.ndarray: The points. Shape: (N, 3)
        np.ndarray: The names of the points. Shape: (N,)
    """
    with open(path, "r") as file:
        lines = file.readlines()
    points = []
    names = []
    for line in lines:
        if line.startswith("#"):
            continue
        point = line.split(",")[1:4]
        point = [float(p) for p in point]
        points.append(point)
        name = line.split(",")[11].strip()
        names.append(name)
    points = np.array(points)
    return points, names
