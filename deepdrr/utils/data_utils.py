from typing import Optional, Any
import os
import logging
import numpy as np
from pathlib import Path
from torchvision.datasets.utils import download_url, extract_archive
import urllib
import subprocess
import json

logger = logging.getLogger(__name__)


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
    if root is None and os.environ.get("DEEPDRR_DATA_DIR") is not None:
        root = os.environ["DEEPDRR_DATA_DIR"]
    elif root is None:
        root = "~/datasets/DeepDRR_Data"

    root = Path(root).expanduser()
    if not root.exists():
        root.mkdir(parents=True)

    if filename is None:
        filename = os.path.basename(url)

    try:
        download_url(url, root, filename=filename, md5=md5)
    except urllib.error.HTTPError:
        logger.warning(f"Pretty download failed. Attempting with wget...")
        subprocess.call(["wget", "-O", str(root / filename), url])
    except FileNotFoundError:
        logger.warning(
            f"Download failed. Try installing wget. This is probably because you are on windows."
        )
        exit()
    except Exception as e:
        logger.error(f"Download failed: {e}")
        exit()

    path = root / filename
    if extract_name is not None:
        extract_archive(path, root / extract_name, remove_finished=True)
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


def load_json(path: str):
    with open(path, "r") as file:
        out = json.load(file)
    return out
