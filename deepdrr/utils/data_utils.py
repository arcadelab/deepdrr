from typing import Optional
import os
import logging
from pathlib import Path
from torchvision.datasets.utils import download_url
import urllib
import subprocess

logger = logging.getLogger(__name__)


def download(url: str, filename: Optional[str] = None, root: str = "~/datasets/DeepDRR_Data", md5: Optional[str] = None) -> Path:
    """Download a data file and place it in the default root for DeepDRR.

    Args:
        url (str): The download link.
        filename (str, optional): The name the save the file under. If None, uses the name from the URL. Defaults to None.
        root (str, optional): The directory to place downloaded data in. Can be overriden by setting the environment variable DEEPDRR_DATA_DIR. Defaults to "~/datasets/DeepDRR_Data".
        md5 (str, optional): MD5 checksum of the download. Defaults to None.

    Returns:
        Path: The path of the downloaded file.
    """
    if os.environ.get("DEEPDRR_DATA_DIR") is not None:
        root = os.environ["DEEPDRR_DATA_DIR"]

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
    return root / filename
