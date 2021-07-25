"""Util functions for automated tests."""

import logging
from pathlib import Path
import zipfile

from . import data_utils

logger = logging.getLogger(__name__)

_sampledata = {
    'CT-chest': 'http://www.slicer.org/w/img_auth.php/3/31/CT-chest.nrrd',
    'CTPelvic1K_sample': "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/EfynPcmVdVJHs_eoTTXPIFEB71mWSrZCERGxJKNQNovqwA?e=OPkyJc&download=1"
}

_filenames = {
    'CT-chest': 'CT-chest.nrrd',
    'CTPelvic1K_sample': 'CTPelvic1K_dataset6_CLINIC_0001.zip',
}


def download_sampledata(name: str = 'CT-chest', **kwargs) -> Path:
    """Download the given sample data for testing.

    Options include:
    * `"CT-chest"`: a NRRD file containing a torso.
    * `"CTPelvic1K"`: a CT scan of the pelvis with a right superior pubic ramus fracture, 
        selected from the `CTPelvik1K dataset <https://github.com/ICT-MIRACLE-lab/CTPelvic1K>`_.
        This downloads the CT, pelvis segmentation, and KWire trajectory annotations and unzips them,
        returning the directory where they are located.

    Args:
        name (str, optional): The name of the volume, used as a key to the public domain downloadable data. Defaults to 'CT-chest'.

    Returns:
        Path: The path to the downloaded file or directory (if unzipped).
    """
    if name not in _sampledata:
        logger.error(
            f"unrecognized sample data name: {name}. Options are:\n{list(_sampledata.keys())}")

    filename = _filenames[name]

    path = data_utils.download(
        _sampledata[name], filename=filename, **kwargs)

    if path.suffix == '.zip':
        if not (path.parent / path.stem).exists():
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(path.parent)
        path = path.parent / path.stem

    return path


def get_output_dir() -> Path:
    output_dir = Path.cwd() / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir
