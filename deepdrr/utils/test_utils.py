"""Util functions for automated tests."""

import logging
from pathlib import Path

from . import data_utils

logger = logging.getLogger(__name__)

_sampledata = {
    'CT-chest': 'http://www.slicer.org/w/img_auth.php/3/31/CT-chest.nrrd',
}


def download_sampledata(name: str = 'CT-chest', **kwargs) -> Path:
    """Download the given sample volume for testing.

    Args:
        name (str, optional): The name of the volume, used as a key to the public domain downloadable data. Defaults to 'CT-chest'.

    Returns:
        Path: The path to the downloaded file.
    """
    if name not in _sampledata:
        logger.error(
            f"unrecognized sample data name: {name}. Options are:\n{list(_sampledata.keys())}")

    filename = f"{name}.nrrd"
    return data_utils.download(_sampledata[name], filename=filename, **kwargs)
