"""Util functions for automated tests."""

import logging
from pathlib import Path
from torchvision.datasets.utils import download_url

logger = logging.getLogger(__name__)

_sampledata = {
    'CT-chest': 'http://www.slicer.org/w/img_auth.php/3/31/CT-chest.nrrd',
}

def download_sampledata(name: str = 'CT-chest', root: str = "~/datasets/") -> Path:
    """Download the sample volumes for testing, if they are not already present.

    Args:
        name (str, optional): The name of the volume, used as a key to the public domain downloadable data. Defaults to 'CT-chest'.
        root (str, optional): The root where datasets are kept. A new dataset directory will be created in `root`, called `DeepDRR_SampleData`. Defaults to "~/datasets/".

    Returns:
        Path: The path to the downloaded file.
    """
    if name not in _sampledata:
        logger.error(f"unrecognized sample data name: {name}. Options are:\n{list(_sampledata.keys())}")

    root = Path(root).expanduser()
    if not root.exists():
        root.mkdir()

    dataset_dir = root / 'DeepDRR_SampleData'
    if not dataset_dir.exists():
        dataset_dir.mkdir()

    fpath = dataset_dir / f"{name}.nrrd"
    if fpath.exists():
        return fpath

    logger.info(f"Downloading sample volume to {fpath}")

    download_url(_sampledata[name], root=dataset_dir, filename=fpath.name)

    logger.info("Done.")
    return fpath