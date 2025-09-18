import logging

import deepdrr
from deepdrr import geo
from PIL import Image
from deepdrr.utils import test_utils
import numpy as np
from pathlib import Path


def test_multivolume():
    file_paths = [test_utils.download_sampledata(
        "CT-chest"), test_utils.download_sampledata("CT-chest")]
    volumes = [deepdrr.Volume.from_nrrd(file_path) for file_path in file_paths]
    volumes[0].rotate(geo.Rotation.from_euler(
        "x", -90, degrees=True), center=volumes[0].center_in_world)
    volumes[1].translate([0, 200, 0])
    carm = deepdrr.MobileCArm(isocenter=volumes[0].center_in_world)
    with deepdrr.Projector(
        volume=volumes,
        priorities=[1, 0],  # equivalent to priorities=None
        carm=carm,
        step=0.1,
        mode="linear",
        max_block_index=200,
        spectrum="90KV_AL40",
        photon_count=100000,
        add_scatter=False,
        threads=8,
        neglog=True,
    ) as projector:
        image = projector.project()

    image = (image * 255).astype(np.uint8)

    try:
        d = Path(__file__).resolve().parent
        truth = d / "reference"
        output_dir = d / "output"
        output_dir.mkdir(exist_ok=True)
        Image.fromarray(image).save(output_dir / "test_multivolume.png")
    except e:
        print(e)


if __name__ == "__main__":
    logging.getLogger("deepdrr").setLevel(logging.DEBUG)
    test_multivolume()
