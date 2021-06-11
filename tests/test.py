#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import deepdrr
from deepdrr import geo
from deepdrr.utils import testing
from PIL import Image


class TestSingleVolume:
    output_dir = Path.cwd() / 'output'
    output_dir.mkdir(exist_ok=True)

    def test_simple(self):
        file_path = testing.download_sampledata("CT-chest")
        volume = deepdrr.Volume.from_nrrd(file_path)
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
        with deepdrr.Projector(
            volume=volume,
            carm=carm,
            step=0.1,  # stepsize along projection ray, measured in voxels
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
        Image.fromarray(image).save(self.output_dir / 'test_simple.png')

if __name__ == "__main__":
    TestSingleVolume().test_simple()