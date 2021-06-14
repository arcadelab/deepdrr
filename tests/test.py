#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import deepdrr
from deepdrr import geo
from deepdrr.utils import testing
from PIL import Image


def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )

class TestSingleVolume:
    output_dir = Path.cwd() / 'output'
    output_dir.mkdir(exist_ok=True)
    file_path = testing.download_sampledata("CT-chest")

    params = {
        "test_simple": [dict()],
        "test_translation": [dict(t=[0, 0, 100]), dict(t=[200, 0, 0])],
    }

    def project(self, volume, carm, name):
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
        Image.fromarray(image).save(self.output_dir / name)

    def test_simple(self):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
        self.project(volume, carm, 'test_simple.png')

    def test_translation(self, t):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        volume.translate(t)
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
        self.project(volume, carm, f'test_translation_{int(t[0])}_{int(t[1])}_{int(t[2])}.png')


if __name__ == "__main__":
    TestSingleVolume().test_simple()