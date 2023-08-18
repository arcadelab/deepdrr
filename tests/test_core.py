#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils
from PIL import Image
import pytest
import copy
import time

import pyvista as pv
import logging
import pyrender
from deepdrr.pyrenderdrr.material import DRRMaterial
from deepdrr.utils.mesh_utils import *

def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames]
                   for funcargs in funcarglist]
    )


class TestSingleVolume:
    d = Path(__file__).resolve().parent
    truth = d / "reference"
    output_dir = d / "output"
    output_dir.mkdir(exist_ok=True)
    file_path = test_utils.download_sampledata("CT-chest")

    params = {
        "test_simple": [dict()],
        "test_collected_energy": [dict()],
        "test_translate": [
            dict(t=[0, 0, 0]),
            dict(t=[100, 0, 0]),
            dict(t=[0, 100, 0]),
            dict(t=[0, 0, 100]),
        ],
        "test_rotate_x": [dict(x=0), dict(x=30), dict(x=45), dict(x=90), dict(x=180)],
        "test_angle": [dict(alpha=0, beta=90)],
    }

    def load_volume(self):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        volume.rotate(Rotation.from_euler("x", -90, degrees=True))
        return volume

    def project(self, volume, carm, name, verify=True, **kwargs):
        # if verify:
        #     try: 
        #         truth_img = np.array(Image.open(self.truth / name))
        #     except FileNotFoundError:
        #         print(f"Truth image not found: {self.truth / name}")
        #         # pytest.skip("Truth image not found")
        #         pytest.fail("Truth image not found")

        projector = deepdrr.Projector(
            volume=volume,
            carm=carm,
            step=0.01,  # stepsize along projection ray, measured in voxels
            mode="linear",
            max_block_index=65535,
            spectrum="90KV_AL40",
            photon_count=100000,
            scatter_num=0,
            threads=8,
            neglog=True,
            **kwargs
        )

        with projector:
            image = projector.project()

        image_256 = (image * 255).astype(np.uint8)
        Image.fromarray(image_256).save(self.output_dir / name)

        if verify:
            try: 
                truth_img = np.array(Image.open(self.truth / name))
            except FileNotFoundError:
                print(f"Truth image not found: {self.truth / name}")
                pytest.skip("Truth image not found")
                # pytest.fail("Truth image not found")
            diff_im = image_256.astype(np.float32) - truth_img.astype(np.float32)
            from matplotlib import pyplot as plt
            plt.imshow(diff_im, cmap="viridis")
            plt.colorbar()
            plt.savefig(self.output_dir / f"diff_{name}")

            # Image.fromarray(np.abs(image_256.astype(np.float32) - truth_img.astype(np.float32))).save(self.output_dir / f"diff_{name}")


        # with projector:
        #     from timer_util import FPS
        #     start_time = time.time()
        #     fps = FPS()
        #     while True:
        #         for i in range(100):
        #             image = projector.project()
        #             if fps_count := fps():
        #                 print(f"FPS2 {fps_count}")
        #         if time.time() - start_time > 4:
        #             break

        if verify:
            assert np.allclose(image_256, truth_img, atol=1)
            print(f"Test {name} passed")


        return image

    def test_simple(self):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
        self.project(volume, carm, "test_simple.png")

    def test_collected_energy(self):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
        self.project(volume, carm, "test_collected_energy.png", collected_energy=True)

    def test_translate(self, t):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
        volume.translate(t)
        self.project(
            volume, carm, f"test_translate_{int(t[0])}_{int(t[1])}_{int(t[2])}.png"
        )

    def test_rotate_x(self, x):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
        volume.rotate(Rotation.from_euler(
            "x", x, degrees=True), volume.center_in_world)
        self.project(volume, carm, f"test_rotate_x={int(x)}.png")

    def test_angle(self, alpha, beta):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
        carm.move_to(alpha=alpha, beta=beta, degrees=True)
        self.project(
            volume, carm, f"test_angle_alpha={int(alpha)}_beta={int(beta)}.png"
        )


if __name__ == "__main__":
    logging.getLogger("deepdrr").setLevel(logging.WARNING)
    # set projector log level to debug
    logging.basicConfig(level=logging.WARNING)
    test = TestSingleVolume()
    # test.test_layer_depth()
    test.gen_threads()
    # test.test_mesh()
    # volume = test.load_volume()
    # carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
    # test.project(volume, carm, "test.png")
