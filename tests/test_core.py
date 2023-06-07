#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils
from PIL import Image

import pyvista as pv

def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames]
                   for funcargs in funcarglist]
    )


class TestSingleVolume:
    truth = Path.cwd() / "reference"
    output_dir = Path.cwd() / "output"
    output_dir.mkdir(exist_ok=True)
    file_path = test_utils.download_sampledata("CT-chest")

    params = {
        "test_simple": [dict()],
        "test_mesh": [dict()],
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

    def project(self, volume, carm, name):
        # set projector log level to debug
        import logging
        logging.basicConfig(level=logging.DEBUG)

        with deepdrr.Projector(
            volume=volume,
            carm=carm,
            step=0.1,  # stepsize along projection ray, measured in voxels
            mode="linear",
            max_block_index=200,
            spectrum="90KV_AL40",
            photon_count=100000,
            scatter_num=0,
            threads=8,
            neglog=True,
        ) as projector:
            image = projector.project()
            # from timer_util import FPS
            # fps = FPS()
            # for i in range(1000):
            #     image = projector.project()
            #     if fps_count := fps():
            #         print(f"FPS2 {fps_count}")

        image = (image * 255).astype(np.uint8)
        Image.fromarray(image).save(self.output_dir / name)
        try: 
            truth_img = np.array(Image.open(self.truth / name))
            assert np.allclose(image, truth_img, atol=1)
        except FileNotFoundError:
            print(f"Truth image not found: {self.truth / name}")
        return image

    def test_simple(self):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
        self.project(volume, carm, "test_simple.png")

    def test_mesh(self):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        # load 10cmcube.stl from resources folder
        # stl = pv.read("tests/resources/10cmrighttri.stl")
        # stl = pv.read("tests/resources/10cmcube.stl")
        stl = pv.read("tests/resources/suzanne.stl")
        stl.scale([200]*3, inplace=True)
        stl.translate([0, -200, 0], inplace=True)
        # stl = pv.read("tests/resources/suzanne.stl")
        # stl.scale([200, 3000, 200], inplace=True)
        # stl.translate([0, -250, 0], inplace=True)
        morph_targets = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [-1, 0, 1],
            [0, 0, 1],
                                  ]).reshape(1, -1, 3)
        # scale from m to mm
        # mesh = deepdrr.Mesh("titanium", 7, stl, world_from_anatomical=geo.FrameTransform.from_rotation(geo.Rotation.from_euler("y", 90, degrees=True)))
        # mesh = deepdrr.Mesh("air", 0, stl, morph_targets=morph_targets, world_from_anatomical=geo.FrameTransform.from_rotation(geo.Rotation.from_euler("x", 90, degrees=True)))
        mesh = deepdrr.Mesh("titanium", 1, stl, world_from_anatomical=geo.FrameTransform.from_rotation(geo.Rotation.from_euler("x", 90, degrees=True)))
        # mesh = deepdrr.Mesh("polyethylene", 1.05, stl)
        mesh.morph_weights = np.array([-10])
        
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world, sensor_width=300, sensor_height=200, pixel_size=0.6)
        self.project([volume, mesh], carm, "test_mesh.png")


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
    test = TestSingleVolume()
    test.test_mesh()
    # volume = test.load_volume()
    # carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
    # test.project(volume, carm, "test.png")
