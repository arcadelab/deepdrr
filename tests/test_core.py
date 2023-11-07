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
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence
from pathlib import Path
import io
from matplotlib import pyplot as plt

from deepdrr.utils.img_cmp import verify_image

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
    diff_dir = d / "diff"
    diff_dir.mkdir(exist_ok=True)
    params = {
        "test_simple": [dict()],
        "test_collected_energy": [dict()],
        "test_cube": [dict()],
        "test_mesh": [dict()],
        "test_mesh_mesh_sub": [dict()],
        "test_mesh_mesh_1": [dict()],
        "test_mesh_only": [dict()],
        "test_multi_projector": [dict()],
        "test_layer_depth": [dict()],
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
            **kwargs
        )

        with projector:
            image = projector.project()

        # # make fig with colorbar
        # plt.figure()
        # plt.imshow(image, cmap="viridis")
        # plt.colorbar()
        # plt.savefig(self.output_dir / f"plt_{name}")

        image_256 = (image * 255).astype(np.uint8)

        with verify_image(name, actual_dir=self.output_dir, expected_dir=self.truth, diff_dir=self.diff_dir):
            Image.fromarray(image_256).save(self.output_dir / name)

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

        # if verify:



        return image


    def test_simple(self):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
        self.project(volume, carm, "test_simple.png")

    def test_collected_energy(self):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
        self.project(volume, carm, "test_collected_energy.png", collected_energy=True)

    def test_mesh(self):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        
        # load 10cmcube.stl from resources folder
        # stl = pv.read("tests/resources/10cmrighttri.stl")
        stl3 = pv.read("tests/resources/10cmcube.stl")
        stl3.scale([100, 100, 100], inplace=True)
        stl3.rotate_z(60, inplace=True)
        # stl3.translate([0, 00, 0], inplace=True)

        stl2 = pv.read("tests/resources/10cmcube.stl")
        stl2.scale([200, 200, 200], inplace=True)

        # stl3_verts = np.array(stl2.points)
        # stl3_verts[:, 0] *= 1
        # stl3_verts[:, 1] *= 1
        # stl3_verts[:, 2] *= 1
        # stl3_faces = np.array(stl2.faces).reshape(-1, 4)
        # stl3_faces = stl3_faces[:, [0, 2, 1, 3]]
        # stl3_faces = stl3_faces.flatten()
        # stl2 = pv.PolyData(stl3_verts, stl3_faces)

        # stl2.translate([0, 30, 0], inplace=True)
        stl = pv.read("tests/resources/solenoidasm.stl")
        stl.scale([400/1000]*3, inplace=True)
        # stl = pv.read("tests/resources/suzanne.stl")
        # stl.scale([200]*3, inplace=True)
        # stl.translate([0, 0, 0], inplace=True)
        stl.rotate_y(60, inplace=True)
        stl.rotate_x(10, inplace=True)
        stl.rotate_z(80, inplace=True)
        stl.translate([40, -200, -0], inplace=True)
        # stl = pv.read("tests/resources/suzanne.stl")
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
        prim = pyrender.Mesh.from_trimesh(polydata_to_trimesh(stl), material=DRRMaterial("titanium", density=7, subtractive=False))
        # prim = pyrender.Mesh.from_trimesh(polydata_to_trimesh(stl), material=DRRMaterial("bone", density=2, subtractive=True))
        # mesh = deepdrr.Mesh(mesh=prim)
        mesh = deepdrr.Mesh(mesh=prim, world_from_anatomical=geo.FrameTransform.from_rotation(geo.Rotation.from_euler("x", 90, degrees=True)))
        # mesh = deepdrr.Mesh(mesh=prim, world_from_anatomical=geo.FrameTransform.from_rotation(geo.Rotation.from_euler("x", 90, degrees=True) * geo.Rotation.from_euler("y", 30, degrees=True)))

        # prim2 = deepdrr.Primitive("titanium", 2, stl2, subtractive=True)
        prim2 = trimesh_to_pyrender_mesh(polydata_to_trimesh(stl2), material=DRRMaterial("lung", density=2, subtractive=True, layer=1))
        mesh2 = deepdrr.Mesh(mesh=prim2, world_from_anatomical=geo.FrameTransform.from_translation([-30, 50, 200]))

        # prim3 = deepdrr.Primitive("titanium", 0, stl2, subtractive=True)
        prim3 = polydata_to_pyrender_mesh(stl2, material=DRRMaterial("titanium", density=0, subtractive=True, layer=1))
        mesh3 = deepdrr.Mesh(mesh=prim3, world_from_anatomical=geo.FrameTransform.from_translation([30, 20, -70]))
        # mesh = deepdrr.Mesh("polyethylene", 1.05, stl)
        # mesh.morph_weights = np.array([-10])
        
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world, sensor_width=300, sensor_height=200, pixel_size=0.6)
        # self.project([volume], carm, "test_mesh.png")
        # self.project([mesh, mesh2, mesh3], carm, "test_mesh.png")
        self.project([volume, mesh, mesh2, mesh3], carm, "test_mesh.png", verify=True, num_mesh_layers=32)
        # self.project([volume, mesh, mesh2, mesh3], carm, "test_mesh.png", verify=False, num_mesh_layers=64)

    
    def test_mesh_mesh_sub(self):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        
        # load 10cmcube.stl from resources folder
        # stl = pv.read("tests/resources/10cmrighttri.stl")
        stl3 = pv.read("tests/resources/10cmcube.stl")
        stl3.scale([100, 100, 100], inplace=True)
        stl3.rotate_z(60, inplace=True)
        # stl3.translate([0, 00, 0], inplace=True)

        stl2 = pv.read("tests/resources/10cmcube.stl")
        stl2.scale([200, 200, 200], inplace=True)

        # stl3_verts = np.array(stl2.points)
        # stl3_verts[:, 0] *= 1
        # stl3_verts[:, 1] *= 1
        # stl3_verts[:, 2] *= 1
        # stl3_faces = np.array(stl2.faces).reshape(-1, 4)
        # stl3_faces = stl3_faces[:, [0, 2, 1, 3]]
        # stl3_faces = stl3_faces.flatten()
        # stl2 = pv.PolyData(stl3_verts, stl3_faces)

        # stl2.translate([0, 30, 0], inplace=True)
        stl = pv.read("tests/resources/solenoidasm.stl")
        stl.scale([400/1000]*3, inplace=True)
        # stl = pv.read("tests/resources/suzanne.stl")
        # stl.scale([200]*3, inplace=True)
        # stl.translate([0, 0, 0], inplace=True)
        stl.rotate_y(60, inplace=True)
        stl.rotate_x(10, inplace=True)
        stl.rotate_z(80, inplace=True)
        stl.translate([40, -200, -0], inplace=True)
        # stl = pv.read("tests/resources/suzanne.stl")
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
        prim = pyrender.Mesh.from_trimesh(polydata_to_trimesh(stl), material=DRRMaterial("titanium", density=7, subtractive=False))
        # prim = pyrender.Mesh.from_trimesh(polydata_to_trimesh(stl), material=DRRMaterial("bone", density=2, subtractive=True))
        # mesh = deepdrr.Mesh(mesh=prim)
        mesh = deepdrr.Mesh(mesh=prim, world_from_anatomical=geo.FrameTransform.from_rotation(geo.Rotation.from_euler("x", 90, degrees=True)))
        # mesh = deepdrr.Mesh(mesh=prim, world_from_anatomical=geo.FrameTransform.from_rotation(geo.Rotation.from_euler("x", 90, degrees=True) * geo.Rotation.from_euler("y", 30, degrees=True)))

        # prim2 = deepdrr.Primitive("titanium", 2, stl2, subtractive=True)
        prim2 = trimesh_to_pyrender_mesh(polydata_to_trimesh(stl2), material=DRRMaterial("bone", density=0.1, subtractive=True, layer=1))
        mesh2 = deepdrr.Mesh(mesh=prim2, world_from_anatomical=geo.FrameTransform.from_translation([10, 30, 5]))

        # prim3 = deepdrr.Primitive("titanium", 0, stl2, subtractive=True)
        prim3 = polydata_to_pyrender_mesh(stl2, material=DRRMaterial("titanium", density=1, subtractive=False))
        mesh3 = deepdrr.Mesh(mesh=prim3, world_from_anatomical=geo.FrameTransform.from_translation([0, 20, 0]))
        # mesh = deepdrr.Mesh("polyethylene", 1.05, stl)
        # mesh.morph_weights = np.array([-10])
        
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world, sensor_width=300, sensor_height=200, pixel_size=0.6)
        # self.project([volume], carm, "test_mesh.png")
        # self.project([mesh, mesh2, mesh3], carm, "test_mesh.png")
        self.project([mesh2, mesh3], carm, "test_mesh_mesh_sub.png", verify=False, num_mesh_layers=32, neglog=True)

    


    def test_mesh_mesh_1(self):

        obj_names = [
            "body.stl",
            "Cube_001.stl",
            "Cube_002.stl",
            "Cube_003.stl",
            "Cube_004.stl",
        ]

        stls = [pv.read(f"tests/resources/meshmesh_1/{name}") for name in obj_names]

        

        # # print max and min of verts
        # for stl in stls:
        #     print(stl.points.max(axis=0))
        #     print(stl.points.min(axis=0))

        cube_meshes = []
        for i, stl in enumerate(stls[1:]):
            prim = trimesh_to_pyrender_mesh(polydata_to_trimesh(stl), material=DRRMaterial("bone", density=0, subtractive=True, layer=1))
            cube_meshes.append(deepdrr.Mesh(mesh=prim, world_from_anatomical=geo.FrameTransform.from_translation([10, 30, 5])))

        body = polydata_to_pyrender_mesh(stls[0], material=DRRMaterial("titanium", density=0.1, subtractive=False))
        body = deepdrr.Mesh(mesh=body, world_from_anatomical=geo.FrameTransform.from_translation([0, 20, 0]))

        # TODO: implemnet blend shapes - should support up to 8 morph targets
        

        stl3 = pv.read("tests/resources/threads.stl")
        stl3.scale([10000, 10000, 10000], inplace=True)

        # print max and min of verts
        print(stl3.points.max(axis=0))
        print(stl3.points.min(axis=0))


        kwire_density = 0
        prim3 = polydata_to_pyrender_mesh(stl3, material=DRRMaterial("titanium", density=kwire_density, subtractive=True, layer=1))
        meshtransform = None
        mesh3 = deepdrr.Mesh(mesh=prim3, world_from_anatomical=meshtransform)


        N = 4
        i=0

        res_scale = 1
        carm = deepdrr.SimpleDevice(sensor_width=400*res_scale, sensor_height=400*res_scale, pixel_size=2*4/res_scale, source_to_detector_distance=4000)

        z = geo.FrameTransform.from_translation([0,0,0])
        b = geo.FrameTransform.from_rotation(geo.Rotation.from_euler("x", 0*-np.pi/2))
        b2 = geo.FrameTransform.from_rotation(geo.Rotation.from_euler("y", -i/N*np.pi*2))
        # c = geo.FrameTransform.from_translation([0, 0, -30])
        c = geo.FrameTransform.from_translation([0, 0, -2000])
        new = z @ b @ b2 @ c

        carm._device_from_camera3d = new

        projector = deepdrr.Projector(
            volume=[body,mesh3]+cube_meshes,
            carm=carm,
            step=0.01,  # stepsize along projection ray, measured in voxels
            mode="linear",
            max_block_index=65535,
            spectrum="90KV_AL40",
            photon_count=100000,
            scatter_num=0,
            threads=8,
            num_mesh_layers=128,
        )

        images = []

        # set np seed
        np.random.seed(2)
        rand_coeffs = np.random.rand(4, 3) * 100

        duration = 10000
        # N = 100
        N = 20
        # N = 200

        # fps = 25
        # N = int(duration/1000*fps)
        with projector:

            for i in tqdm.tqdm(range(N)):
                # i = 35

                # move each cube in xyz in a periodic motion that occurs 3 times during the gif with a random phase by setting m.world_from_anatomical
                for m_idx, m in enumerate(cube_meshes):
                    motion_magnitude = 300
                    motion_cycle = 3

                    # t=  35
                    t=  i
                    a = geo.FrameTransform.from_translation([
                        motion_magnitude * np.sin(t/N*np.pi*2*motion_cycle + rand_coeffs[m_idx, 0]),
                        # motion_magnitude * np.sin(i/N*np.pi*2*motion_cycle + rand_coeffs[m_idx, 1]),
                        0,
                        motion_magnitude * np.sin(t/N*np.pi*2*motion_cycle + rand_coeffs[m_idx, 2]),
                    ])
                    if m_idx in [2]:
                        mesh3.world_from_anatomical = a

                    if m_idx in [2, 1]:
                        a =  geo.FrameTransform.from_translation([0,0,0])
                        # a = geo.FrameTransform.from_translation([1000,1000,1000])

                    m.world_from_anatomical = a

                z = geo.FrameTransform.from_translation([0,0,0])
                b = geo.FrameTransform.from_rotation(geo.Rotation.from_euler("x", -np.pi/2))
                b2 = geo.FrameTransform.from_rotation(geo.Rotation.from_euler("y", -i/N*np.pi*2))
                # c = geo.FrameTransform.from_translation([0, 0, -30])
                c = geo.FrameTransform.from_translation([0, 0, -2000])
                new = z @ b @ b2 @ c

                carm._device_from_camera3d = new

                image = projector.project()
                
                image_256 = (image * 255).astype(np.uint8)
                images.append(Image.fromarray(image_256))

        
        name = f'test_mesh_mesh_1.gif'
        output_gif_path = self.output_dir/name

        with verify_image(name, actual_dir=self.output_dir, expected_dir=self.truth, diff_dir=self.diff_dir):
            images[0].save(
                output_gif_path,
                save_all=True,
                append_images=images[1:],
                duration=duration/N,  # Duration between frames in milliseconds
                loop=0,  # 0 means loop indefinitely, you can set another value if needed
                disposal=1,  # 2 means replace with background color (use 1 for no disposal)
            )


    def test_mesh_only(self):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        
        # load 10cmcube.stl from resources folder
        # stl = pv.read("tests/resources/10cmrighttri.stl")
        stl3 = pv.read("tests/resources/10cmcube.stl")
        stl3.scale([100, 100, 100], inplace=True)
        stl3.rotate_z(60, inplace=True)
        # stl3.translate([0, 00, 0], inplace=True)

        stl2 = pv.read("tests/resources/10cmcube.stl")
        stl2.scale([200, 200, 200], inplace=True)
        # stl2.translate([0, 30, 0], inplace=True)
        stl = pv.read("tests/resources/solenoidasm.stl")
        stl.scale([400/1000]*3, inplace=True)
        # stl = pv.read("tests/resources/suzanne.stl")
        # stl.scale([200]*3, inplace=True)
        # stl.translate([0, 0, 0], inplace=True)
        stl.rotate_y(60, inplace=True)
        stl.rotate_x(10, inplace=True)
        stl.rotate_z(80, inplace=True)
        stl.translate([40, -200, -0], inplace=True)
        # stl = pv.read("tests/resources/suzanne.stl")
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
        prim = pyrender.Mesh.from_trimesh(polydata_to_trimesh(stl), material=DRRMaterial("titanium", density=7, subtractive=False))
        # prim = pyrender.Mesh.from_trimesh(polydata_to_trimesh(stl), material=DRRMaterial("bone", density=2, subtractive=True))
        # mesh = deepdrr.Mesh(mesh=prim)
        mesh = deepdrr.Mesh(mesh=prim, world_from_anatomical=geo.FrameTransform.from_rotation(geo.Rotation.from_euler("x", 90, degrees=True)))
        # mesh = deepdrr.Mesh(mesh=prim, world_from_anatomical=geo.FrameTransform.from_rotation(geo.Rotation.from_euler("x", 90, degrees=True) * geo.Rotation.from_euler("y", 30, degrees=True)))

        # prim2 = deepdrr.Primitive("titanium", 2, stl2, subtractive=True)
        prim2 = trimesh_to_pyrender_mesh(polydata_to_trimesh(stl2), material=DRRMaterial("lung", density=2, subtractive=True))
        mesh2 = deepdrr.Mesh(mesh=prim2, world_from_anatomical=geo.FrameTransform.from_translation([30, 50, 200]))

        # prim3 = deepdrr.Primitive("titanium", 0, stl2, subtractive=True)
        prim3 = polydata_to_pyrender_mesh(stl2, material=DRRMaterial("titanium", density=0, subtractive=True))
        mesh3 = deepdrr.Mesh(mesh=prim3, world_from_anatomical=geo.FrameTransform.from_translation([-30, 20, -70]))
        # mesh = deepdrr.Mesh("polyethylene", 1.05, stl)
        # mesh.morph_weights = np.array([-10])
        
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world, sensor_width=300*2, sensor_height=200*2, pixel_size=0.6/2)
        # self.project([volume], carm, "test_mesh.png")
        # self.project([mesh, mesh2, mesh3], carm, "test_mesh.png")
        self.project([mesh, mesh2, mesh3], carm, "test_mesh_only.png", verify=True, num_mesh_layers=32)
        # self.project([volume, mesh, mesh2, mesh3], carm, "test_mesh.png", verify=False, num_mesh_layers=64)

    
    def test_cube(self):
        # TODO: There is some bug that causes this test to fail sometimes when run with other tests, but it works when run by itself

        # volume = deepdrr.Volume.from_nrrd(self.file_path)
        vol_voxel_N = 2
        # vol_voxel_N = 100
        density_arr = np.ones((vol_voxel_N, vol_voxel_N*5, vol_voxel_N), dtype=np.float32)*7
        titanium_arr = np.ones([vol_voxel_N, vol_voxel_N*5, vol_voxel_N], dtype=np.float32)
        anatomical_from_IJK = np.zeros((4,4), dtype=np.float32)
        anatomical_from_IJK[0,0] = 0.02
        anatomical_from_IJK[1,1] = 0.02
        anatomical_from_IJK[2,2] = 0.02
        anatomical_from_IJK[3,3] = 1

        volume = deepdrr.Volume(density_arr, {"titanium": titanium_arr}, anatomical_from_IJK=anatomical_from_IJK)
        # load 10cmcube.stl from resources folder
        # stl = pv.read("tests/resources/10cmrighttri.stl")
        # stl3 = pv.read("tests/resources/10cmcube.stl")
        stl3 = pv.read("tests/resources/xyzfaces.stl")
        # stl3 = pv.read("tests/resources/threads.stl")
        stl3.scale([100, 100, 100], inplace=True)
        # stl3.translate([0, 0, 2], inplace=True)
        stl3.rotate_x(30, inplace=True)
        stl3.rotate_y(30, inplace=True)
        stl3.rotate_z(30, inplace=True)

        # stl3_verts = np.array(stl3.points)
        # stl3_verts[:, 0] *= 1
        # stl3_verts[:, 1] *= 1
        # stl3_verts[:, 2] *= 1
        # stl3_faces = np.array(stl3.faces).reshape(-1, 4)
        # stl3_faces = stl3_faces[:, [0, 2, 1, 3]]
        # stl3_faces = stl3_faces.flatten()

        # stl3 = pv.PolyData(stl3_verts, stl3_faces)
        # stl3.rotate_z(180, inplace=True)


        # prim3 = polydata_to_pyrender_mesh(stl3, material=DRRMaterial("titanium", density=7, subtractive=False))
        prim3 = polydata_to_pyrender_mesh(stl3, material=DRRMaterial("titanium", density=0, subtractive=True, layer=1))
        meshtransform = geo.FrameTransform.from_translation([-3, 2, -7]) @ geo.FrameTransform.from_rotation(geo.Rotation.from_euler("x", 60, degrees=True))
        mesh3 = deepdrr.Mesh(mesh=prim3, world_from_anatomical=meshtransform)

        # stl4 = pv.read("tests/resources/threads.stl")
        # stl4 = pv.read("tests/resources/xyzfaces.stl")
        # stl4.scale([100, 100, 100], inplace=True)
        # stl4.rotate_x(30, inplace=True)
        # stl4.rotate_y(30, inplace=True)
        # stl4.rotate_z(30, inplace=True)



        # stl4.rotate_z(180, inplace=True)

        cube_vol = deepdrr.Volume.from_meshes(voxel_size = 0.3, world_from_anatomical=meshtransform, surfaces=[("titanium", 7, stl3)])
        
        carm = deepdrr.SimpleDevice(sensor_width=400*4, sensor_height=400*4, pixel_size=2)
        # carm = deepdrr.SimpleDevice(sensor_width=200*4, sensor_height=400*4, pixel_size=0.02)

        projector = deepdrr.Projector(
            # volume=[volume, mesh4],
            # volume=[mesh3],
            # volume=[mesh4],
            # volume=[mesh3],
            volume=[cube_vol, mesh3],
            # volume=[cube_vol, mesh3],
            # volume=[volume, mesh3],
            carm=carm,
            step=0.01,  # stepsize along projection ray, measured in voxels
            mode="linear",
            max_block_index=65535,
            spectrum="90KV_AL40",
            photon_count=100000,
            scatter_num=0,
            threads=8,
            neglog=True,
            num_mesh_layers=32
        )

        images = []

        N = 10
        with projector:
            for i in tqdm.tqdm(range(N)):
                z = geo.FrameTransform.from_translation([10*np.sin(-i/N*np.pi*2*2), 10*np.sin(-i/N*np.pi*2), 0])
                a = geo.FrameTransform.from_rotation(geo.Rotation.from_euler("x", -i/N*np.pi*2))
                b = geo.FrameTransform.from_rotation(geo.Rotation.from_euler("y", -i/N*np.pi*2))
                c = geo.FrameTransform.from_translation([0, 0, -30])
                # c = geo.FrameTransform.from_translation([0, 0, -10])
                new = z @ a @ b @ c
                carm._device_from_camera3d = new

                image = projector.project()

                image_256 = (image * 255).astype(np.uint8)
                images.append(Image.fromarray(image_256))

        # Save the list of images as a GIF
        name = f"test_cube.gif"
        output_gif_path = self.output_dir/name

        with verify_image(name, actual_dir=self.output_dir, expected_dir=self.truth, diff_dir=self.diff_dir):
            images[0].save(
                output_gif_path,
                save_all=True,
                append_images=images[1:],
                duration=5000/N,  # Duration between frames in milliseconds
                loop=0,  # 0 means loop indefinitely, you can set another value if needed
                disposal=1,  # 2 means replace with background color (use 1 for no disposal)
            )



    def gen_threads(self):
        # pass

        for kwire_visible in [False, True]:
            for vol_voxel_N in [10]:
            # for vol_voxel_N in [10, 100]:
                for step in [0.5, 0.1, 0.01]:
                    for mesh_enabled in [False, True]:
                        for thread_voxel_size in [0.2,0.05] if not mesh_enabled else [0.2]:
                            self.gen_threads_gif(vol_voxel_N=vol_voxel_N, step=step, thread_voxel_size=thread_voxel_size, mesh_enabled=mesh_enabled, kwire_visible=kwire_visible)

    def gen_threads_gif(self, vol_voxel_N=100, step=0.01, thread_voxel_size=0.05, mesh_enabled=True, kwire_visible=True, name="output"):
        name = f"kwire_visible={kwire_visible}_vol_voxel_N={vol_voxel_N}_step={step}_thread_voxel_size={thread_voxel_size}_mesh_enabled={mesh_enabled}"
        print(name)
        # vol_voxel_N = 2
        vol_voxel_N = 100
        density_arr = np.ones((vol_voxel_N, vol_voxel_N*5, vol_voxel_N), dtype=np.float32)*7
        titanium_arr = np.ones([vol_voxel_N, vol_voxel_N*5, vol_voxel_N], dtype=np.float32)
        anatomical_from_IJK = np.zeros((4,4), dtype=np.float32)
        anatomical_from_IJK[0,0] = 0.02/vol_voxel_N*100
        anatomical_from_IJK[1,1] = 0.02/vol_voxel_N*100
        anatomical_from_IJK[2,2] = 0.02/vol_voxel_N*100
        anatomical_from_IJK[3,3] = 1

        volume = deepdrr.Volume(density_arr, {"titanium": titanium_arr}, anatomical_from_IJK=anatomical_from_IJK)

        # stl3 = pv.read("tests/resources/xyzfaces.stl")
        stl3 = pv.read("tests/resources/threads.stl")
        stl3.scale([100, 100, 100], inplace=True)


        kwire_density = 7 if kwire_visible else 0
        prim3 = polydata_to_pyrender_mesh(stl3, material=DRRMaterial("titanium", density=kwire_density, subtractive=True))
        # prim3 = polydata_to_pyrender_mesh(stl3, material=DRRMaterial("titanium", density=0, subtractive=True))
        meshtransform = None
        mesh3 = deepdrr.Mesh(mesh=prim3, world_from_anatomical=meshtransform)

        if not mesh_enabled:
            mesh3 = deepdrr.Volume.from_meshes(voxel_size = thread_voxel_size, world_from_anatomical=meshtransform, surfaces=[("titanium", kwire_density, stl3)])
        # mesh4 = deepdrr.Volume.from_meshes(voxel_size = 0.05, world_from_anatomical=meshtransform, surfaces=[("titanium", 7, stl3)])
        
        carm = deepdrr.SimpleDevice(sensor_width=200*2, sensor_height=400*2, pixel_size=2)

        projector = deepdrr.Projector(
            # volume=[volume, mesh4],
            # volume=[mesh4, mesh3],
            volume=[volume, mesh3],
            carm=carm,
            step=step,  # stepsize along projection ray, measured in voxels
            mode="linear",
            max_block_index=65535,
            spectrum="90KV_AL40",
            photon_count=100000,
            scatter_num=0,
            threads=8,
            neglog=True,
            num_mesh_layers=32
        )

        images = []

        # N = 20
        N = 100
        with projector:
            for i in range(N):
                z = geo.FrameTransform.from_translation([0, 5, 0])
                # z = geo.FrameTransform.from_translation([10*np.sin(-i/N*np.pi*2*2), 10*np.sin(-i/N*np.pi*2), 0])
                # a = geo.FrameTransform.from_rotation(geo.Rotation.from_euler("x", -i/N*np.pi*2))
                b = geo.FrameTransform.from_rotation(geo.Rotation.from_euler("y", -i/N*np.pi*2))
                # c = geo.FrameTransform.from_translation([0, 0, -30])
                c = geo.FrameTransform.from_translation([0, 0, -10])
                new = z @ b @ c
                carm._device_from_camera3d = new

                image = projector.project()

                image_256 = (image * 255).astype(np.uint8)
                images.append(Image.fromarray(image_256))

        # Save the list of images as a GIF
        output_gif_path = self.output_dir/f'{name}.gif'
        images[0].save(
            output_gif_path,
            save_all=True,
            append_images=images[1:],
            duration=7000/N,  # Duration between frames in milliseconds
            loop=0,  # 0 means loop indefinitely, you can set another value if needed
            disposal=1,  # 2 means replace with background color (use 1 for no disposal)
        )


    def test_multi_projector(self):
        for i in range(10):
            self.test_mesh()

    def test_layer_depth(self):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        # load 10cmcube.stl from resources folder
        # stl = pv.read("tests/resources/10cmrighttri.stl")
        stl3 = pv.read("tests/resources/10cmcube.stl")
        stl3.scale([100, 100, 100], inplace=True)
        stl3.rotate_z(60, inplace=True)
        # stl3.translate([0, 00, 0], inplace=True)

        stl2 = pv.read("tests/resources/10cmcube.stl")
        stl2.scale([200, 200, 200], inplace=True)
        # stl2.translate([0, 30, 0], inplace=True)
        stl = pv.read("tests/resources/suzanne.stl")
        stl.scale([200]*3, inplace=True)
        stl.translate([0, -200, 0], inplace=True)
        # stl = pv.read("tests/resources/suzanne.stl")
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
        prim = pyrender.Mesh.from_trimesh(polydata_to_trimesh(stl), material=DRRMaterial("titanium", density=2, subtractive=True))
        # prim = pyrender.Mesh.from_trimesh(polydata_to_trimesh(stl), material=DRRMaterial("bone", density=2, subtractive=True))
        mesh = deepdrr.Mesh(mesh=prim, world_from_anatomical=geo.FrameTransform.from_rotation(geo.Rotation.from_euler("x", 90, degrees=True) * geo.Rotation.from_euler("y", 30, degrees=True)))

        # prim2 = deepdrr.Primitive("titanium", 2, stl2, subtractive=True)
        prim2 = trimesh_to_pyrender_mesh(polydata_to_trimesh(stl2), material=DRRMaterial("lung", density=2, subtractive=True))
        mesh2 = deepdrr.Mesh(mesh=prim2, world_from_anatomical=geo.FrameTransform.from_translation([30, 50, 200]))

        # prim3 = deepdrr.Primitive("titanium", 0, stl2, subtractive=True)
        mat = DRRMaterial("titanium", density=0, subtractive=True)
        overlapping = [
            deepdrr.Mesh(mesh=polydata_to_pyrender_mesh(stl2, material=mat), world_from_anatomical=geo.FrameTransform.from_translation([-30, 40, -70])),
            # deepdrr.Mesh(mesh=polydata_to_pyrender_mesh(stl2, material=mat), world_from_anatomical=geo.FrameTransform.from_translation([-30, 30, -70])),
            deepdrr.Mesh(mesh=polydata_to_pyrender_mesh(stl2, material=mat), world_from_anatomical=geo.FrameTransform.from_translation([-30, 50, -50])),
            deepdrr.Mesh(mesh=polydata_to_pyrender_mesh(stl2, material=mat), world_from_anatomical=geo.FrameTransform.from_translation([-30, 60, -30])),
            # deepdrr.Mesh(mesh=polydata_to_pyrender_mesh(stl2, material=mat), world_from_anatomical=geo.FrameTransform.from_translation([-30, 20, -70])),
            # deepdrr.Mesh(mesh=polydata_to_pyrender_mesh(stl2, material=mat), world_from_anatomical=geo.FrameTransform.from_translation([-30, 20, -70])),
        ]
        # mesh = deepdrr.Mesh("polyethylene", 1.05, stl)
        # mesh.morph_weights = np.array([-10])
        
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world, sensor_width=300, sensor_height=200, pixel_size=0.6)
        # self.project([volume], carm, "test_mesh.png")
        # self.project([mesh, mesh2, mesh3], carm, "test_mesh.png")
        self.project([volume, mesh, mesh2] + overlapping, carm, "test_mesh_d4.png", num_mesh_layers=4, verify=False)
        self.project([volume, mesh, mesh2] + overlapping, carm, "test_mesh_d64.png", num_mesh_layers=64, verify=False)
        self.project([volume, mesh, mesh2] + overlapping, carm, "test_mesh_d128.png", num_mesh_layers=128, verify=False)



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
    # test.test_mesh_only()
    test.test_mesh()
    # test.gen_threads()
    # test.test_cube()
    # test.test_mesh_mesh_1()
    # volume = test.load_volume()
    # carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
    # test.project(volume, carm, "test.png")
