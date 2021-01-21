#! python3

import logging
import os
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from time import time

from deepdrr.load_dicom import load_dicom, conv_hu_to_materials_thresholding, conv_hu_to_density
from deepdrr import utils
from deepdrr import Volume, CArm, Projector
from deepdrr import geo


logger = logging.getLogger(__name__)


def main():
    # Define a simple phantom for test: a wire box around a cube.
    volume = np.zeros((120, 100, 80), dtype=np.float32)
    volume[0, 0, :] = 1
    volume[0, -1, :] = 1
    volume[-1, 0, :] = 1
    volume[-1,-1, :] = 1
    volume[:, 0, 0] = 1
    volume[:, 0, -1] = 1
    volume[:, -1, 0] = 1
    volume[:, -1, -1] = 1
    volume[0, :, 0] = 1
    volume[0, :, -1] = 1
    volume[-1,:, 0] = 1
    volume[-1, :, -1] = 1

    volume[40:60, 40:60, 40:60] = 1
    materials = {}
    materials["air"] = volume == 0
    materials["soft tissue"] = volume == 1
    materials["bone"] = volume == 2
    voxel_size = np.array([1, 1, 1], dtype=np.float32)

    # Use the center of the volume as the "world" coordinates. The origin is the (0, 0, 0) index of the volume in the world frame.
    vol_center = (np.array(volume.shape) - 1) / 2 * voxel_size
    origin = geo.point(-vol_center[0], -vol_center[1], -vol_center[2])

    # Create the volume object with segmentation
    volume = Volume.from_parameters(
        data=volume,
        materials=materials, 
        origin=origin, 
        spacing=voxel_size,
        anatomical_coordinate_system=None, # LPS, RAS, or None.
        world_from_anatomical=None, # anatomical coordinate system is same as world
    )

    # defines the C-Arm device, which is a convenience class for positioning the Camera.
    carm = CArm(
        isocenter=geo.point(0, 0, 0),
        isocenter_distance=800,
    )

    # camera intrinsics of the projection, this uses 2x2 binning
    camera_intrinsics = geo.CameraIntrinsicTransform.from_sizes(
        sensor_size=(1240, 960),
        pixel_size=0.31,
        source_to_detector_distance=1200,
    )

    # Angles to take projections over
    min_theta = 0
    max_theta = 120
    min_phi = 0
    max_phi = 91
    spacing_theta = 30
    spacing_phi = 90

    t = time()
    with Projector(
        volume=volume,
        camera_intrinsics=camera_intrinsics,
        carm=carm,
        step=0.1, # stepsize along projection ray, measured in voxels
        mode='linear',
        max_block_index=200,
        spectrum='90KV_AL40',
        photon_count=100000,
        add_scatter=False,
        threads=8,
        neglog=True,
    ) as projector:
        images = projector.project_over_carm_range(
            (min_phi, max_phi, spacing_phi),
            (min_theta, max_theta, spacing_theta)
        )
    dt = time() - t
    logger.info(f"projected {images.shape[0]} views in {dt:.03f}s")

    # get the thetas an phis for file names
    phis, thetas = utils.generate_uniform_angles(
        (min_phi, max_phi, spacing_phi),
        (min_theta, max_theta, spacing_theta))

    # save results as matplotlib plots
    output_dir = Path(f'examples')
    output_dir.mkdir(exist_ok=True)
    for i, image in enumerate(images):
        plt.imshow(image, cmap="gray")
        plt.title(f'phi, theta = {phis[i], thetas[i]}')
        output_path = output_dir / f'image_phi={int(phis[i])}_theta={int(thetas[i])}.png'
        logger.info(f'writing image {output_path}')
        plt.savefig(output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
