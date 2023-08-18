import logging
import os
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from deepdrr.load_dicom import (
    load_dicom,
    conv_hu_to_materials_thresholding,
    conv_hu_to_density,
)
from deepdrr import utils
from deepdrr import Volume, MobileCArm, Projector
from deepdrr import geo
import pytest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@pytest.mark.skip(reason="anatomical_coordinate_system=None not implemented")
def test_phantom():
    # Define a simple phantom for test: a wire box around a cube.
    volume = np.zeros((120, 100, 80), dtype=np.float32)
    volume[0, 0, :] = 1
    volume[0, -1, :] = 1
    volume[-1, 0, :] = 1
    volume[-1, -1, :] = 1
    volume[:, 0, 0] = 1
    volume[:, 0, -1] = 1
    volume[:, -1, 0] = 1
    volume[:, -1, -1] = 1
    volume[0, :, 0] = 1
    volume[0, :, -1] = 1
    volume[-1, :, 0] = 1
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
        anatomical_coordinate_system=None,  # LPS, RAS, or None.
        world_from_anatomical=None,  # anatomical coordinate system is same as world
    )

    # defines the C-Arm device, which is a convenience class for positioning the Camera.
    carm = MobileCArm()

    # Angles to take projections over
    #
    with Projector(
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
        images = projector.project()

    # save results as matplotlib plots
    # output_dir = Path(f"examples")
    # output_dir.mkdir(exist_ok=True)
    # for i, image in enumerate(images):
    #     plt.imshow(image, cmap="gray")
    #     plt.title(f"phi, theta = {phis[i], thetas[i]}")
    #     output_path = (
    #         output_dir / f"image_phi={int(phis[i])}_theta={int(thetas[i])}.png"
    #     )
    #     logger.info(f"writing image {output_path}")
    #     plt.savefig(output_path)

if __name__ == "__main__":
    test_phantom()