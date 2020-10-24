#! python3

import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# old deepdrr imports
from deepdrr.load_dicom import load_dicom, conv_hu_to_materials_thresholding, conv_hu_to_density

# new deepdrr imports
from deepdrr import utils
from deepdrr.geometry import Camera, Projection
from deepdrr.projector import Projector


def main():
    ####
    # Use this if you have a volume
    ####
    # load and segment volume
    # volume, materials, voxel_size = load_dicom(volume_path, use_thresholding_segmentation=False)
    # dataset = DeepFluoro(Path.home() / 'datasets')
    # vol = dataset.get_volume(0)
    # volume = np.array(vol['pixels'])
    # # origin = np.array(vol['origin']).reshape(3)
    # oritin = [0, 0, 0]
    # materials = conv_hu_to_materials_thresholding(volume)
    # voxel_size = np.array([1, 1, 1], dtype=np.float32)
    # volume = conv_hu_to_density(volume, smoothAir=False)
    
    ####
    # Otherwise use this simple phantom for test
    ####
    # start of phantom
    volume = np.zeros((100, 100, 100), dtype=np.float32)
    volume[20:40, 20:40, 20:40] = 1
    volume[60:80, 60:80, 60:80] = 2
    materials = {}
    materials["air"] = volume == 0
    materials["soft tissue"] = volume == 1
    materials["bone"] = volume == 2
    voxel_size = np.array([1, 1, 1], dtype=np.float32)
    # end of phantom

    # 2x2 binning
    camera = Camera.from_parameters(
        sensor_size=(1240, 960),
        pixel_size=0.31,
        source_to_detector_distance=1200, 
        isocenter_distance=800,
    )

    # 4x4 binning
    # camera = Camera(
    #     ensor_width=620, 
    #     sensor_height=480, 
    #     pixel_size=0.62,
    #     source_to_detector_distance=1200,
    #     isocenter_distance=800
    # )

    # Define angles
    min_theta = 60
    max_theta = 120
    min_phi = 0
    max_phi = 91
    spacing_theta = 30
    spacing_phi = 90
    photon_count = 100000
        
    # arrange angles as ranges over uniform angles on a sphere
    phi_range = (min_phi, max_phi, spacing_phi)
    theta_range = (min_theta, max_theta, spacing_theta)

    # make the projector object, but do not reserve GPU resources yet.
    projector = Projector(
        volume=volume, # TODO: should be converted to a "VolumeData" object.
        segmentation=materials,
        materials=list(materials.keys()),
        voxel_size=voxel_size,
        camera=camera,
        origin=[0, 0, 0], # corresponds to center of volume
        mode="linear",
        max_block_index=200,
        spectrum='90KV_AL40',
        photon_count=photon_count,
        add_scatter=False, # add photon scatter
        threads=8,
        centimeters=True,
    )

    # use a with block to allocate memory for the projector.
    # Alternatively, use projector.initialize() and projector.free()
    with projector:
        images = projector.over_range(phi_range, theta_range)

    # show result
    plt.imshow(images[0, :, :], cmap="gray")
    plt.show()

    

if __name__ == "__main__":
    main()
