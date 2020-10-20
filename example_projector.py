#! python3

import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# old deepdrr imports
from deepdrr import projector
from deepdrr.load_dicom import load_dicom, conv_hu_to_materials_thresholding, conv_hu_to_density
from deepdrr.analytic_generators import add_noise
from deepdrr import mass_attenuation_gpu as mass_attenuation
from deepdrr import add_scatter

# new deepdrr imports
from deepdrr import utils
from deepdrr import spectrums
from deepdrr.geometry import Camera, Projection
from deepdrr.projector import generate_projections


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
        isocenter_distance=800
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
    # origin [0,0,0] corresponds to the center of the volume
    origin = [0, 0, 0]
    spectrum = spectrums.SPECTRUM90KV_AL40
    scatter = False
        
    # generate projection matrices over uniform angles on a sphere
    projections = camera.make_projections_from_range(
        phi_range=(min_phi, max_phi, spacing_phi),
        theta_range=(min_theta, max_theta, spacing_theta)
    )

    # forward project densities
    forward_projections = projector.generate_projections(
        projections,
        volume,
        materials,
        origin,
        voxel_size,
        camera.sensor_width,
        camera.sensor_height,
        mode="linear",
        max_blockind=200,
        threads=8,
    )

    # calculate intensity at detector (images: mean energy one photon emitted from the source
    # deposits at the detector element, photon_prob: probability of a photon emitted from the
    # source to arrive at the detector)
    images, photon_prob = mass_attenuation.calculate_intensity_from_spectrum(forward_projections, spectrum)
    # add scatter
    if scatter:
        scatter_net = add_scatter.ScatterNet()
        scatter = scatter_net.add_scatter(images, camera)
        photon_prob *= 1 + scatter / images
        images += scatter
        
    # transform to collected energy in keV per cm^2
    # images = images * (photon_count / (camera.pixel_size * camera.pixel_size))
    
    # add poisson noise
    images = add_noise(images, photon_prob, photon_count)

    # show result
    plt.imshow(images[0, :, :], cmap="gray")
    plt.show()

    

if __name__ == "__main__":
    main()
