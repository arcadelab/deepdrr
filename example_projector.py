#! python3

import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from deepdrr import projector
from deepdrr import projection_matrix
from deepdrr.load_dicom import load_dicom, conv_hu_to_materials_thresholding, conv_hu_to_density
from deepdrr.utils import image_saver, Camera, param_saver
from deepdrr.analytic_generators import add_noise
from deepdrr import mass_attenuation_gpu as mass_attenuation
from deepdrr import spectrums
from deepdrr import add_scatter


def generate_projections_on_sphere(
        volume_path: str,
        save_path: str,
        min_theta,
        max_theta,
        min_phi,
        max_phi,
        spacing_theta,
        spacing_phi,
        photon_count,
        camera,
        spectrum,
        scatter=False,
        origin=[0, 0, 0]):

    print(f'volume_path: {volume_path}')
    print(f'save_path: {save_path}')
    print(f'min_theta: {min_theta}')
    print(f'max_theta: {max_theta}')
    print(f'min_phi: {min_phi}')
    print(f'max_phi: {max_phi}')
    print(f'spacing_theta: {spacing_theta}')
    print(f'spacing_phi: {spacing_phi}')
    print(f'photon_count: {photon_count}')
    print(f'camera: {camera}')
    print(f'spectrum: {spectrum}')
    print(f'scatter: {scatter})')
    print(f'origin: {origin}')

    # generate angle pairs on a sphere
    thetas, phis = projection_matrix.generate_uniform_angels(
        min_theta,
        max_theta,
        min_phi,
        max_phi,
        spacing_theta,
        spacing_phi,
    )

    print(f'thetas, phis: {thetas}, {phis}')
    
    # generate projection matrices from angles
    proj_mats = projection_matrix.generate_projection_matrices_from_values(
        camera.source_to_detector_distance,
        camera.pixel_size,
        camera.pixel_size,
        camera.sensor_width,
        camera.sensor_height,
        camera.isocenter_distance,
        phis,
        thetas,
    )

    print(f'projection matrices: {proj_mats}')
    
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
    
    # forward project densities
    forward_projections = projector.generate_projections(
        proj_mats,
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
    
    # save images
    image_saver(images, "DRR", save_path)
    # save parameters
    param_saver(thetas, phis, proj_mats, camera, origin, photon_count, spectrum, "simulation_data", save_path)

    # show result
    plt.imshow(images[0, :, :], cmap="gray")
    plt.show()


def main():
    # 2x2 binning
    camera = Camera(sensor_width=1240, sensor_height=960, pixel_size=0.31,
                    source_to_detector_distance=1200, isocenter_distance=800)
    # 4x4 binning
    # camera = Camera(sensor_width=620, sensor_height=480, pixel_size=0.62,
    # source_to_detector_distance=1200,isocenter_distance=800)

    ####
    # define the path to your dicoms here or use the simple phantom from the code above
    ####
    dicompath = r"./your_dicom_directory/"
    
    save_path = r"./generated_data/test"
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
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        
    generate_projections_on_sphere(
        dicompath,
        save_path,
        min_theta,
        max_theta,
        min_phi,
        max_phi,
        spacing_theta,
        spacing_phi,
        photon_count,
        camera,
        spectrum,
        origin=origin,
        scatter=False,
    )


if __name__ == "__main__":
    main()
