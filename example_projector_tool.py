import os
import matplotlib.pyplot as plt
import numpy as np

from deepdrr import projector
from deepdrr import projection_matrix
from deepdrr.load_dicom import load_dicom
from deepdrr.load_dicom_tool import load_dicom_metal, load_dicom_CT, replace_material
from deepdrr.downsample_tool import downsample_tool
from deepdrr.analytic_generators import add_noise
from deepdrr import mass_attenuation_gpu as mass_attenuation
from deepdrr import spectrum_generator
from deepdrr import add_scatter
from deepdrr.utils import image_saver, Camera, param_saver


def generate_projections_on_sphere(CT_volume_path, save_path, min_theta, max_theta, min_phi, max_phi, spacing_theta, spacing_phi, photon_count, camera, spectrum, scatter=False, origin=[0, 0, 0]):
    # generate angle pairs on a sphere
    thetas, phis = projection_matrix.generate_uniform_angels(min_theta, max_theta, min_phi, max_phi, spacing_theta, spacing_phi)
    # generate projection matrices from angles
    proj_mats = projection_matrix.generate_projection_matrices_from_values(camera.source_to_detector_distance, camera.pixel_size, camera.pixel_size, camera.sensor_width, camera.sensor_height, camera.isocenter_distance, phis, thetas)

    ####
    # Recommend origin settings
    origin = [70, 0, 10]
    origin_metal = [0, 50, 10]

    metal_path = r"./sample_metal_volume/sample.dcm"
    # load CT volume
    trun = 1000
    ori_CT_volume, CT_volume, CT_materials, CT_voxel_size = load_dicom_CT(CT_volume_path, fixed_slice_thinckness=1.0, truncate=[[0, None], [0, None], [0, trun]], use_thresholding_segmentation=True)

    # load metal volume
    metal_volume, metal_materials, metal_voxel_size = load_dicom_metal(metal_path, fixed_slice_thinkness=1.0, sortBy="InstanceNumber", use_thresholding_segmentation=True)

    # replace metal material with CT materials
    metal_materials_n = replace_material(metal_volume, smooth_air=False, use_thresholding_segmentation=True)

    # upsamle metal volume to CT volume
    metal_volume_m, metal_volume_m_ori, metal_materials_m = downsample_tool(ori_CT_volume, CT_volume, CT_voxel_size, metal_volume, metal_voxel_size, CT_materials, metal_materials_n, origin, origin_metal)

    # forward CT project densities of CT materials
    forward_projections = projector.generate_projections(proj_mats, CT_volume, CT_materials, origin, CT_voxel_size, camera.sensor_width, camera.sensor_height, mode="linear", max_blockind=200, threads=8)

    # forward metal project densities of metal material
    metal_projections = projector.generate_projections(proj_mats, metal_volume, metal_materials, origin_metal, metal_voxel_size, camera.sensor_width, camera.sensor_height, mode="linear", max_blockind=200, threads=8)

    # forward metal project densities of CT materials
    metal_projections_m = projector.generate_projections(proj_mats, metal_volume_m, metal_materials_m, origin_metal, metal_voxel_size, camera.sensor_width, camera.sensor_height, mode="linear", max_blockind=200, threads=8)

    # subtract in projection domain
    for mat in CT_materials:
        forward_projections[mat] = forward_projections[mat] - metal_projections_m[mat]

    # add back metal projection
    forward_projections.update(metal_projections)

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
    camera = Camera(sensor_width=512, sensor_height=512, pixel_size=0.62, source_to_detector_distance=1200, isocenter_distance=450)
    # 4x4 binning
    # camera = Camera(sensor_width=620, sensor_height=480, pixel_size=0.62, source_to_detector_distance=1200,isocenter_distance=800)

    ####
    # define the path to your dicoms here or use the simple phantom from the code above
    ####
    CT_volume_path = r".\your_dicom_directory\\"

    #save_path = r".\generated_data\test"
    save_path = r"./test"
    min_theta = 75
    max_theta = 105
    min_phi = 75
    max_phi = 105
    spacing_theta = 30
    spacing_phi = 30
    photon_count = 100000
    # origin [0,0,0] corresponds to the center of the volume
    origin = [0, 0, 0]
    spectrum = spectrum_generator.SPECTRUM90KV_AL40

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    generate_projections_on_sphere(CT_volume_path, save_path, min_theta, max_theta, min_phi, max_phi, spacing_theta, spacing_phi, photon_count, camera, spectrum, origin=origin, scatter=False)


if __name__ == "__main__":
    main()
