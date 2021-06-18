"""Legacy code for loading tool volumes."""

import pydicom as dicom
import glob
import numpy as np
from skimage.transform import resize

from . import segmentation


materials = {1: "air", 2: "soft tissue", 3: "cortical bone"}


def replace_material(metal_volume_m_ori, smooth_air=False, use_thresholding_segmentation=True):
    volume = metal_volume_m_ori

    # convert hu_values to materials
    if not use_thresholding_segmentation:
        materials = conv_hu_to_materials(volume)
    else:
        materials = conv_hu_to_materials_thresholding(volume)

    return materials


def load_dicom_CT(source_path=r"./*/*/", fixed_slice_thinckness=None, new_resolution=None, truncate=None, smooth_air=False, use_thresholding_segmentation=False, file_extension=".dcm"):
    #source_path += "*"+ file_extension
    source_path += "/*.dcm"
    files = np.array(glob.glob(source_path))
    one_slice = dicom.read_file(files[0])
    if hasattr(one_slice, "InstanceNumber"):
        sliceOrder = [dicom.read_file(
            curDCM).InstanceNumber for curDCM in files]
        files = files[np.argsort(sliceOrder).astype(np.int32)]
    else:
        sliceOrder = [dicom.read_file(
            curDCM).SliceLocation for curDCM in files]
        files = files[np.argsort(sliceOrder).astype(np.int32)]

    files = list(files)

    # Get ref file
    refDs = dicom.read_file(files[0])

    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    volume_size = [int(refDs.Rows), int(refDs.Columns), files.__len__()]

    if not hasattr(refDs, "SliceThickness"):
        print('Volume has no attribute Slice Thickness, please provide it manually!')
        print('using fixed slice thickness of:', fixed_slice_thinckness)
        voxel_size = [float(refDs.PixelSpacing[1]), float(
            refDs.PixelSpacing[0]), fixed_slice_thinckness]
    else:
        voxel_size = [float(refDs.PixelSpacing[1]), float(
            refDs.PixelSpacing[0]), float(refDs.SliceThickness)]

    # The array is sized based on 'PixelDims'
    volume = np.zeros(volume_size, dtype=np.float64)

    # loop through all the DICOM files
    for filenameDCM in files:
        # read the file
        ds = dicom.read_file(filenameDCM)
        # store the raw image data
        if files.index(filenameDCM) < volume.shape[2]:
            volume[:, :, files.index(filenameDCM)
                   ] = ds.pixel_array.astype(np.int32)

    # use intercept point
    if hasattr(refDs, "RescaleIntercept"):
        volume += int(refDs.RescaleIntercept)

    volume = np.moveaxis(volume, [0, 1, 2], [1, 0, 2]).copy()

    # truncate
    if truncate:
        volume = volume[truncate[0][0]:truncate[0][1], truncate[1]
                        [0]:truncate[1][1], truncate[2][0]:truncate[2][1]]

    # volume = np.flip(volume,2)
    # upsample Volume
    if new_resolution:
        volume, volume_size, voxel_size = upsample(
            volume, new_resolution, voxel_size)

    # convert hu_values to density
    densities = conv_hu_to_density(volume, smoothAir=smooth_air)

    # convert hu_values to materials
    if not use_thresholding_segmentation:
        materials = conv_hu_to_materials(volume)
    else:
        materials = conv_hu_to_materials_thresholding(volume)

    return volume, densities.astype(np.float32), materials, np.array(voxel_size, dtype=np.float32)


def load_dicom_metal(source_path=r"./*/*/", sortBy="SliceLocation", fixed_slice_thinkness=None, new_resolution=None, truncate=None, smooth_air=False, use_thresholding_segmentation=False, flip=False):
    # Metal Volume
    files = np.array(glob.glob(source_path))
    one_slice_body = dicom.read_file(files[0])
    if hasattr(one_slice_body, "InstanceNumber"):
        sliceOrder = [dicom.read_file(
            curDCM).InstanceNumber for curDCM in files]
        files = files[np.argsort(sliceOrder).astype(np.int8)]
    else:
        sliceOrder = [dicom.read_file(
            curDCM).SliceLocation for curDCM in files]
        files = files[np.argsort(sliceOrder).astype(np.int8)]

        files = list(files)

    # Get ref file
    refDs_body = dicom.read_file(files[0])

    volume_size = [int(refDs_body.Rows), int(refDs_body.Columns), int(
        refDs_body.NumberOfFrames)]  # The last number needs to be changed
    voxel_spacing = float(
        refDs_body.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[0])
    voxel_size = [voxel_spacing, voxel_spacing, voxel_spacing]

    # The array is sized based on 'PixelDims'
    volume = np.zeros(volume_size, dtype=np.float32)

    # loop through all the DICOM files
    ds = dicom.read_file(files[0])
    for index in range(int(refDs_body.NumberOfFrames)):
        # read the file
        # store the raw image data
        volume[:, :, index] = ds.pixel_array[index].astype(np.int8)

    volume = np.moveaxis(volume, [0, 1, 2], [1, 0, 2]).copy()

    # truncate
    if truncate:
        volume = volume[truncate[0][0]:truncate[0][1], truncate[1]
                        [0]:truncate[1][1], truncate[2][0]:truncate[2][1]]

    # upsample Volume
    if new_resolution:
        volume, volume_size, voxel_size = upsample(
            volume, new_resolution, voxel_size)

    # convert hu_values to density
    # apply density
    density_metal = 2
    densities = volume * density_metal

    # flip densities
    if flip:
        densities = np.flip(densities, 0)

    # convert hu_values to materials
    materials = {}
    materials["titanium"] = volume > 0

    return densities.astype(np.float32), materials, np.array(voxel_size, dtype=np.float32)


def upsample(volume, newResolution, voxelSize):
    upsampled_voxel_size = list(
        np.array(voxelSize) * np.array(volume.shape) / newResolution)
    upsampled_volume = resize(volume, newResolution, order=1, cval=-1000)
    return upsampled_volume, upsampled_voxel_size, upsampled_voxel_size


def conv_hu_to_density(hu_values, smoothAir=False):
    # Use two linear interpolations from data: (HU,g/cm^3)
    # -1000 0.00121000000000000
    # -98    0.930000000000000
    # -97    0.930486000000000
    # 14 1.03000000000000
    # 23 1.03100000000000
    # 100    1.11990000000000
    # 101    1.07620000000000
    # 1600   1.96420000000000
    # 3000   2.80000000000000
    # use fit1 for lower HU: density = 0.001029*HU + 1.030 (fit to first 4)
    # use fit2 for upper HU: density = 0.0005886*HU + 1.03 (fit to last 5)

    # set air densities
    if smoothAir:
        hu_values[hu_values <= -900] = -1000
    #hu_values[hu_values > 600] = 5000;
    densities = np.maximum(np.minimum(
        0.001029 * hu_values + 1.030, 0.0005886 * hu_values + 1.03), 0)
    return densities


def conv_hu_to_materials_thresholding(hu_values):
    # ranges taken from schneider and Buzug CT
    #     materials = np.zeros(hu_values.shape,dtype=np.int32)
    #
    # materials[hu_values <= -800] = 1
    #
    # # Lung
    # mask = (-800 < hu_values) * (hu_values <= -200)
    # materials[mask] = 9;
    #
    # # Fat
    # mask = (-200 < hu_values) * (hu_values <= -75)
    # materials[mask] = 6;
    #
    # # Connective Tissue
    # mask = (-75 < hu_values) * (hu_values <= -5)
    # materials[mask] = 8;
    #
    # # Water
    # mask = (-5 < hu_values) * (hu_values <= 5)
    # materials[mask] = 15;
    #
    # # Soft Tissue
    # mask = (5 < hu_values) * (hu_values <= 35)
    # materials[mask] = 3;
    #
    # # Muscle
    # mask = (35 < hu_values) * (hu_values <= 50)
    # materials[mask] = 2;
    #
    # # Blood
    # mask = (50 < hu_values) * (hu_values <= 60)
    # materials[mask] = 7;
    #
    # # Liver
    # mask = (60 < hu_values) * (hu_values <= 100)
    # materials[mask] = 13;
    #
    # # Bone Marrow
    # mask = (100 < hu_values) * (hu_values <= 400)
    # materials[mask] = 12;
    #
    # # Bone
    # mask = (400 < hu_values) * (hu_values <= 3000)
    # materials[mask] = 4;
    #
    # # Titanium
    # mask = 3000 < hu_values
    # materials[mask] = 5;

    # # Air
    # materials[hu_values <= -800] = 1
    #
    # # Soft Tissue
    # mask = (-800 < hu_values) * (hu_values <= 500)
    # materials[mask] = 2;
    #
    # # Bone
    # mask = 500 < hu_values
    # materials[mask] = 3;

    materials = {}
    # Air
    materials["air"] = hu_values <= -800

    # Soft Tissue
    materials["soft tissue"] = (-800 < hu_values) * (hu_values <= 350)

    # Bone
    materials["bone"] = (350 < hu_values)

    return materials


def conv_hu_to_materials(hu_values):
    segmentation_network = segmentation.SegmentationNet()
    materials = segmentation_network.segment(hu_values)

    return materials
