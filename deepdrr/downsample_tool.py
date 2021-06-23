import numpy as np
import pydicom as dicom
import glob
import copy
from skimage.transform import resize
import math

from . import segmentation


def downsample_tool(ori_CT_volume, CT_volume, CT_voxel_size, tool_volume, tool_voxel_size, CT_materials, tool_materials, origin, tool_origin):
    tool_size = np.shape(tool_volume)
    CT_size = np.shape(CT_volume)

    tool_volume_temp = copy.copy(tool_volume)
    tool_volume_temp_ori = copy.copy(tool_volume)
    CT_volume_temp = copy.copy(CT_volume)
    CT_volume_temp_ori = copy.copy(ori_CT_volume)
    CT_materials_temp = copy.copy(CT_materials)
    tool_materials_temp = copy.copy(tool_materials)

    # Calculate origin bias
    bias_x = tool_origin[0] - origin[0]
    bias_y = tool_origin[1] - origin[1]
    bias_z = tool_origin[2] - origin[2]

    [i, j, k] = np.nonzero(np.array(tool_volume_temp))
    x = np.ceil((bias_x + i*tool_voxel_size[0] + CT_size[0]*CT_voxel_size[0] /
                2 - tool_size[0]*tool_voxel_size[0]/2) / CT_voxel_size[0])
    x = np.minimum(x, CT_size[0])

    y = np.ceil((bias_y + j*tool_voxel_size[1] + CT_size[1]*CT_voxel_size[1] /
                2 - tool_size[1]*tool_voxel_size[1]/2) / CT_voxel_size[1])
    y = np.minimum(y, CT_size[1])

    z = np.ceil((bias_z + k*tool_voxel_size[2] + CT_size[2]*CT_voxel_size[2] /
                2 - tool_size[2]*tool_voxel_size[2]/2) / CT_voxel_size[2])
    z = np.minimum(z, CT_size[2])

    tool_volume_temp[i, j, k] = CT_volume_temp[x.astype(
        int), y.astype(int), z.astype(int)]
    tool_volume_temp_ori[i, j, k] = CT_volume_temp_ori[x.astype(
        int), y.astype(int), z.astype(int)]
    for mat in CT_materials_temp:
        tool_materials_temp[mat][i, j, k] = CT_materials_temp[mat][x.astype(
            int), y.astype(int), z.astype(int)]

    return tool_volume_temp, tool_volume_temp_ori, tool_materials_temp
