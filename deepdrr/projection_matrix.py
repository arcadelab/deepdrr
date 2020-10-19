"""This file is deprecated."""

print(f'this file is deprecated: {__file__}')


import numpy as np
import os


class ProjMatrix():
    def __init__(self, R, K, t):
        self.R = np.array(R, dtype=np.float32)
        self.t = np.array(t, dtype=np.float32)
        self.K = np.array(K, dtype=np.float32)
        self.P = np.matmul(self.K, np.concatenate((self.R, np.expand_dims(self.t, 1)), axis=1))
        self.rtk_inv = np.matmul(np.transpose(self.R), np.linalg.inv(self.K))

    def get_rtk_inv(self):
        return self.rtk_inv

    def get_projection(self):
        return self.P

    def get_camera_ceter(self):
        return -np.matmul(np.transpose(self.R), self.t)

    def get_principle_axis(self):
        axis = self.R[2, :] / self.K[2, 2]
        return axis

    def get_conanical_proj_matrix(self, voxel_size, volume_size, origin_shift):
        inv_voxel_scale = np.zeros([3, 3])
        inv_voxel_scale[0][0] = 1 / voxel_size[0]
        inv_voxel_scale[1][1] = 1 / voxel_size[1]
        inv_voxel_scale[2][2] = 1 / voxel_size[2]
        inv_ar = np.matmul(inv_voxel_scale, self.rtk_inv)

        source_point = np.zeros((3, 1), dtype=np.float32)
        camera_ceter = - self.get_camera_ceter()
        source_point[0] = -(-0.5 * (volume_size[0] - 1.0) + origin_shift[0] * inv_voxel_scale[0, 0] + inv_voxel_scale[0, 0] * camera_ceter[0])
        source_point[1] = -(-0.5 * (volume_size[1] - 1.0) + origin_shift[1] * inv_voxel_scale[1, 1] + inv_voxel_scale[1, 1] * camera_ceter[1])
        source_point[2] = -(-0.5 * (volume_size[2] - 1.0) + origin_shift[2] * inv_voxel_scale[2, 2] + inv_voxel_scale[2, 2] * camera_ceter[2])
        return inv_ar, source_point


def generate_projection_matrices_from_values(source_to_detector_distance, pixel_dim_x, pixel_dim_y, sensor_size_x, sensor_size_y, isocenter_distance, phi_list, theta_list, rho_list=None, offset_list=None):
    number_of_projections = phi_list.__len__()
    print("generating ", number_of_projections, "matrices")
    matrices = []
    if not rho_list:
        rho_list = [0 for i in range(0, number_of_projections)]
    if not offset_list:
        offset_list = [np.zeros(3) for i in range(0, number_of_projections)]
    K = generate_camera_intrinsics(source_to_detector_distance, pixel_dim_x, pixel_dim_y, sensor_size_x, sensor_size_y)
    for phi, theta, rho, offset in zip(phi_list, theta_list, rho_list, offset_list):
        R = generat_rotation_from_angles(phi, theta, rho)
        t = generate_translation(isocenter_distance, offset[0], offset[1], offset[2])
        matrices.append(ProjMatrix(R, K, t))
    return matrices


def generate_camera_intrinsics(source_to_detector_distance, pixel_dim_x, pixel_dim_y, sensor_size_x, sensor_size_y):
    K = np.zeros([3, 3])
    K[0, 0] = source_to_detector_distance / pixel_dim_x
    K[1, 1] = source_to_detector_distance / pixel_dim_y
    K[0, 2] = sensor_size_x / 2
    K[1, 2] = sensor_size_y / 2
    K[2, 2] = 1.0
    return K


def generate_translation(isocenter_distance, offsetx=0, offsety=0, offsetz=0):
    t = np.array([offsetx, offsety, isocenter_distance + offsetz])
    return t


def generat_rotation_from_angles(phi, theta, rho=0):
    # rotation around phi and theta
    sin_p = np.sin(phi)
    neg_cos_p = -np.cos(phi)
    z = 0
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    omc = 1 - cos_t
    R = np.array([[sin_p * sin_p * omc + cos_t, sin_p * neg_cos_p * omc - z * sin_t, sin_p * z * omc + neg_cos_p * sin_t],
                  [sin_p * neg_cos_p * omc + z * sin_t, neg_cos_p * neg_cos_p * omc + cos_t, neg_cos_p * z * omc - sin_p * sin_t],
                  [sin_p * z * omc - neg_cos_p * sin_t, neg_cos_p * z * omc + sin_p * sin_t, z * z * omc + cos_t]])
    # rotation around detector priniciple axis
    rho = -phi + np.pi * 0.5 + rho
    R_principle = np.array([[np.cos(rho), -np.sin(rho), 0],
                            [np.sin(rho), np.cos(rho), 0],
                            [0, 0, 1]])
    R = np.matmul(R_principle, R)

    return R


def read_matrices_from_file(path, lim=100000000):
    R_in = open(os.path.join(path, "R.txt")).read().split("\n")
    t_in = open(os.path.join(path, "T.txt")).read().split("\n")
    K_in = open(os.path.join(path, "K.txt")).read().split("\n")
    projs = []
    for i in range(R_in.__len__()):
        if R_in[i] == "" or i > lim:
            return projs
        R = np.array(list(map(float, R_in[i].split(" ")[0:9])))
        R.shape = (3, 3)
        K = np.array(list(map(float, K_in[i].split(" ")[0:9])))
        K.shape = (3, 3)
        t = np.array(list(map(float, t_in[i].split(" ")[0:3])))
        t.shape = (3)
        projs.append(ProjMatrix(R, K, t))
    return projs


def generate_uniform_angels(min_theta, max_theta, min_phi, max_phi, spacing_theta, spacing_phi):
    thetas = np.array(np.arange(min_theta, max_theta + spacing_theta / 2, step=spacing_theta)) / 180 * np.pi
    num_thetas = thetas.__len__()
    phis = np.array(np.arange(min_phi, max_phi, step=spacing_phi)) / 180 * np.pi
    num_phis = phis.__len__()
    thetas = np.tile(thetas, num_phis)
    phis = phis.repeat(num_thetas, 0)
    return thetas, phis
