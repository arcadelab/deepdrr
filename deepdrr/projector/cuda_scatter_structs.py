import typing
import logging

logger = logging.getLogger(__name__)

try:
    import pycuda.driver as cuda
except ImportError:
    logger.critical("pycuda unavailable")

from .plane_surface import PlaneSurface
from .rita import RITA
from .mcgpu_compton_data import MAX_NSHELLS
from .mcgpu_mfp_data import MFP_DATA
import numpy as np


class CudaPlaneSurfaceStruct:
    MEMSIZE = 80  # from using sizeof(plane_surface_t)

    def __init__(self, psurf: PlaneSurface, struct_gpu_ptr):
        """Copies the PlaneSurface to memory location 'struct_gpu_ptr' on the GPU 
        """
        self.n = psurf.plane_vector[0:3].astype(np.float32)
        self.d = np.float32(psurf.plane_vector[3])
        cuda.memcpy_htod(int(struct_gpu_ptr), self.n)
        cuda.memcpy_htod(int(struct_gpu_ptr) + (3 * 4), self.d)

        self.ori = np.array(psurf.surface_origin).astype(np.float32)
        cuda.memcpy_htod(int(struct_gpu_ptr) + (4 * 4), self.ori)

        self.b1 = np.array(psurf.basis_1).astype(np.float32)
        self.b2 = np.array(psurf.basis_2).astype(np.float32)
        cuda.memcpy_htod(int(struct_gpu_ptr) + (7 * 4), self.b1)
        cuda.memcpy_htod(int(struct_gpu_ptr) + (10 * 4), self.b2)

        self.bound1 = np.ascontiguousarray(np.array(psurf.bounds[0, :])).astype(
            np.float32
        )
        self.bound2 = np.ascontiguousarray(np.array(psurf.bounds[1, :])).astype(
            np.float32
        )
        cuda.memcpy_htod(int(struct_gpu_ptr) + (13 * 4), self.bound1)
        cuda.memcpy_htod(int(struct_gpu_ptr) + (15 * 4), self.bound2)

        self.orthogonal = np.int32(psurf.orthogonal)
        cuda.memcpy_htod(int(struct_gpu_ptr) + (17 * 4), self.orthogonal)


MAX_RITA_N_PTS = 128
MAX_MFP_BINS = 25005
RAYLEIGH_FF_COLUMN = 5

class CudaRayleighStruct:
    MEMSIZE = 104120  # from using sizeof(rayleigh_data_t)

    def __init__(self, rita_obj: RITA, mat_name: str, struct_gpu_ptr):
        """Copies the RITA object to memory location 'struct_gpu_ptr' on the GPU
        """
        self.x = rita_obj.x_arr.copy().astype(np.float64)
        self.y = rita_obj.y_arr.copy().astype(np.float64)
        self.a = rita_obj.a_arr.copy().astype(np.float64)
        self.b = rita_obj.b_arr.copy().astype(np.float64)

        cuda.memcpy_htod(int(struct_gpu_ptr) + 0 * (8 * MAX_RITA_N_PTS), self.x)
        cuda.memcpy_htod(int(struct_gpu_ptr) + 1 * (8 * MAX_RITA_N_PTS), self.y)
        cuda.memcpy_htod(int(struct_gpu_ptr) + 2 * (8 * MAX_RITA_N_PTS), self.a)
        cuda.memcpy_htod(int(struct_gpu_ptr) + 3 * (8 * MAX_RITA_N_PTS), self.b)
        
        self.n_gridpts = np.int32(rita_obj.n_grid_points)
        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 * (8 * MAX_RITA_N_PTS), self.n_gridpts)
        
        self.pmax = np.array([MFP_DATA[mat_name][i, RAYLEIGH_FF_COLUMN] for i in range(MFP_DATA[mat_name].shape[0])]).astype(np.float32)
        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 * (8 * MAX_RITA_N_PTS) + 4, self.pmax)


class CudaComptonStruct:
    MEMSIZE = 364  # from using sizeof(compton_data_t)

    def __init__(self, compton_arr: np.ndarray, struct_gpu_ptr):
        """Copies the Compton data (see mcgpu_compton_data.py) to memory location 'struct_gpu_ptr' on the GPU
        """
        self.nshells = np.int32(compton_arr.shape[0])
        cuda.memcpy_htod(int(struct_gpu_ptr), self.nshells)

        self.f = np.ascontiguousarray(compton_arr[:, 0].copy()).astype(np.float32)
        self.ui = np.ascontiguousarray(compton_arr[:, 1].copy()).astype(np.float32)
        self.jmc = np.ascontiguousarray(compton_arr[:, 2].copy()).astype(np.float32)

        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 0 * (4 * MAX_NSHELLS), self.f)
        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 1 * (4 * MAX_NSHELLS), self.ui)
        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 2 * (4 * MAX_NSHELLS), self.jmc)


class CudaMatMfpStruct:
    MEMSIZE = 400084  # from using sizeof(mat_mfp_data_t)

    def __init__(self, mfp_arr: np.ndarray, struct_gpu_ptr):
        """Copies the MFP data (see mcgpu_mfp_data.py) to memory location 'struct_gpu_ptr'
        """
        self.n_bins = np.int32(mfp_arr.shape[0])
        cuda.memcpy_htod(int(struct_gpu_ptr), self.n_bins)

        self.energy = np.ascontiguousarray(mfp_arr[:, 0].copy()).astype(np.float32)
        self.mfp_Ra = np.ascontiguousarray(mfp_arr[:, 1].copy()).astype(np.float32)
        self.mfp_Co = np.ascontiguousarray(mfp_arr[:, 2].copy()).astype(np.float32)
        self.mfp_Tot = np.ascontiguousarray(mfp_arr[:, 4].copy()).astype(np.float32)

        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 0 * (4 * MAX_MFP_BINS), self.energy)
        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 1 * (4 * MAX_MFP_BINS), self.mfp_Ra)
        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 2 * (4 * MAX_MFP_BINS), self.mfp_Co)
        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 3 * (4 * MAX_MFP_BINS), self.mfp_Tot)


class CudaWoodcockStruct:
    MEMSIZE = 200044  # from using sizeof(wc_mfp_data_t)

    def __init__(self, mfp_arr: np.ndarray, struct_gpu_ptr):
        """Copies the Woodcock MFP data (see scatter.py:make_woodcock_mfp(...)) to memory location 'struct_gpu_ptr'
        """
        self.n_bins = np.int32(mfp_arr.shape[0])
        cuda.memcpy_htod(int(struct_gpu_ptr), self.n_bins)

        self.energy = np.ascontiguousarray(mfp_arr[:, 0].copy()).astype(np.float32)
        self.mfp_wc = np.ascontiguousarray(mfp_arr[:, 1].copy()).astype(np.float32)

        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 0 * (4 * MAX_MFP_BINS), self.energy)
        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 1 * (4 * MAX_MFP_BINS), self.mfp_wc)
