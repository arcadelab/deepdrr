import typing

try:
    import pycuda.driver as cuda
except ImportError:
    logging.warning('pycuda unavailable')

from .plane_surface import PlaneSurface
from .rita import RITA
from .mcgpu_compton_data import MAX_NSHELLS
import numpy as np

class CudaPlaneSurfaceStruct:
    MEMSIZE = 72 # 18 * 4
    def __init__(self, psurf: PlaneSurface, struct_gpu_ptr):
        """Copies the PlaneSurface to memory location 'struct_gpu_ptr' on the GPU 
        """
        self.n = psurf.plane_vector[0:3].astype(np.float32)
        self.d = np.float32(psurf.plane_vector[4])
        cuda.memcpy_htod(int(struct_gpu_ptr), np.getbuffer(self.n))
        cuda.memcpy_htod(int(struct_gpu_ptr) + (3 * 4), np.getbuffer(self.d))

        self.ori = np.array(psurf.surface_origin).astype(np.float32)
        print(f"CudaPlaneSurfaceStruct.ori: {self.ori}")
        cuda.memcpy_htod(int(struct_gpu_ptr) + (4 * 4), np.getbuffer(self.ori))

        self.b1 = np.array(psurf.basis_1).astype(np.float32)
        self.b2 = np.array(psurf.basis_2).astype(np.float32)
        print(f"CudaPlaneSurfaceStruct.b1: {self.b1}")
        print(f"CudaPlaneSurfaceStruct.b2: {self.b2}")
        cuda.memcpy_htod(int(struct_gpu_ptr) + (7 * 4), np.getbuffer(self.b1))
        cuda.memcpy_htod(int(struct_gpu_ptr) + (10 * 4), np.getbuffer(self.b2))

        self.bound1 = np.ascontiguousarray(np.array(psurf.bounds[0, :])).astype(np.float32)
        self.bound2 = np.ascontiguousarray(np.array(psurf.bounds[1, :])).astype(np.float32)
        print(f"CudaPlaneSurfaceStruct.bound1: {self.bound1}")
        print(f"CudaPlaneSurfaceStruct.bound1: {self.bound2}")
        cuda.memcpy_htod(int(struct_gpu_ptr) + (13 * 4), np.getbuffer(self.bound1))
        cuda.memcpy_htod(int(struct_gpu_ptr) + (15 * 4), np.getbuffer(self.bound2))

        self.orthogonal = np.int32(psurf.orthogonal)
        print(f"CudaPlaneSurfaceStruct.orthogonal: {self.orthogonal}")
        cuda.memcpy_htod(int(struct_gpu_ptr) + (17 * 4), np.getbuffer(self.orthogonal))

MAX_RITA_N_PTS = 128
class CudaRitaStruct:
    MEMSIZE = 4 + (128 * 8) * 4
    def __init__(self, rita_obj: RITA, struct_gpu_ptr):
        """Copies the RITA object to memory location 'struct_gpu_ptr' on the GPU
        """
        self.n_gridpts = np.int32(rita_obj.n_grid_points)
        cuda.memcpy_htod(int(struct_gpu_ptr), np.getbuffer(self.n_gridpts))

        self.x = rita_obj.x_arr.copy().astype(np.float64)
        self.y = rita_obj.y_arr.copy().astype(np.float64)
        self.a = rita_obj.a_arr.copy().astype(np.float64)
        self.b = rita_obj.b_arr.copy().astype(np.float64)

        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 0 * (8 * MAX_RITA_N_PTS), np.getbuffer(self.x))
        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 1 * (8 * MAX_RITA_N_PTS), np.getbuffer(self.y))
        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 2 * (8 * MAX_RITA_N_PTS), np.getbuffer(self.a))
        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 3 * (8 * MAX_RITA_N_PTS), np.getbuffer(self.b))
        
class CudaComptonStruct:
    MEMSIZE = 4 + (4 * MAX_NSHELLS) * 3
    def __init__(self, compton_arr: np.ndarray, struct_gpu_ptr):
        """Copies the Compton data (see mcgpu_compton_data.py) to memory location 'struct_gpu_ptr' on the GPU
        """
        self.nshells = np.int32(compton_arr.shape[0])
        cuda.memcpy_htod(int(struct_gpu_ptr), np.getbuffer(self.nshells))

        self.f = np.ascontiguousarray(compton_arr[:, 0].copy()).astype(np.float32)
        self.ui = np.ascontiguousarray(compton_arr[:, 1].copy()).astype(np.float32)
        self.jmc = np.ascontiguousarray(compton_arr[:, 2].copy()).astype(np.float32)

        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 0 * (4 * MAX_NSHELLS), np.getbuffer(self.f))
        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 1 * (4 * MAX_NSHELLS), np.getbuffer(self.ui))
        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 2 * (4 * MAX_NSHELLS), np.getbuffer(self.jmc))

MAX_MFP_BINS = 25005
class CudaMatMfpStruct:
    MEMSIZE = 4 + (4 * MAX_MFP_BINS) * 4
    def __init__(self, mfp_arr: np.ndarray, struct_gpu_ptr):
        """Copies the MFP data (see mcgpu_mfp_data.py) to memory location 'struct_gpu_ptr'
        """
        self.n_bins = np.int32(mfp_arr.shape[0])
        cuda.memcpy_htod(int(struct_gpu_ptr), np.getbuffer(self.n_bins))

        self.energy = np.ascontiguousarray(mfp_arr[:, 0].copy()).astype(np.float32)
        self.mfp_Ra = np.ascontiguousarray(mfp_arr[:, 1].copy()).astype(np.float32)
        self.mfp_Co = np.ascontiguousarray(mfp_arr[:, 2].copy()).astype(np.float32)
        self.mfp_Tot = np.ascontiguousarray(mfp_arr[:, 4].copy()).astype(np.float32)

        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 0 * (4 * MAX_MFP_BINS), np.getbuffer(self.energy))
        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 1 * (4 * MAX_MFP_BINS), np.getbuffer(self.mfp_Ra))
        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 2 * (4 * MAX_MFP_BINS), np.getbuffer(self.mfp_Co))
        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 3 * (4 * MAX_MFP_BINS), np.getbuffer(self.mfp_Tot))

class CudaWoodcockStruct:
    MEMSIZE = 4 + (4 * MAX_MFP_BINS) * 2
    def __init__(self, mfp_arr: np.ndarray, struct_gpu_ptr):
        """Copies the Woodcock MFP data (see scatter.py:make_woodcock_mfp(...)) to memory location 'struct_gpu_ptr'
        """
        self.n_bins = np.int32(mfp_arr.shape[0])
        cuda.memcpy_htod(int(struct_gpu_ptr), np.getbuffer(self.n_bins))

        self.energy = np.ascontiguousarray(mfp_arr[:, 0].copy()).astype(np.float32)
        self.mfp_wc = np.ascontiguousarray(mfp_arr[:, 1].copy()).astype(np.float32)

        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 0 * (4 * MAX_MFP_BINS), np.getbuffer(self.energy))
        cuda.memcpy_htod(int(struct_gpu_ptr) + 4 + 1 * (4 * MAX_MFP_BINS), np.getbuffer(self.mfp_wc))
