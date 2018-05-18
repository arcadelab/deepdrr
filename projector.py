import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.compiler import SourceModule
import numpy as np
import os,inspect


class ForwardProjector():
    def __init__(self,volume,segmentation,voxelsize,origin=[0.0,0.0,0.0],stepsize = 0.1,mode="linear"):
        #generate kernels
        self.mod = self.generateKernelModuleProjector()
        self.projKernel = self.mod.get_function("projectKernel")
        self.volumesize = volume.shape
        self.volume = np.moveaxis(volume, [0, 1, 2], [2, 1, 0]).copy()
        self.segmentation = np.moveaxis(segmentation.astype(np.float32), [0, 1, 2], [2, 1, 0]).copy()
        # print("done swap")
        self.volume_gpu = cuda.np_to_array(self.volume, order='C')
        self.texref_volume = self.mod.get_texref("tex_density")
        cuda.bind_array_to_texref(self.volume_gpu, self.texref_volume)
        self.segmentation_gpu = cuda.np_to_array(self.segmentation, order='C')
        self.texref_segmentation = self.mod.get_texref("tex_segmentation")
        cuda.bind_array_to_texref(self.segmentation_gpu, self.texref_segmentation)
        if mode =="linear":
            self.texref_volume.set_filter_mode(cuda.filter_mode.LINEAR)
            self.texref_segmentation.set_filter_mode(cuda.filter_mode.LINEAR)
        self.voxelsize = voxelsize
        self.stepsize = np.float32(stepsize)
        self.origin = origin
        self.initialized = False
        print("initialized projector")

    def initialize_sensor(self, proj_width, proj_height):
        self.proj_width = np.int32(proj_width)
        self.proj_height = np.int32(proj_height)
        self.initialized = True

    def setOrigin(self,origin):
        self.origin = origin

    def generateKernelModuleProjector(self):
        #path to files for cubic interpolation (folder cubic in DeepDRR)
        bicubic_path = os.path.join(os.path.dirname(os.path.abspath(inspect.stack()[0][1])),"cubic")
        print(bicubic_path)
        mod = SourceModule("""
            #include <stdio.h>
            #include <cubicTex3D.cu>
            
            texture<float, 3, cudaReadModeElementType> tex_density;
            texture<float, 3, cudaReadModeElementType> tex_segmentation;
            extern "C" {
              __global__  void projectKernel(int proj_width, int proj_height, float stepsize, float gVolumeEdgeMinPointX, float gVolumeEdgeMinPointY, float gVolumeEdgeMinPointZ, float gVolumeEdgeMaxPointX, float gVolumeEdgeMaxPointY, float gVolumeEdgeMaxPointZ, float gVoxelElementSizeX, float gVoxelElementSizeY, float gVoxelElementSizeZ, float sx, float sy, float sz, float* gInvARmatrix, float* pixel, int offsetW, int offsetH)
            {
            int udx = threadIdx.x + (blockIdx.x + offsetW) * blockDim.x;
            int vdx = threadIdx.y + (blockIdx.y + offsetH) * blockDim.y;
            int idx = udx*proj_height + vdx;
            
            if (udx >= proj_width || vdx >= proj_height) {
        	    return;}
            float u = (float) udx + 0.5;
        	float v = (float) vdx + 0.5;

                // compute ray direction
                float rx = gInvARmatrix[2] + v * gInvARmatrix[1] + u * gInvARmatrix[0];
                float ry = gInvARmatrix[5] + v * gInvARmatrix[4] + u * gInvARmatrix[3];
                float rz = gInvARmatrix[8] + v * gInvARmatrix[7] + u * gInvARmatrix[6];

                // normalize ray direction float
                float normFactor = 1.0f / (sqrt((rx * rx) + (ry * ry) + (rz * rz)));
                rx *= normFactor;
                ry *= normFactor;
                rz *= normFactor;

                //calculate projections
                // Step 1: compute alpha value at entry and exit point of the volume
            float minAlpha, maxAlpha;
        	minAlpha = 0;
        	maxAlpha = INFINITY;

            if (0.0f != rx)
            {
                float reci = 1.0f / rx;
                float alpha0 = (gVolumeEdgeMinPointX - sx) * reci;
                float alpha1 = (gVolumeEdgeMaxPointX - sx) * reci;
                minAlpha = fmin(alpha0, alpha1);
                maxAlpha = fmax(alpha0, alpha1);
            }
            else if (gVolumeEdgeMinPointX > sx || sx > gVolumeEdgeMaxPointX)
            {
                return;
            }

            if (0.0f != ry)
            {
                float reci = 1.0f / ry;
                float alpha0 = (gVolumeEdgeMinPointY - sy) * reci;
                float alpha1 = (gVolumeEdgeMaxPointY - sy) * reci;
                minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
                maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
            }
            else if (gVolumeEdgeMinPointY > sy || sy > gVolumeEdgeMaxPointY)
            {
                return;
            }

            if (0.0f != rz)
            {
                float reci = 1.0f / rz;
                float alpha0 = (gVolumeEdgeMinPointZ - sz) * reci;
                float alpha1 = (gVolumeEdgeMaxPointZ - sz) * reci;
                minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
                maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
            }
            else if (gVolumeEdgeMinPointZ > sz || sz > gVolumeEdgeMaxPointZ)
            {
                return;
            }

            // we start not at the exact entry point 
            // => we can be sure to be inside the volume
            //minAlpha += stepsize * 0.5f;

            // Step 2: Cast ray if it intersects the volume

            // Trapezoidal rule (interpolating function = piecewise linear func)
            float px, py, pz;

            // Entrance boundary
            // In CUDA, voxel centers are located at (xx.5, xx.5, xx.5),
            //  whereas, SwVolume has voxel centers at integers.
            // For the initial interpolated value, only a half stepsize is
            //  considered in the computation.
            if (minAlpha < maxAlpha) {
                px = sx + minAlpha * rx;
                py = sy + minAlpha * ry;
                pz = sz + minAlpha * rz;
                pixel[idx] += 0.5 * tex3D(tex_density, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ) * round(cubicTex3D(tex_segmentation, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ));
                minAlpha += stepsize;
            }

            // Mid segments
            while (minAlpha < maxAlpha)
            {
                px = sx + minAlpha * rx;
                py = sy + minAlpha * ry;
                pz = sz + minAlpha * rz;
                pixel[idx] += tex3D(tex_density, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ) * round(cubicTex3D(tex_segmentation, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ));
                minAlpha += stepsize;
            }
            // Scaling by stepsize;
            pixel[idx] *= stepsize;

            // Last segment of the line
            if (pixel[idx] > 0.0f ) {
                pixel[idx] -= 0.5 * stepsize * tex3D(tex_density, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ) * round(cubicTex3D(tex_segmentation, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ));
                minAlpha -= stepsize;
                float lastStepsize = maxAlpha - minAlpha;
                pixel[idx] += 0.5 * lastStepsize * tex3D(tex_density, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ) * round(cubicTex3D(tex_segmentation, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ));

                px = sx + maxAlpha * rx;
                py = sy + maxAlpha * ry;
                pz = sz + maxAlpha * rz;
                // The last segment of the line integral takes care of the
                // varying length.
                pixel[idx] += 0.5 * lastStepsize * tex3D(tex_density, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ) * round(cubicTex3D(tex_segmentation, px + 0.5, py + 0.5, pz - gVolumeEdgeMinPointZ));
            }
            
            // normalize pixel value to world coordinate system units
            pixel[idx] *= sqrt((rx * gVoxelElementSizeX)*(rx * gVoxelElementSizeX) + (ry * gVoxelElementSizeY)*(ry * gVoxelElementSizeY) + (rz * gVoxelElementSizeZ)*(rz * gVoxelElementSizeZ));
                
            return;
              }}
              """, include_dirs=[bicubic_path], no_extern_c=True)
        return mod


    def project(self, proj_mat, threads = 8, max_blockind = 1024):
        if not self.initialized:
            print("Projector is not initialized")
            return

        inv_ar_mat, source_point = proj_mat.get_conanical_proj_matrix(voxel_size=self.voxelsize, volume_size=self.volumesize, origin_shift=self.origin)

        can_proj_matrix = inv_ar_mat.astype(np.float32)
        pixel_array = np.zeros((self.proj_width, self.proj_height)).astype(np.float32)
        sourcex = source_point[0]
        sourcey = source_point[1]
        sourcez = source_point[2]
        g_volume_edge_min_point_x = np.float32(-0.5)
        g_volume_edge_min_point_y = np.float32(-0.5)
        g_volume_edge_min_point_z = np.float32(-0.5)
        g_volume_edge_max_point_x = np.float32(self.volumesize[0] - 0.5)
        g_volume_edge_max_point_y = np.float32(self.volumesize[1] - 0.5)
        g_volume_edge_max_point_z = np.float32(self.volumesize[2] - 0.5)
        g_voxel_element_size_x = self.voxelsize[0]
        g_voxel_element_size_y = self.voxelsize[1]
        g_voxel_element_size_z = self.voxelsize[2]

        #copy to gpu
        proj_matrix_gpu = cuda.mem_alloc(can_proj_matrix.nbytes)
        cuda.memcpy_htod(proj_matrix_gpu, can_proj_matrix)
        pixel_array_gpu = cuda.mem_alloc(pixel_array.nbytes)
        cuda.memcpy_htod(pixel_array_gpu, pixel_array)

        #calculate required blocks
        #threads = 8
        blocks_w = np.int(np.ceil(self.proj_width / threads))
        blocks_h = np.int(np.ceil(self.proj_height / threads))
        print("running:", blocks_w, "x", blocks_h, "blocks with ", threads, "x", threads, "threads")

        if blocks_w <= max_blockind and blocks_h <= max_blockind:
            #run kernel
            offset_w = np.int32(0)
            offset_h = np.int32(0)
            self.projKernel(self.proj_width, self.proj_height, self.stepsize, g_volume_edge_min_point_x, g_volume_edge_min_point_y, g_volume_edge_min_point_z,
                            g_volume_edge_max_point_x, g_volume_edge_max_point_y, g_volume_edge_max_point_z, g_voxel_element_size_x, g_voxel_element_size_y, g_voxel_element_size_z, sourcex, sourcey, sourcez,
                            proj_matrix_gpu, pixel_array_gpu, offset_w, offset_h, block=(8, 8, 1), grid=(blocks_w, blocks_h))
        else:
            print("running kernel patchwise")
            for w in range(0, (blocks_w-1)//max_blockind+1):
                for h in range(0, (blocks_h-1) // max_blockind+1):
                    offset_w = np.int32(w * max_blockind)
                    offset_h = np.int32(h * max_blockind)
                    # print(offset_w, offset_h)
                    self.projKernel(self.proj_width, self.proj_height, self.stepsize, g_volume_edge_min_point_x,
                                    g_volume_edge_min_point_y, g_volume_edge_min_point_z,
                                    g_volume_edge_max_point_x, g_volume_edge_max_point_y, g_volume_edge_max_point_z,
                                    g_voxel_element_size_x, g_voxel_element_size_y, g_voxel_element_size_z, sourcex, sourcey,
                                    sourcez,
                                    proj_matrix_gpu, pixel_array_gpu, offset_w, offset_h, block=(8, 8, 1),
                                    grid=(max_blockind, max_blockind))
                    context.synchronize()

        #context.synchronize()
        cuda.memcpy_dtoh(pixel_array, pixel_array_gpu)

        pixel_array = np.swapaxes(pixel_array, 0, 1)
        #normalize to cm
        return pixel_array/10

def generate_projections(projection_matrices, density, materials, origin, voxel_size, sensor_width, sensor_height, mode ="linear", max_blockind = 1024, threads = 8):
    # projections = np.zeros((projection_matrices.__len__(), sensor_width, sensor_height, materials.__len__()),dtype=np.float32)
    projections = {}

    for mat in materials:
        print("projecting:", mat)
        curr_projections = np.zeros((projection_matrices.__len__(), sensor_height, sensor_width), dtype=np.float32)
        projector = ForwardProjector(density, materials[mat], voxel_size, origin=origin, mode = mode)
        projector.initialize_sensor(sensor_width, sensor_height)
        for i, proj_mat in enumerate(projection_matrices):
            curr_projections[i, :, :] =projector.project(proj_mat, max_blockind= max_blockind, threads=threads)
        projections[mat] = curr_projections
        #clean projector to free Memory on GPU
        projector = None

    return projections