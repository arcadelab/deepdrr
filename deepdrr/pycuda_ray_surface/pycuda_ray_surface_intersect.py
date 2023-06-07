#============================================================================
#Copyright (c) 2023, Raymond Leung
#All rights reserved.
#
#This source code is licensed under the BSD-3-clause license.
#See the LICENSE.md file in the root directory for details.
#
#Purpose: Provide the PyCudaRSI class and API for the PyCUDA implementation
#============================================================================
import importlib
import numpy as np
import os
import time

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from .diagnostic_utils import display_node_contents
from .diagnostic_graphics import bvh_graphviz, bvh_spatial

from .pycuda_source import get_cuda_template

default_paths = {'PATH': '/usr/local/cuda-11.2/bin',
                 'LD_LIBRARY_PATH': '/usr/local/cuda-11.2/lib64',
                 'CUDA_INC_DIR': '/usr/local/cuda-11.2/include'}


class PyCudaRSI(object):
    def __init__(self, n_rays, params=None, max_intersections=32):
        # - The constant parameters QUANT_LEVELS and LARGE_POS_VALUE
        #   represent design choices that ought to be fixed.
        # - PATH, LD_LIBRARY_PATH, CUDA_INC_DIR are environment variables
        #   that should be overridden using the `params` dictionary or
        #   set externally using the `export` command.
        # - If USE_EXTRA_BVH_FIELDS is True, additional fields will be
        #   added to the BVHNode structure to provide tree debugging info.
        #   - This data may be inspected using `display_node_contents`
        #     from diagnostic_utils.py given a numpy.int32 array
        #     representation of the BVHNode bytestream copied from
        #     device memory [see "OPTION: Check BVH integrity"]
        #   - USE_EXTRA_BVH_FIELDS may be disabled (set to False) to reduce
        #     memory footprint. The extra fields (e.g. parent and child
        #     pointers) are not needed for the algorithm to work correctly.
        # - TOLERANCE may be used to overwrite the default zero threshold
        #   in the Moller-Trumbore ray-triangle intersection algorithm.
        import pycuda.autoinit

        self.max_intersections = max_intersections

        if params is None:
            params = {}

        self.params = {}
        self.params['QUANT_LEVELS'] = params.get('QUANT_LEVELS', (1 << 21) - 1)
        self.params['USE_EXTRA_BVH_FIELDS'] = params.get('USE_EXTRA_BVH_FIELDS', False)
        self.params['USE_DOUBLE_PRECISION_MOLLER'] = params.get('USE_DOUBLE_PRECISION_MOLLER', False)
        self.quiet = params.get('QUIET', True)
        self.tolerance = '%.6g' % params.get('TOLERANCE', 0.0)
        # self.tolerance = '%.6g' % params.get('TOLERANCE', 0.000000001)
        # self.tolerance = '%.6g' % params.get('TOLERANCE', 0.0000001)
        # Check environment variables
        for k, v in default_paths.items():
            location = params.get(k, default_paths[k])
            if location not in os.environ.get(k, ''):
                os.environ[k] = '{}:{}'.format(location, os.environ.get(k, ''))
                if not self.quiet:
                    print('{}={}'.format(k, os.environ[k]))
        # Compile CUDA code
        # - Import module dynamically using user-configured module name.
        # - By default, 'pycuda_source' refers to a presumably stable version of the code
        # - User can override this with a version under development (say, 'pycuda_dev')
        #   to test for code changes and easily compare their results.
        # name = params.get('CUDA_SOURCE_MODULE', 'pycuda_source')
        # module = importlib.import_module(name)

        subst_dict = {
            'MORTON': 'uint64_t', 'COORD': 'unsigned int', 'TOLERANCE': self.tolerance,
            'MOLLER_DOUBLE_PRECISION_DIRECTIVE': '#define COMPILE_DOUBLE_PRECISION_MOLLER 1' \
                if self.params['USE_DOUBLE_PRECISION_MOLLER'] else '',
            'BVH_PREPROCESSOR_DIRECTIVE': '#define COMPILE_NON_ESSENTIAL 1' \
                if self.params['USE_EXTRA_BVH_FIELDS'] else ''
        }
        self.module = SourceModule(self.fill(get_cuda_template(), subst_dict))
        self.perform_bindings_()

    
        self.n_rays = n_rays
        self.d_szQuery = cuda.mem_alloc(np.ones(1, dtype=np.int32).nbytes)
        get_ = lambda x : self.struct_size(x, self.d_szQuery)

        self.h_interceptCounts = np.zeros(self.n_rays, dtype=np.int32)
        self.d_rayBox = cuda.mem_alloc(self.n_rays * get_(self.bytes_in_AABB))
        self.d_interceptCounts = cuda.mem_alloc(self.h_interceptCounts.nbytes)

        self.d_raysFrom = cuda.mem_alloc(self.n_rays * 3 * np.float32().nbytes)
        self.d_raysTo = cuda.mem_alloc(self.n_rays * 3 * np.float32().nbytes)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def struct_size(self, cuda_szQuery, d_answer):
        # Use cuda_szQuery API to convey C struct size to Python
        # This is known at compile time and padding may be introduced.
        h_answer = np.zeros(1, dtype=np.int32)
        cuda_szQuery(d_answer, block=(1,1,1), grid=(1,1))
        cuda.memcpy_dtoh(h_answer, d_answer)
        return int(h_answer[0])

    def fill(self, template, mapping):
        # Override symbols in .cu source file string
        for word, subs in mapping.items():
            template = template.replace(word, subs)
        return template

    def get_min_max_extent_of_surface(self, vertices):
        # Find min, max coordinates for the surface. Discretisation is
        # only used to obtain Morton codes for the location of triangles.
        minvals = np.min(vertices, axis=0).astype(np.float32)
        maxvals = np.max(vertices, axis=0).astype(np.float32)
        inv_delta = (self.params['QUANT_LEVELS'] / (maxvals - minvals)).astype(np.float32)
        half_delta = (0.5 / inv_delta).astype(np.float32)
        return minvals, maxvals, half_delta, inv_delta

    def perform_bindings_(self):
        bind = lambda x : self.module.get_function(x)
        self.bytes_in_AABB = bind("bytesInAABB")
        self.bytes_in_BVHNode = bind("bytesInBVHNode")
        self.bytes_in_CollisionList = bind("bytesInCollisionList")
        self.bytes_in_InterceptDistances = bind("bytesInInterceptDistances")
        self.kernel_create_morton_code = bind("kernelMortonCode")
        self.kernel_compute_ray_bounds = bind("kernelRayBox")
        self.kernel_bvh_reset = bind("kernelBVHReset")
        self.kernel_bvh_construct = bind("kernelBVHConstruct")
        self.kernel_bvh_find_intersections1 = bind("kernelBVHIntersection1")
        self.kernel_bvh_find_intersections2 = bind("kernelBVHIntersection2")
        self.kernel_bvh_find_intersections3 = bind("kernelBVHIntersection3")
        self.kernel_tide = bind("kernelTide")
        try:
            self.kernel_intersect_distances = bind("kernelIntersectDistances")
            self.kernel_intersect_points = bind("kernelIntersectPoints")
        except cuda.LogicError as e:
            if "cuModuleGetFunction failed: named symbol not found" in str(e):
                self.kernel_intersect_distances = None
                self.kernel_intersect_points = None

    def configure_(self, vertices, triangles, raysFrom, raysTo):
        self.h_vertices = np.array(vertices, dtype=np.float32)
        self.h_triangles = np.array(triangles, dtype=np.int32)
        self.h_raysFrom = np.array(raysFrom, np.float32)
        # self.h_raysTo = np.array(raysTo, np.float32)

        # Handling special cases
        # - BVH traversal expects root node in binary radix tree
        #   to have 2 descendants (see ISSUES.md: 1 in SHA 77cf088a4525)
        if len(self.h_triangles) == 1:
            self.h_triangles = np.vstack([self.h_triangles, [0,0,0]])
        # - For large spatial coordinates, shift the origin to maximise
        #   numerical precision (see ISSUES.md: 2 in SHA 1edbd3e39f2f)
        # self.translate_data_if_appropriate() # TODO: I don't think this is relevant for DeepDRR... See ISSUES.md

        # Get device attributes and specify grid-block partitions
        self.block_x = 512 if self.params['USE_DOUBLE_PRECISION_MOLLER'] else 1024
        grid_xlim = np.inf
        for devicenum in range(cuda.Device.count()):
            attrs = cuda.Device(devicenum).get_attributes()
            self.block_x = min(attrs[cuda.device_attribute.MAX_BLOCK_DIM_X], self.block_x)
            grid_xlim = min(attrs[cuda.device_attribute.MAX_GRID_DIM_X], grid_xlim)

        # self.n_rays = len(self.h_raysFrom)
        self.n_triangles = len(self.h_triangles)
        self.grid_xR = int(np.ceil(self.n_rays / self.block_x))
        self.grid_xT = int(np.ceil(self.n_triangles / self.block_x))
        self.grid_xLambda = 16
        assert max([self.grid_xR, self.grid_xT]) <= grid_xlim, \
              'Limit exceeded: use blockDim.y with 2D grid-blocks'
        self.block_dims = (self.block_x,1,1)
        self.grid_dimsR = (self.grid_xR,1)
        self.grid_dimsT = (self.grid_xT,1)
        self.grid_lambda = (self.grid_xLambda,1)
        if not self.quiet:
            print('CUDA partitions: {} threads/block, '
                  'grids: [rays: {}, bvh_construct: {}, bvh_intersect: {}]'.format(
                   self.block_x, self.grid_xR, self.grid_xT, self.grid_xLambda))

    def allocate_memory_(self):
        # Create a buffer for querying the bytesize of a data structure
        self.d_szQuery = cuda.mem_alloc(np.ones(1, dtype=np.int32).nbytes)
        # Allocate memory on host and device
        self.h_morton = np.zeros(self.n_triangles, dtype=np.uint64)
        # self.h_crossingDetected = np.zeros(self.n_rays, dtype=np.int32)
        # self.h_interceptTs = np.zeros((self.n_rays, self.max_intersections), dtype=np.float32)
        # self.h_interceptFacing = np.zeros((self.n_rays, self.max_intersections), dtype=np.int8)
        self.d_vertices = cuda.mem_alloc(self.h_vertices.nbytes)
        self.d_triangles = cuda.mem_alloc(self.h_triangles.nbytes)

        sz_ = lambda x : np.ones(1, dtype=x).nbytes
        get_ = lambda x : self.struct_size(x, self.d_szQuery)
        self.d_morton = cuda.mem_alloc(self.n_triangles * sz_(np.uint64))
        self.d_sortedTriangleIDs = cuda.mem_alloc(self.n_triangles * sz_(np.int32))
        # Data structures used in agglomerative LBVH construction
        self.d_leafNodes = cuda.mem_alloc(self.n_triangles * get_(self.bytes_in_BVHNode))
        self.d_internalNodes = cuda.mem_alloc(self.n_triangles * get_(self.bytes_in_BVHNode))
        self.d_hitIDs = cuda.mem_alloc(self.grid_xLambda * self.block_x * get_(self.bytes_in_CollisionList))


        # self.d_interceptTs = cuda.mem_alloc(self.h_interceptTs.nbytes)
        # self.d_interceptFacing = cuda.mem_alloc(self.h_interceptFacing.nbytes)

        # TODO: check free


    def transfer_data_(self):
        # Initialise memory or copy data from host to device
        cuda.memcpy_htod(self.d_vertices, self.h_vertices)
        cuda.memcpy_htod(self.d_triangles, self.h_triangles)
        cuda.memcpy_htod(self.d_raysFrom, self.h_raysFrom)
        # cuda.memcpy_htod(self.d_raysTo, self.h_raysTo)


    def test(self, vertices, triangles, raysFrom, raysTo, trace_dist, mesh_hit_alphas_gpu, mesh_hit_facing_gpu, cfg):
        # Set up resources
        t_start = time.time()
        self.configure_(vertices, triangles, raysFrom, raysTo)

        self.d_raysTo = raysTo

        self.allocate_memory_()
        
        self.d_interceptTs = mesh_hit_alphas_gpu
        self.d_interceptFacing = mesh_hit_facing_gpu

        self.transfer_data_()

        # Establish spatial domain of surface
        minvals, maxvals, half_delta, inv_delta = \
            self.get_min_max_extent_of_surface(self.h_vertices)

        # Pre-compute line segment bounding boxes
        self.kernel_compute_ray_bounds(
            self.d_raysFrom, self.d_raysTo, self.d_rayBox, np.int32(self.n_rays),
            block=self.block_dims, grid=self.grid_dimsR)
 
        # Sort triangles using Morton code
        self.kernel_create_morton_code(
            self.d_vertices, self.d_triangles, cuda.In(minvals),
            cuda.In(half_delta), cuda.In(inv_delta), self.d_morton,
            np.int32(self.n_triangles), block=self.block_dims, grid=self.grid_dimsT)
        cuda.memcpy_dtoh(self.h_morton, self.d_morton)

        h_sortedTriangleIDs = np.argsort(self.h_morton).astype(np.int32)
        '''
        Casting to 32-bit integer is super important as numpy.argsort returns
        an int64 array. Without it, the `t` variable that corresponds to
        node[i].triangleID in kernelBVHReset will be stepping through
        sortedTriangleIDs[] at half the required rate. As a consequence, not
        all the triangles in the mesh will be discovered. CUDA regards int* as
        a 32-bit integer pointer, so it is not compatible with an int64 array.
        '''
        self.h_morton = self.h_morton[h_sortedTriangleIDs]
        cuda.memcpy_htod(self.d_morton, self.h_morton)
        cuda.memcpy_htod(self.d_sortedTriangleIDs, h_sortedTriangleIDs)

        # Build bounding volume hierarchy for mesh triangles
        self.kernel_bvh_reset(
            self.d_vertices, self.d_triangles, self.d_internalNodes,
            self.d_leafNodes, self.d_sortedTriangleIDs, np.int32(self.n_triangles),
            block=self.block_dims, grid=self.grid_dimsT)

        self.kernel_bvh_construct(
            self.d_internalNodes, self.d_leafNodes, self.d_morton,
            np.int32(self.n_triangles), block=self.block_dims, grid=self.grid_dimsT)


        self.kernel_bvh_find_intersections3(
            self.d_vertices, self.d_triangles,
            self.d_raysFrom, self.d_raysTo,
            self.d_internalNodes, self.d_rayBox, self.d_hitIDs,
            self.d_interceptCounts, self.d_interceptTs, self.d_interceptFacing,
            np.int32(self.n_triangles), np.int32(self.n_rays), np.float32(trace_dist),
            block=self.block_dims, grid=self.grid_lambda)

        self.kernel_tide(
            self.d_interceptCounts, 
            self.d_interceptTs, 
            self.d_interceptFacing,
            np.int32(self.n_rays), 
            block=(int(self.block_x/2),1,1),  # TODO ??
            # block=self.block_dims,  # TODO ??
            grid=self.grid_lambda
        )

        # cuda.memcpy_dtoh(self.h_interceptCounts, int(self.d_interceptCounts))
        # cuda.memcpy_dtoh(self.h_interceptTs, int(self.d_interceptTs))
        # cuda.memcpy_dtoh(self.h_interceptFacing, int(self.d_interceptFacing))

        t_end = time.time()
        if not self.quiet:
            print('{}s\n'.format(t_end - t_start))

        # return self.h_interceptCounts, self.h_interceptTs, self.h_interceptFacing
        # return self.h_interceptCounts, self.d_interceptTs, self.d_interceptFacing

