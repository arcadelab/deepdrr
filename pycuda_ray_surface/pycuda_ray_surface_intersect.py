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

from diagnostic_utils import display_node_contents
from diagnostic_graphics import bvh_graphviz, bvh_spatial

default_paths = {'PATH': '/usr/local/cuda-11.2/bin',
                 'LD_LIBRARY_PATH': '/usr/local/cuda-11.2/lib64',
                 'CUDA_INC_DIR': '/usr/local/cuda-11.2/include'}


class PyCudaRSI(object):
    """
    A stand-alone PyCUDA implementation of the ray-surface intersection
    algorithms described in https://arxiv.org/pdf/2209.02878.pdf
    This encompasses the core functions of gpu_ray_surface_intersect.cu,
    rsi_geometry.h, morton3D.h and bvh_structure.h. It performs parallel
    computations on a CUDA-capable Nvidia GPU device. The supported modes
    of operations include "boolean", "barycentric" and "intercept_count".
    Refer to <repo>/pycuda/README.md for installation steps and comments.
    """
    def __init__(self, params={}):
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

        self.params = {'QUANT_LEVELS': (1 << 21) - 1, 'LARGE_POS_VALUE': 2.5e+8}
        self.params['USE_EXTRA_BVH_FIELDS'] = params.get('USE_EXTRA_BVH_FIELDS', False)
        self.params['USE_DOUBLE_PRECISION_MOLLER'] = params.get('USE_DOUBLE_PRECISION_MOLLER', False)
        self.quiet = params.get('QUIET', False)
        self.tolerance = '%.6g' % params.get('TOLERANCE', 0.00001)
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
        name = params.get('CUDA_SOURCE_MODULE', 'pycuda_source')
        module = importlib.import_module(name)

        subst_dict = {
            'MORTON': 'uint64_t', 'COORD': 'unsigned int', 'TOLERANCE': self.tolerance,
            'MOLLER_DOUBLE_PRECISION_DIRECTIVE': '#define COMPILE_DOUBLE_PRECISION_MOLLER 1' \
                if self.params['USE_DOUBLE_PRECISION_MOLLER'] else '',
            'BVH_PREPROCESSOR_DIRECTIVE': '#define COMPILE_NON_ESSENTIAL 1' \
                if self.params['USE_EXTRA_BVH_FIELDS'] else ''
        }
        self.module = SourceModule(self.fill(module.get_cuda_template(), subst_dict))
        self.perform_bindings_()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    #------------------------------------------------------------------
    # Helper methods
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

    #------------------------------------------------------------------
    # Auxiliary functions
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
        try:
            self.kernel_intersect_distances = bind("kernelIntersectDistances")
            self.kernel_intersect_points = bind("kernelIntersectPoints")
        except cuda.LogicError as e:
            if "cuModuleGetFunction failed: named symbol not found" in str(e):
                self.kernel_intersect_distances = None
                self.kernel_intersect_points = None

    def translate_data_if_appropriate(self):
        '''
        Subtract the minimum coordinates from mesh and ray data if appropriate.
        Refer to comments on `translate_data` in "gpu_ray_surface_intersect.py"
        where issues relating to Universal Transverse Mercator (UTM) data,
        IEEE754 float32 representation and precision limit are discussed.
        '''
        self.shift_required = np.max(np.abs(self.h_vertices)) > 16384
        self.min_coords = np.zeros(3)
        if self.shift_required:
            self.min_coords = np.min(self.h_vertices, axis=0)
            self.h_vertices -= self.min_coords
            self.h_raysFrom -= self.min_coords
            self.h_raysTo -= self.min_coords

    def configure_(self, vertices, triangles, raysFrom, raysTo):
        self.h_vertices = np.array(vertices, dtype=np.float32)
        self.h_triangles = np.array(triangles, dtype=np.int32)
        self.h_raysFrom = np.array(raysFrom, np.float32)
        self.h_raysTo = np.array(raysTo, np.float32)

        # Handling special cases
        # - BVH traversal expects root node in binary radix tree
        #   to have 2 descendants (see ISSUES.md: 1 in SHA 77cf088a4525)
        if len(self.h_triangles) == 1:
            self.h_triangles = np.vstack([self.h_triangles, [0,0,0]])
        # - For large spatial coordinates, shift the origin to maximise
        #   numerical precision (see ISSUES.md: 2 in SHA 1edbd3e39f2f)
        self.translate_data_if_appropriate()

        # Get device attributes and specify grid-block partitions
        self.block_x = 512 if self.params['USE_DOUBLE_PRECISION_MOLLER'] else 1024
        grid_xlim = np.inf
        for devicenum in range(cuda.Device.count()):
            attrs = cuda.Device(devicenum).get_attributes()
            self.block_x = min(attrs[cuda.device_attribute.MAX_BLOCK_DIM_X], self.block_x)
            grid_xlim = min(attrs[cuda.device_attribute.MAX_GRID_DIM_X], grid_xlim)

        self.n_rays = len(self.h_raysFrom)
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
        self.h_crossingDetected = np.zeros(self.n_rays, dtype=np.int32)
        MAX_INTERSECTIONS = 32 # TODO
        self.h_interceptCounts = np.zeros(self.n_rays, dtype=np.int32)
        self.h_interceptTs = np.zeros((self.n_rays, MAX_INTERSECTIONS), dtype=np.float32)
        self.d_vertices = cuda.mem_alloc(self.h_vertices.nbytes)
        self.d_triangles = cuda.mem_alloc(self.h_triangles.nbytes)
        self.d_raysFrom = cuda.mem_alloc(self.h_raysFrom.nbytes)
        self.d_raysTo = cuda.mem_alloc(self.h_raysTo.nbytes)
        sz_ = lambda x : np.ones(1, dtype=x).nbytes
        get_ = lambda x : self.struct_size(x, self.d_szQuery)
        self.d_morton = cuda.mem_alloc(self.n_triangles * sz_(np.uint64))
        self.d_sortedTriangleIDs = cuda.mem_alloc(self.n_triangles * sz_(np.int32))
        self.d_rayBox = cuda.mem_alloc(self.n_rays * get_(self.bytes_in_AABB))
        # Data structures used in agglomerative LBVH construction
        self.d_leafNodes = cuda.mem_alloc(self.n_triangles * get_(self.bytes_in_BVHNode))
        self.d_internalNodes = cuda.mem_alloc(self.n_triangles * get_(self.bytes_in_BVHNode))
        self.d_hitIDs = cuda.mem_alloc(self.grid_xLambda * self.block_x *
                        get_(self.bytes_in_CollisionList))
        if self.mode == 'intercept_count':
            # self.d_interceptDists = cuda.mem_alloc(self.grid_xLambda * self.block_x *
                        # get_(self.bytes_in_InterceptDistances))
            self.d_interceptCounts = cuda.mem_alloc(self.h_interceptCounts.nbytes)
            self.d_interceptTs = cuda.mem_alloc(self.h_interceptTs.nbytes)
        if self.mode != 'barycentric':
            self.d_crossingDetected = cuda.mem_alloc(self.h_crossingDetected.nbytes)
        else:
            self.h_intersectTriangle = -np.ones(self.n_rays, dtype=np.int32)
            self.h_baryT = self.params['LARGE_POS_VALUE'] * np.ones(self.n_rays, dtype=np.float32)
            self.d_intersectTriangle = cuda.mem_alloc(self.h_intersectTriangle.nbytes)
            self.d_baryT = cuda.mem_alloc(self.h_baryT.nbytes)
            self.d_baryU = gpuarray.zeros((self.n_rays,), np.float32)
            self.d_baryV = gpuarray.zeros((self.n_rays,), np.float32)

    def transfer_data_(self):
        # Initialise memory or copy data from host to device
        cuda.memcpy_htod(self.d_vertices, self.h_vertices)
        cuda.memcpy_htod(self.d_triangles, self.h_triangles)
        cuda.memcpy_htod(self.d_raysFrom, self.h_raysFrom)
        cuda.memcpy_htod(self.d_raysTo, self.h_raysTo)

        if self.mode != 'barycentric':
            cuda.memset_d32(self.d_crossingDetected, 0, self.n_rays)
        else:
            cuda.memcpy_htod(self.d_intersectTriangle, self.h_intersectTriangle)
            cuda.memcpy_htod(self.d_baryT, self.h_baryT)
        
    #------------------------------------------------------------------
    # Core function
    def test(self, vertices, triangles, raysFrom, raysTo, cfg):
        # Specify operating mode
        self.show_morton = cfg.get('show_morton', False)
        self.quiet = cfg.get('quiet', False)
        self.mode = cfg.get('mode', 'boolean')
        assert(self.mode in ['boolean', 'barycentric', 'intercept_count'])
        # Set up resources
        t_start = time.time()
        self.configure_(vertices, triangles, raysFrom, raysTo)
        self.allocate_memory_()
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
        if self.show_morton:
            print('{}: {}'.format(self.h_morton, h_sortedTriangleIDs))

        # Build bounding volume hierarchy for mesh triangles
        self.kernel_bvh_reset(
            self.d_vertices, self.d_triangles, self.d_internalNodes,
            self.d_leafNodes, self.d_sortedTriangleIDs, np.int32(self.n_triangles),
            block=self.block_dims, grid=self.grid_dimsT)

        self.kernel_bvh_construct(
            self.d_internalNodes, self.d_leafNodes, self.d_morton,
            np.int32(self.n_triangles), block=self.block_dims, grid=self.grid_dimsT)

        #OPTION: Check BVH integrity
        #- Decode bytestream to recover the leaf and internal BVHNode arrays.
        #- The content is first copied from device memory to host memory
        #  using numpy.int32 conversion. The relevant fields are subsequently
        #  interpreted based on the type definitions given in struct BVHNode.
        #- In the following, a "word" refers to 4 contiguous bytes.
        examine_bvh = cfg.get('examine_bvh', False)
        bvh_visualisation = cfg.get('bvh_visualisation', [])
        if examine_bvh or len(bvh_visualisation) > 0:
            sz_ = lambda x : np.ones(1, dtype=x).nbytes
            get_ = lambda x : self.struct_size(x, self.d_szQuery)
            sz_BVHNode = int(get_(self.bytes_in_BVHNode) / sz_(np.int32))
            h_leafNodes = np.zeros(self.n_triangles * sz_BVHNode, dtype=np.int32)
            h_internalNodes = np.zeros(self.n_triangles * sz_BVHNode, dtype=np.int32)
            cuda.memcpy_dtoh(h_leafNodes, self.d_leafNodes)
            cuda.memcpy_dtoh(h_internalNodes, self.d_internalNodes)
            
        if examine_bvh:
            print('BVH tree structure')
            print('---------------------------\nInternal nodes')
            for t in range(self.n_triangles):
                words = h_internalNodes[t*sz_BVHNode:(t+1)*sz_BVHNode]
                display_node_contents(words, t, self.params['USE_EXTRA_BVH_FIELDS'])
            print('---------------------------\nLeaf nodes')
            for t in range(self.n_triangles):
                words = h_leafNodes[t*sz_BVHNode:(t+1)*sz_BVHNode]
                display_node_contents(words, t, self.params['USE_EXTRA_BVH_FIELDS'])
            print('---------------------------')

        if 'graph' in bvh_visualisation:
            assert self.params['USE_EXTRA_BVH_FIELDS']
            bvh_graphviz(h_internalNodes, h_leafNodes, self.n_triangles, sz_BVHNode)
        if 'spatial' in bvh_visualisation:
            assert self.params['USE_EXTRA_BVH_FIELDS']
            bvh_spatial(h_internalNodes, h_leafNodes, self.n_triangles, sz_BVHNode)

        # Apply the appropriate ray-surface intersection test
        if self.mode == 'boolean':
            self.kernel_bvh_find_intersections1(
                self.d_vertices, self.d_triangles,
                self.d_raysFrom, self.d_raysTo,
                self.d_internalNodes, self.d_rayBox,
                self.d_hitIDs, self.d_crossingDetected,
                np.int32(self.n_triangles), np.int32(self.n_rays),
                block=self.block_dims, grid=self.grid_lambda)
            cuda.memcpy_dtoh(self.h_crossingDetected, self.d_crossingDetected)
        elif self.mode == 'barycentric':
            self.kernel_bvh_find_intersections2(
                self.d_vertices, self.d_triangles,
                self.d_raysFrom, self.d_raysTo,
                self.d_internalNodes, self.d_rayBox,
                self.d_hitIDs, self.d_intersectTriangle,
                self.d_baryT, self.d_baryU, self.d_baryV,
                np.int32(self.n_triangles), np.int32(self.n_rays),
                block=self.block_dims, grid=self.grid_lambda)
            cuda.memcpy_dtoh(self.h_intersectTriangle, self.d_intersectTriangle)
            cuda.memcpy_dtoh(self.h_baryT, self.d_baryT)
            intersecting_rays = i = np.where(self.h_intersectTriangle >= 0)[0]
            #- Support computation on host or device across different versions
            if self.kernel_intersect_distances is None:
                normL2 = lambda x,y : np.sqrt(np.sum((x - y)**2, axis=1))
                distances = self.h_baryT[i] * normL2(self.h_raysTo[i], self.h_raysFrom[i])
            else: #perform the same calculation on the GPU
                d_output = self.d_baryU #reuse instead of cuda.mem_alloc
                self.kernel_intersect_distances(
                    self.d_raysFrom, self.d_raysTo, self.d_baryT, d_output,
                    np.int32(self.n_rays), block=self.block_dims, grid=self.grid_dimsR)
                distances = d_output.get()[i]
            hit_triangles = f = self.h_intersectTriangle[i]
            #- Compute intersecting points on GPU if kernel is defined
            if self.kernel_intersect_points is None:
                hit_points = self.h_raysFrom[i] + self.h_baryT[i][:, np.newaxis] * (
                             self.h_raysTo[i] - self.h_raysFrom[i])
            else:
                hit_points = np.empty(self.h_raysFrom.shape, dtype=np.float32)
                d_result = cuda.mem_alloc(hit_points.nbytes)
                self.kernel_intersect_points(
                    self.d_raysFrom, self.d_raysTo, self.d_baryT, d_result,
                    np.int32(self.n_rays), block=self.block_dims, grid=self.grid_dimsR)
                cuda.memcpy_dtoh(hit_points, d_result)
                hit_points = hit_points[i]
            #- Adjust for translation
            if self.shift_required:
                hit_points += self.min_coords
        elif self.mode == 'intercept_count':
            self.kernel_bvh_find_intersections3(
                self.d_vertices, self.d_triangles,
                self.d_raysFrom, self.d_raysTo,
                self.d_internalNodes, self.d_rayBox, self.d_hitIDs,
                self.d_interceptCounts, self.d_interceptTs,
                np.int32(self.n_triangles), np.int32(self.n_rays),
                block=self.block_dims, grid=self.grid_lambda)
            cuda.memcpy_dtoh(self.h_interceptCounts, self.d_interceptCounts)
            cuda.memcpy_dtoh(self.h_interceptTs, self.d_interceptTs)
            # cuda.memcpy_dtoh(self.h_crossingDetected, self.d_crossingDetected)

        t_end = time.time()
        if not self.quiet:
            print('{}s\n'.format(t_end - t_start))

        if self.mode == 'barycentric':
            return intersecting_rays, distances, hit_triangles, hit_points
        if self.mode == 'intercept_count':
            return self.h_interceptCounts, self.h_interceptTs
        else:
            return self.h_crossingDetected
