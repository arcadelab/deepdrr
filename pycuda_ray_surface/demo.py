#!/usr/bin/env python3
#============================================================================
#Copyright (c) 2023, Raymond Leung
#All rights reserved.
#
#This source code is licensed under the BSD-3-clause license.
#See the LICENSE.md file in the root directory for details.
#
#Purpose: Show how `pycuda_ray_surface_intersect.py` is used.
#         It runs the examples described in pycuda/README.md
#============================================================================
import numpy as np
import os

from pycuda_ray_surface_intersect import PyCudaRSI
from diagnostic_input import synthesize_data
from diagnostic_graphics import visualise_example1

info_msg = """
NOTE: Before running this script, users will need to set the
      CUDA bin/lib/include paths using the `export` command or
      specify the ENVIRONMENT VARIABLES using the `params`
      dict given to PyCudaRSI.__init__(self, params={})

      On a Linux machine, the paths might look like these:
          PATH=/usr/local/cuda-xy.z/bin:${PATH}
          LD_LIBRARY_PATH=/usr/local/cuda-xy.z/lib64
          CUDA_INC_DIR=/usr/local/cuda-xy.z/include
      where xy.z denotes the CUDA compiler version such as 11.2
"""

def bin2array(filename, precision, dims=2):
    # Read content from binary file into numpy.array
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=precision)
    return data.reshape([int(len(data)/3), 3]) if dims==2 else data

def run_example1(mode, makeplot):
    # Scenario 1: Rectangular surface with 4 triangles and 8 line segments.
    #             Ray 1,2,4 and 7 intersect triangle 0,1,2 and 3, resp.
    vertices = np.array([[12.,2.,1.],[13.,2.,1.2], [12.5,2.5,1.1],
                         [12.,3.,1.2],[13.,3.,1.3]], dtype=np.float32)
    triangles = np.array([[0,1,2],[1,4,2],[2,4,3],[0,2,3]], dtype=np.int32)
    raysFrom = np.array([[12.2,2.1,0], [12.7,2.2,2], [12.9,2.4,1],
                        [12.9,2.6,1.5], [12.6,2.9,1], [12.35,2.8,0],
                        [12.25,2.5,0], [12.2,2.4,0.5]], dtype=np.float32)
    raysTo = np.array([[12.2,2.1,0.8], [12.7,2.2,0], [12.9,2.4,2],
                      [12.9,2.6,2.5], [12.6,2.9,2], [12.35,2.8,1],
                      [12.25,2.5,1], [12.2,2.4,1.5]], dtype=np.float32)

    # For "intercept_count", add a canopy (two triangular patches)
    # above the rectangular base surface to make it more interesting.
    if mode == 'intercept_count':
        vertices = np.r_[vertices, vertices[1:] + [0,0,0.2]]
        triangles = np.r_[triangles, [[5,6,8], [6,7,8]]]
        gt_crossings_detected = [0,1,2,0,2,0,0,1]
    else:
        gt_crossings_detected = [0,1,1,0,1,0,0,1]
    gt_rays = np.where(gt_crossings_detected)[0]
    gt_intercepts = np.array([[12.7,2.2,1.14], [12.9,2.4,1.21],
                              [12.6,2.9,1.23], [12.2,2.4,1.08]])
    gt_triangles = [0,1,2,3]
    gt_distances = np.sqrt(np.sum((gt_intercepts - raysFrom[gt_rays])**2, axis=1))

    if makeplot:
        visualise_example1(vertices, triangles, raysFrom, raysTo,
                           gt_rays, gt_triangles, gt_intercepts)

    #----------------------------------------------------------------------
    # Usage pattern
    params = {'USE_EXTRA_BVH_FIELDS': True, 'EPSILON': 0.001}
    cfg = {'mode': mode,
           'examine_bvh': mode=='intercept_count',
           'bvh_visualisation': [] }
    # 'examine_bvh' prints out the tree and node attributes when True.
    # 'bvh_visualisation' (an action list) must contain the word
    #    'graph' to generate a graph of the mesh triangles binary radix tree
    #    'spatial' to produce a hierarchical representation of this tree
    '''
    Note:
    - 'USE_EXTRA_BVH_FIELDS' is set to False by default.
       Both 'examine_bvh' and 'bvh_visualisation' are disabled,
       defaulting to False and [] outside of testing.
    - 'EPSILON' represents the zero threshold used in the Moller-Trumbore
       algorithm. It is set to 0.00001 by default.
    -  These options are set explicitly here for illustrative purpose.
    '''
    with PyCudaRSI(params) as pycu:
        if mode != 'barycentric':
            # Determine whether an intersection had occured
            ray_intersects = pycu.test(vertices, triangles, raysFrom, raysTo, cfg)
        else:
            # Find intersecting [rays, distances, triangles and points]
            (intersecting_rays, distances, hit_triangles, hit_points) \
                = pycu.test(vertices, triangles, raysFrom, raysTo, cfg)
    #----------------------------------------------------------------------

    # Check the results
    if mode in ['boolean', 'intercept_count']:
        print(f'- ray_surface_intersect test {mode} results')
        for i,result in enumerate(ray_intersects):
            print(f'{i}: {result}')
        assert(np.all(ray_intersects == gt_crossings_detected))
    else:
        print(f'- ray_surface_intersect test {mode} results')
        for i in range(len(intersecting_rays)):
            p, d = hit_points[i], distances[i]
            print('ray={}: triangle={}, dist={}, point={}'.format(
                   intersecting_rays[i], hit_triangles[i], '%.6g' % d,
                  '[%.6g,%.6g,%.6g]' % (p[0],p[1],p[2])))
        #- verification
        assert(np.all(intersecting_rays == gt_rays))
        assert(np.all(hit_triangles == gt_triangles))
        assert(np.all(np.isclose(distances, gt_distances)))
        assert(np.all(np.isclose(hit_points, gt_intercepts, atol=1e-3)))

def run_example2(mode):
    # Scenario 2: Synthesized surface with ~30k triangles, and 100k rays.
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Generate input data once
    vertices_file = os.path.join(data_dir, 'vertices_f32')
    if not os.path.exists(vertices_file):
        print('Generating test input in {}'.format(data_dir))
        desc = dict()
        synthesize_data(data_dir, n_triangles_approx=30000, n_rays=10000000,
                        perturb_centroid=True, feedback=desc)
        print('Synthesized data with {} vertices, {} triangles and {} rays'.format(
               desc['nVertices'], desc['nTriangles'], desc['nRays']))

    # Read data
    vertices = bin2array('data/vertices_f32', np.float32)
    triangles = bin2array('data/triangles_i32', np.int32)
    raysFrom = bin2array('data/rayFrom_f32', np.float32)
    raysTo = bin2array('data/rayTo_f32', np.float32)
    ground_truth = bin2array('data/ground_truth', np.int32, dims=1)

    # Run CUDA program (execute GPU kernels)
    params = dict()
    cfg = {'quiet': False, 'mode': mode}
    '''
    # Uncomment to produce spatial representations of the binary radix tree
    if mode == 'intercept_count':
        params['USE_EXTRA_BVH_FIELDS'] = True
        cfg['bvh_visualisation'] = ['spatial']
    '''
    with PyCudaRSI(params) as pycu:
        if mode != 'barycentric':
            ray_intersects = pycu.test(vertices, triangles, raysFrom, raysTo, cfg)
        else:
            (intersecting_rays, distances, hit_triangles, hit_points) \
                = pycu.test(vertices, triangles, raysFrom, raysTo, cfg)

    # Check the results
    if mode in ['boolean', 'intercept_count']:
        print(f'- ray_surface_intersect test {mode} results[-10:]')
        ray_idx = np.arange(len(raysFrom))
        for i,result in zip(ray_idx[-10:], ray_intersects[-10:]):
            print(f'{i}: {result}')
        assert(np.all(ray_intersects == ground_truth))
    else:
        gt_intercepts = bin2array('data/intercepts', np.float32)
        gt_triangles = bin2array('data/intersect_triangle', np.int32, dims=1)
        gt_distances = np.sqrt(np.sum((gt_intercepts - raysFrom)**2, axis=1))
        #- intersecting rays
        gt_rays = np.where(gt_triangles >= 0)[0]
        print(f'- ray_surface_intersect test {mode} results[-6:]')
        for j in range(6)[::-1]:
            p = hit_points[-j]
            print('ray={}: triangle={}, dist={}, point={}'.format(
                   gt_rays[-j], hit_triangles[-j], '%.6g' % distances[-j],
                  '[%.6g,%.6g,%.6g]' % (p[0],p[1],p[2])))
        #- verification
        assert(np.all(intersecting_rays == gt_rays))
        assert(np.all(hit_triangles == gt_triangles[gt_rays]))
        assert(np.all(np.isclose(distances, gt_distances[gt_rays])))
        assert(np.all(np.isclose(hit_points, gt_intercepts[gt_rays], atol=1e-3)))

def run_example3(mode):
    vertices = bin2array('data/vertices_f32', np.float32)
    triangles = bin2array('data/triangles_i32', np.int32)
    raysFrom = bin2array('data/rayFrom_f32', np.float32)
    raysTo = bin2array('data/rayTo_f32', np.float32)
    # Compare results produced by different versions of the code
    # - Let's assume "pycuda_source_legacy.py" represents the stable version
    #   while "pycuda_source.py" is under development (contains new changes)
    # - Note: PyCudaRSI.__init__ uses the cuda_template from 'pycuda_source'
    #   by default when the key 'CUDA_SOURCE_MODULE' is omitted from params.
    cfg = {'quiet': True, 'mode': mode}
    params = {} 
    with PyCudaRSI(params) as pycu:
        results1 = pycu.test(vertices, triangles, raysFrom, raysTo, cfg)

    # self.h_crossingDetected, self.h_interceptCounts, self.h_interceptTs
    interceptCounts, interceptTs = results1
    print(f'interceptCounts[-10:]: {interceptCounts[-10:]}')
    print(f'interceptTs[-10:]: {interceptTs[-10:]}')    

    # print max min and count nonzero
    print(f"interceptCounts.max(): {interceptCounts.max()}")
    print(f"interceptCounts.min(): {interceptCounts.min()}")
    print(f"interceptCounts number of nonzero: {np.count_nonzero(interceptCounts)}")
    print(f"interceptTs.max(): {interceptTs.max()}")
    print(f"interceptTs.min(): {interceptTs.min()}")
    print(f"interceptTs count close to zero: {np.count_nonzero(np.isclose(interceptTs.sum(axis=1), 0))}")


    print(f"interceptCounts.shape: {interceptCounts.shape}")
    print(f"interceptTs.shape: {interceptTs.shape}")    
    print(f"interceptTs.shape: {interceptTs.sum(axis=1).shape}")    
    print(np.unique(interceptCounts, return_counts=True))    


    print("done")
    # params['CUDA_SOURCE_MODULE'] = 'pycuda_source_legacy'
    # with PyCudaRSI(params) as pycu:
    #     results2 = pycu.test(vertices, triangles, raysFrom, raysTo, cfg)

    # finding = "- results are the same"
    # if type(results1) == tuple:
    #     for x, y in zip(results1, results2):
    #         if not np.all(np.isclose(x, y)):
    #             finding = "- results are different"
    # else:
    #     if not np.all(np.isclose(results1, results2)):
    #         finding = "- results are different"
    # print(finding)


if __name__ == "__main__":

    from pytools.prefork import ExecError

    try:
        # for mode in ['boolean', 'barycentric', 'intercept_count']:
        #     print(f'\nRunning example 1, mode={mode}')
        #     run_example1(mode, makeplot=False)

        # for mode in ['boolean', 'barycentric', 'intercept_count']:
        #     print(f'\nRunning example 2, mode={mode}')
        #     run_example2(mode)

        for mode in ['intercept_count']:
            print(f'\nRunning example 3, mode={mode}')
            run_example3(mode)
    except (FileNotFoundError, ExecError) as error:
        if "No such file or directory: 'nvcc'" in str(error):
             print(info_msg)
        raise error
