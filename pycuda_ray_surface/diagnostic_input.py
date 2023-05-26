#============================================================================
#Copyright (c) 2023, Raymond Leung
#All rights reserved.
#
#This source code is licensed under the BSD-3-clause license.
#See the LICENSE.md file in the root directory for details.
#
#Purpose: Generate input data used in the second example in demo.py
#============================================================================
import numpy as np
import os
import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy import array

#Configuration parameters
#------------------------------
x_min, x_max = 100, 3100
y_min, y_max = 200, 2700
default_num_triangles = 30000
default_num_rays = 10000000
#------------------------------

#define spatial frequencies (w = 2*pi*f) for surface undulation
w = 2 * np.pi * np.array(
    [0.0487, 0.0912,
     0.0125, 0.0318,
     0.00672, 0.00543,
     0.00196, 0.00282,
     0.000571, 0.000386,
     0.000345, 0.000571,
     0.000251, 0.000145,
     0.000208, 0.000139])

#method to compute surface elevation using an analytic expression
def f_xy(x, y):
    return  1 * (np.cos(w[0]*x) + np.cos(w[1]*y)) \
          + 2 * (np.cos(w[2]*x - 0.3*np.pi) + np.cos(w[3]*y - 0.7*np.pi)) + \
          + 2.5 * (np.cos(w[4]*x + 1.2*np.pi) + np.cos(w[5]*y - 1.83*np.pi)) + \
          - 1.8 * (np.cos(w[6]*x + 2.93*np.pi) + np.cos(w[7]*y - 0.67*np.pi)) + \
          + 5 * (np.cos(w[8]*x + 0.04*np.pi) + np.cos(w[9]*y - 1.05*np.pi)) + \
          + 25 * (np.cos(w[10]*x - 0.61*np.pi) + np.cos(w[11]*y + 0.51*np.pi)) + \
          + 4 * (np.cos(w[12]*x - 0.61*np.pi) * np.cos(w[13]*y + 0.51*np.pi)) + \
          + 1.3 * (np.cos(w[14]*x * w[15]*y + 0.12*np.pi)) + \
          + 3 * np.sin(0.642*w[15]*y - 0.573*w[14]*x)

#API for creating a surface and saving the data in binary format
def synthesize_data(outdir,
                    n_triangles_approx=default_num_triangles,
                    n_rays=default_num_rays,
                    perturb_centroid=False,
                    feedback=dict()):
    x_range = x_max - x_min
    y_range = y_max - y_min
    aspect = y_range / x_range
    n_vertices_approx = int(n_triangles_approx / 2)

    #discretisation
    nX = int(np.sqrt(n_vertices_approx / aspect))
    nY = int(aspect * nX)
    xi = np.linspace(x_min, x_max, nX)
    yi = np.linspace(y_min, y_max, nY)
    delta = min(xi[1] - xi[0], yi[1] - yi[0])
    #add some noise to perturb xy coordinates
    np.random.seed(7065)
    noise_x = 0.25 * delta * (np.random.rand(nX) - 0.5)
    noise_y = 0.25 * delta * (np.random.rand(nY) - 0.5)
    xi += noise_x
    yi += noise_y

    #create mesh surface
    vertices = []
    triangles = []
    for y in yi:
        for x in xi:
            vertices.append([x, y, f_xy(x,y)])

    for y in range(nY-1):
        for x in range(nX-1):
            #vertices are ordered consistently in clockwise direction
            triangles.append([y*nX+x, y*nX+x+1, (y+1)*nX+x])
            triangles.append([y*nX+x+1, (y+1)*nX+x+1, (y+1)*nX+x])

    vertices = np.array(vertices, dtype=float)
    vertices[:,-1] -= min(vertices[:,-1])
    triangles = np.array(triangles, dtype=int)
    feedback['nVertices'] = len(vertices)
    feedback['nTriangles'] = len(triangles)
    feedback['nRays'] = n_rays

    #compute centroids and normal vectors for surface patches
    centroids = []
    normals = []
    for t in triangles:
        n = np.cross(vertices[t[1]] - vertices[t[0]],
                     vertices[t[2]] - vertices[t[0]])
        normals.append(n / np.linalg.norm(n))
        centroids.append(np.mean(vertices[t], axis=0))

    normals = np.array(normals)
    centroids = np.array(centroids)
    if perturb_centroid:
        np.random.seed(9571)
        a1 = 0.2 * (np.random.rand(n_rays) - 0.5)
        a2 = 0.2 * (np.random.rand(n_rays) - 0.5)
        for i, t in enumerate(triangles):
            centroids[i] += a1[i] * (vertices[t[1]] - vertices[t[0]]) \
                         +  a2[i] * (vertices[t[2]] - vertices[t[0]])

    #create rays
    #idea: Line segment starts from "centroid - (k/2) * normal"
    #      and extends for distance k*rand() in the normal direction.
    #      In the end, about half will intersect the surface.
    #- rand() generates random variates in union{(0,0.498],[0.502,1]}
    #  introduce deadzone (0.498,0.502) to make the result unambiguous.
    def rand(n):
        r = np.random.rand(n)
        r[r < 0.5] *= 0.996
        r[r >= 0.5] = 0.502 + (r[r >= 0.5] - 0.5) * 0.996
        return r

    np.random.seed(8215)
    r = rand(n_rays)
    s = 0.6 + 0.4 * np.random.rand(n_rays) #stochastic segment length scaling factor
    t = np.random.randint(len(triangles), size=n_rays) #random triangle selections
    rayFrom = []
    rayTo = []
    lower = []
    upper = []
    crossing = []
    max_segment_length = 4 * delta
    magnitude = max_segment_length * s
    lower = -0.5 * magnitude
    upper = r * magnitude
    rayFrom = centroids[t] + lower[:,np.newaxis] * normals[t]
    rayTo = rayFrom + upper[:,np.newaxis] * normals[t]
    crossing = np.array(r > 0.5, dtype=np.int32)

    #shift the coordinates to anonymise data and preserve precision as float32
    xyz_min = np.min(vertices, axis=0)
    vertices -= xyz_min
    rayFrom -= xyz_min
    rayTo -= xyz_min

    #write data to bin files
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fw = lambda f: os.path.join(outdir, f)
    t0 = time.time()
    with open(fw('vertices_f32'), 'wb') as f:
        np.array(vertices.flatten(),'float32').tofile(f)
    with open(fw('triangles_i32'), 'wb') as f:
        np.array(triangles.flatten(),'int32').tofile(f)
    with open(fw('rayFrom_f32'), 'wb') as f:
        np.array(rayFrom.flatten(),'float32').tofile(f)
    with open(fw('rayTo_f32'), 'wb') as f:
        np.array(rayTo.flatten(),'float32').tofile(f)
    t1 = time.time()
    print('Essential files written in {}s'.format(t1 - t0))

    print('Saving ground-truth...')
    with open(fw('ground_truth'), 'wb') as f:
        crossing.tofile(f)
    if perturb_centroid:
        intercepts = centroids[t] - xyz_min
        intercepts[crossing==0] = 0
        intersect_triangle = t
        intersect_triangle[crossing==0] = -1
        with open(fw('intercepts'), 'wb') as f:
            np.array(intercepts.flatten(),'float32').tofile(f)
        with open(fw('intersect_triangle'), 'wb') as f:
            np.array(intersect_triangle.flatten(),'int32').tofile(f)
