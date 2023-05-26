#============================================================================
#Copyright (c) 2023, Raymond Leung
#All rights reserved.
#
#This source code is licensed under the BSD-3-clause license.
#See the LICENSE.md file in the root directory for details.
#
#Purpose: Visualise bounding volume hierarchy spatially or as a graph
#============================================================================
import graphviz
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import matplotlib.cm as cm
import numpy as np

from diagnostic_utils import bin_to_float


def visualise_example1(vertices, triangles, raysFrom, raysTo,
                       gt_rays, gt_triangles, gt_intercepts):
    np.random.seed(4261)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    vx = [[vertices[v,0] for v in t] for t in triangles]
    vy = [[vertices[v,1] for v in t] for t in triangles]
    vz = [[vertices[v,2] for v in t] for t in triangles]
    for x,y,z in zip(vx,vy,vz):
        p = Poly3DCollection([list(zip(x,y,z))])
        p.set_facecolor(np.random.rand(3))
        p.set_alpha(0.5)
        ax.add_collection3d(p)
    ax.scatter(raysFrom[gt_rays,0], raysFrom[gt_rays,1],
               raysFrom[gt_rays,2], s=10, c='k', marker='o')
    ax.scatter(raysTo[gt_rays,0], raysTo[gt_rays,1],
               raysTo[gt_rays,2], s=10, c='k', marker='o')
    ax.set_xlim([np.min(vx)-0.1, np.max(vx)+0.1])
    ax.set_ylim([np.min(vy)-0.1, np.max(vy)+0.1])
    ax.set_zlim([np.min(vz)-2, np.max(vz)+2])
    raysIdx = range(len(raysFrom))
    for i, a, b in zip(raysIdx, raysFrom, raysTo):
        hit = i in gt_rays
        ax.plot([a[0],b[0]], [a[1],b[1]], [a[2],b[2]], c='k' if hit else (.5,.5,.5))
        if hit:
            z = gt_intercepts[gt_rays.tolist().index(i)]
            ax.plot([z[0]], [z[1]], [z[2]], c='k', marker='x')
    ax.set_aspect('auto')
    plt.show()

#----------------------------------------------------------------------
# BVH visualisation
'''
Nomenclature:
- words          represents the content of a BVHNode as a numpy.int32 array
- internal_nodes represents all the internal BVHNodes in a BVH binary radix
                 tree, where each BVHNode corresponds to a contiguous byte-
                 stream or an numpy.int32 array with sz_BVHNode elements.
- leaf_nodes     ditto for all the leaf nodes in a BVH binary radix tree.

Convention:
- the leaf_nodes and internal_nodes are typically obtained using
| leaf_nodes = np.zeros(nTriangles * int(struct_size(bytes_in_BVHNode, d_szQuery)
|                     / np.ones(1, dtype=np.int32).nbytes), dtype=np.int32)
| cuda.memcpy_dtoh(leaf_nodes, device_leaf_nodes)
'''

def decode_(words):
    isLeafSelf = sum(words[6:10])==0
    idxSelf, idxChildL, idxChildR, isLeafChildL, isLeafChildR = words[14:19]
    triangleID, atomic, rangeL, rangeR = words[19:23]
    return isLeafSelf, isLeafChildL, isLeafChildR, idxSelf, idxChildL, idxChildR, \
           triangleID, rangeL, rangeR

def decode2_(words):
    xMin = bin_to_float(np.binary_repr(words[0],32),4)
    xMax = bin_to_float(np.binary_repr(words[1],32),4)
    yMin = bin_to_float(np.binary_repr(words[2],32),4)
    yMax = bin_to_float(np.binary_repr(words[3],32),4)
    zMin = bin_to_float(np.binary_repr(words[4],32),4)
    zMax = bin_to_float(np.binary_repr(words[5],32),4)
    idxChildL, idxChildR, isLeafChildL, isLeafChildR = words[15:19]
    rangeL, rangeR = words[21:23]
    return [xMin,xMax], [yMin,yMax], [zMin,zMax], isLeafChildL, idxChildL, \
            isLeafChildR, idxChildR, rangeL, rangeR

def get_colormap(N=256):
    cmap = cm.get_cmap('Blues', N)
    palette = []
    for i in range(cmap.N):
        rgba = cmap(i)
        palette.append(mcl.rgb2hex(rgba))
    return palette, N

def draw_rect_prism_(x_range, y_range, z_range, rgb, ax, fill=True):
    #https://codereview.stackexchange.com/a/155601
    mat = lambda x : np.array([[x]*2]*2)
    rW = rgb if fill else (0,0,0)
    rS = rgb if fill else (1,1,1)
    aS = 0.2 if fill else 0.0
    #draw xy edges
    xx, yy = np.meshgrid(x_range, y_range)
    for v in [0,1]:
        ax.plot_wireframe(xx, yy, mat(z_range[v]), color=rW)
        ax.plot_surface(xx, yy, mat(z_range[v]), color=rS, alpha=aS)
    #draw yz edges
    yy, zz = np.meshgrid(y_range, z_range)
    for v in [0,1]:
        ax.plot_wireframe(mat(x_range[v]), yy, zz, color=rgb)
        ax.plot_surface(mat(x_range[v]), yy, zz, color=rS, alpha=aS)
    #draw xz edges
    xx, zz = np.meshgrid(x_range, z_range)
    for v in [0,1]:
        ax.plot_wireframe(xx, mat(y_range[v]), zz, color=rgb)
        ax.plot_surface(xx, mat(y_range[v]), zz, color=rS, alpha=aS)

def bvh_spatial(internal_nodes, leaf_nodes, n_triangles,
                sz_BVHNode, max_depth=10):
    #draw bounding volumes at each level of the hierarchy
    #starting from the root node and marching toward leaf nodes
    print('Visualising binary radix tree bounding volumes...')
    plt.rcParams.update({
        "axes.facecolor":    (0.4, 0.4, 0.4), #dark gray background
        "savefig.facecolor": (0.4, 0.4, 0.4)
    })
    palette, N = get_colormap()
    root_node_idx = decode_(internal_nodes[-sz_BVHNode:])[4]
    INTERNAL, LEAF = 0, 1
    CURRENT, NEXT = 0, 1
    level = 0
    nodes = {INTERNAL: internal_nodes, LEAF: leaf_nodes}
    queue = {CURRENT: [(INTERNAL, root_node_idx)], NEXT: []}

    while len(queue[CURRENT]) > 0 and level < max_depth:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i,j in queue[CURRENT]:
            (x_range, y_range, z_range, isLeafChildL, idxChildL,
             isLeafChildR, idxChildR, rangeL, rangeR) = decode2_(
             nodes[i][j*sz_BVHNode:(j+1)*sz_BVHNode])
            if i==INTERNAL:
                fL, fR = int((rangeL / n_triangles) * N), int((rangeR / n_triangles) *N)
                fC = min(max(int(0.5 * (fL + fR)), 0), N-1)
                draw_rect_prism_(x_range, y_range, z_range, palette[fC], ax=ax, fill=False)
                queue[NEXT].append((LEAF if isLeafChildL else INTERNAL, idxChildL))
                queue[NEXT].append((LEAF if isLeafChildR else INTERNAL, idxChildR))
            else: #LEAF
                fM = min(max(int(rangeL / n_triangles * N), 0), N-1)
                draw_rect_prism_(x_range, y_range, z_range, palette[fM], ax=ax)
        print('hierarchy {}: expanded {} nodes'.format(level, len(queue[CURRENT])))
        plt.title(f'Bounding volumes at hierarchy {level}')
        plt.savefig('bvh_spatial%03d.png' % (level))
        plt.close()
        queue[CURRENT] = list(queue[NEXT])
        queue[NEXT] = []
        level += 1

def bvh_graphviz(internal_nodes, leaf_nodes, n_triangles, sz_BVHNode):
    names = []
    labels = []
    colors = []
    palette, N = get_colormap()
    f = graphviz.Digraph(filename='data/bvh_structure.gv')
    #!dot -Kfdp -n -Tjpeg -Gdpi=300 -O bvh_structure.gv
    for t in range(n_triangles):
        #for internal node
        words = internal_nodes[t*sz_BVHNode:(t+1)*sz_BVHNode]
        (leafSelf, leafL, leafR, nodeSelf,
         nodeL, nodeR, triangle, rangeL, rangeR) = decode_(words)
        fL, fR = int((rangeL / n_triangles) * N), int((rangeR / n_triangles) *N)
        fC = min(max(int(0.5 * (fL + fR)), 0), N-1)
        if t < n_triangles - 1:
            names.append('I_{}'.format(nodeSelf))
            labels.append('[%d,%d]' % (rangeL, rangeR))
            colors.append(palette[fC])
            f.node(names[-1], labels[-1], style='filled',
                   color='black', fillcolor=colors[-1])
            typeL = 'L' if leafL else 'I'
            typeR = 'L' if leafR else 'I'
            f.edge('I_{}'.format(nodeSelf), '{}_{}'.format(typeL, nodeL))
            f.edge('I_{}'.format(nodeSelf), '{}_{}'.format(typeR, nodeR))
        #for leaf node
        words = leaf_nodes[t*sz_BVHNode:(t+1)*sz_BVHNode]
        triangle, rangeM = decode_(words)[6:8]
        names.append('L_{}'.format(nodeSelf))
        #- display [node index] {triangleID}
        labels.append('<[%d] <b>%d</b>>' % (rangeM, triangle))
        fM = min(max(int(rangeM / n_triangles * N), 0), N-1)
        colors.append(palette[fM])
        f.node(names[-1], labels[-1], shape='box', style='filled',
               color='black', fillcolor=colors[-1])
    f.view()

#----------------------------------------------------------------------
