# #============================================================================
# #Copyright (c) 2023, Raymond Leung
# #All rights reserved.
# #
# #This source code is licensed under the BSD-3-clause license.
# #See the LICENSE.md file in the root directory for details.
# #
# #Purpose: Encapsulate the cuda module source code
# #This version includes features up to commit 6ecfbc8e057cd16f
# #============================================================================
# cuda_template = """
# #include <stdint.h>

# //-----------------------------------------------------------
# // Component 1: Ray-surface intersection geometry tests
# //-----------------------------------------------------------
# #define EPSILON TOLERANCE
# #define MAX_INTERSECTIONS 32

# using namespace std;

# /* data structures */
# struct AABB
# {   //axes-aligned bounding box
#     float xMin, xMax, yMin, yMax, zMin, zMax;
# };
# struct InterceptDistances
# {
#     float t[MAX_INTERSECTIONS];
#     int count;
# };

# __global__ void bytesInAABB(int &x){ x = sizeof(AABB); }
# __global__ void bytesInInterceptDistances(int &x){ x = sizeof(InterceptDistances); }

# /* algebraic operations */
# __device__ void subtract(const float *a, const float *b, float *out)
# {
#     out[0] = a[0] - b[0];
#     out[1] = a[1] - b[1];
#     out[2] = a[2] - b[2];
# }
# __device__ void dot(const float *a, const float *b, float &out)
# {
#     out = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
# }
# __device__ void cross(const float *a, const float *b, float *out)
# {
#     out[0] = a[1]*b[2] - a[2]*b[1];
#     out[1] = a[2]*b[0] - a[0]*b[2];
#     out[2] = a[0]*b[1] - a[1]*b[0];
# }

# /* auxiliary functions */
# __device__ void lineSegmentBbox(const float *p0, const float *p1, AABB &box)
# {
#     if (p0[0] > p1[0]) { box.xMin = p1[0]; box.xMax = p0[0]; }
#     else               { box.xMin = p0[0]; box.xMax = p1[0]; }
#     if (p0[1] > p1[1]) { box.yMin = p1[1]; box.yMax = p0[1]; }
#     else               { box.yMin = p0[1]; box.yMax = p1[1]; }
#     if (p0[2] > p1[2]) { box.zMin = p1[2]; box.zMax = p0[2]; }
#     else               { box.zMin = p0[2]; box.zMax = p1[2]; }
# }
# __device__ bool notOverlap(const float *tMin, const float *tMax,
#                            const float *rayMin, const float *rayMax)
# {   //this version uses precomputed rayMin and rayMax
#     if (rayMin[0] > tMax[0] || rayMax[0] < tMin[0])
#         return true;
#     if (rayMin[1] > tMax[1] || rayMax[1] < tMin[1])
#         return true;
#     if (rayMin[2] > tMax[2] || rayMax[2] < tMin[2])
#         return true;
#     return false;
# }

# __global__ void kernelRayBox(const float* __restrict__ rayFrom,
#                              const float* __restrict__ rayTo,
#                              AABB* __restrict__ rayBox, int numRays)
# {
#     //Pre-compute min/max coordinates for all line segments
#     //instead of repeating the same in each thread-block.
#     const int i = blockIdx.x * blockDim.x + threadIdx.x;
#     if (i < numRays) {
#         const float *start = &rayFrom[3*i], *finish = &rayTo[3*i];
#         lineSegmentBbox(start, finish, rayBox[i]);
#     }
# }

# __global__ void initArrayKernel(float* array, float value, int numElements)
# {
#     const int i = blockIdx.x * blockDim.x + threadIdx.x;
#     if (i < numElements) {  array[i] = value;  }
# }

# __global__ void setIntArrayKernel(int* array, int value, int numElements)
# {
#     const int i = blockIdx.x * blockDim.x + threadIdx.x;
#     if (i < numElements) {  array[i] = value;  }
# }

# /* Moller-Trumbore ray-triangle intersection algorithm */
# // - Ray model: R(t) = Q0 + t *(Q1 - Q0) given line segment end points Q0, Q1
# // - Intersection on plane of triangle: T(u,v) = (1-u-v)*V0 + u*V1 + v*V2
# __device__ int intersectMoller(
#                 const float *v0, const float *v1, const float *v2,
#                 const float *edge1, const float *edge2,
#                 const float *q0, const float *q1,
#                 float &t, float &u, float &v)
# {
#     float direction[3], avec[3], bvec[3], tvec[3], det, inv_det;
#     subtract(q1, q0, direction);
#     cross(direction, edge2, avec);
#     dot(avec, edge1, det);
#     if (det > EPSILON) {
#         subtract(q0, v0, tvec);
#         dot(avec, tvec, u);
#         if (u < 0 || u > det)
#             return 0;
#         cross(tvec, edge1, bvec);
#         dot(bvec, direction, v);
#         if (v < 0 || u + v > det)
#             return 0;
#     }
#     else if (det < -EPSILON) {
#         subtract(q0, v0, tvec);
#         dot(avec, tvec, u);
#         if (u > 0 || u < det)
#             return 0;
#         cross(tvec, edge1, bvec);
#         dot(bvec, direction, v);
#         if (v > 0 || u + v < det)
#             return 0;
#     }
#     else
#         return 0;
#     inv_det = 1.0 / det;
#     dot(bvec, edge2, t);
#     t *= inv_det;
#     if (t < 0 || t > 1) {
#         return 0;
#     }
#     else {
#         u *= inv_det;
#         v *= inv_det;
#         return 1;
#     }
# }

# __device__ void inline computeEdges(const float* __restrict__ vertices,
#                                     const int* __restrict__ triangles,
#                                     const float *v0, const float *v1, const float *v2,
#                                     int triangleID, float *triangleVerts,
#                                     float *edge1, float *edge2)
# {
#     for(int j = 0; j < 3; j++) {
#         int v = triangles[3*triangleID+j];
#         for (int k = 0; k < 3; k++)
#             triangleVerts[3*j+k] = vertices[3*v+k];
#     }
#     subtract(v1, v0, edge1);
#     subtract(v2, v0, edge2);
# }

# // Test v1: reports intersection as boolean in `results`
# __device__ void checkRayTriangleIntersection1(const float* __restrict__ vertices,
#                                               const int* __restrict__ triangles,
#                                               const float* __restrict__ rayFrom,
#                                               const float* __restrict__ rayTo,
#                                               int* __restrict__ results,
#                                               int rayIdx, int triangleID)
# {
#     float t, u, v, triVerts[9], edge1[3], edge2[3];
#     const float *v0=&triVerts[0], *v1=&triVerts[3], *v2=&triVerts[6];
#     const float *start = &rayFrom[3*rayIdx], *finish = &rayTo[3*rayIdx];

#     computeEdges(vertices, triangles, v0, v1, v2, triangleID, triVerts, edge1, edge2);

#     if (intersectMoller(v0, v1, v2, edge1, edge2, start, finish, t, u, v)) {
#         results[rayIdx] = 1;
#     }
# }

# // Test v2: report barycentric coordinates (t,u,v) where t=distance(rayFrom,surface)
# __device__ void checkRayTriangleIntersection2(const float* __restrict__ vertices,
#                                               const int* __restrict__ triangles,
#                                               const float* __restrict__ rayFrom,
#                                               const float* __restrict__ rayTo,
#                                               int* __restrict__ intersectTriangle,
#                                               float* baryT, float* baryU, float* baryV,
#                                               int rayIdx, int triangleID)
# {
#     float t, u, v, triVerts[9], edge1[3], edge2[3];
#     const float *v0=&triVerts[0], *v1=&triVerts[3], *v2=&triVerts[6];
#     const float *start = &rayFrom[3*rayIdx], *finish = &rayTo[3*rayIdx];

#     computeEdges(vertices, triangles, v0, v1, v2, triangleID, triVerts, edge1, edge2);

#     if (intersectMoller(v0, v1, v2, edge1, edge2, start, finish, t, u, v)) {
#         if (t < baryT[rayIdx]) {
#             intersectTriangle[rayIdx] = triangleID;
#             baryT[rayIdx] = t;
#             baryU[rayIdx] = u;
#             baryV[rayIdx] = v;
#         }
#     }
# }

# // Test v3: report number of unique ray-surface intersections (limited to < 32)
# __device__ void checkRayTriangleIntersection3(const float* __restrict__ vertices,
#                                               const int* __restrict__ triangles,
#                                               const float* __restrict__ rayFrom,
#                                               const float* __restrict__ rayTo,
#                                               InterceptDistances &interceptDists,
#                                               int* __restrict__ results,
#                                               int rayIdx, int triangleID)
# {
#     const float tol(EPSILON);
#     float t, u, v, triVerts[9], edge1[3], edge2[3];
#     float *tp = interceptDists.t; //circular buffer
#     const float *v0=&triVerts[0], *v1=&triVerts[3], *v2=&triVerts[6];
#     const float *start = &rayFrom[3*rayIdx], *finish = &rayTo[3*rayIdx];

#     computeEdges(vertices, triangles, v0, v1, v2, triangleID, triVerts, edge1, edge2);

#     if (intersectMoller(v0, v1, v2, edge1, edge2, start, finish, t, u, v)) {
#         bool newIntercept(true);
#         for (int i = 0; i < MAX_INTERSECTIONS; i++)
#             if ((t > tp[i] - tol) && (t < tp[i] + tol)) {
#                 newIntercept = false;
#                 break;
#             }
#         if (newIntercept) {
#             tp[interceptDists.count & (MAX_INTERSECTIONS - 1)] = t;
#             interceptDists.count++;
#             results[rayIdx] += 1;
#         }
#     }
# }

# //--------------------------------------------------------------
# // Component 2: Morton code from github.com/Forceflow/libmorton
# // Copyright (c) 2016 Jeroen Baert, MIT license
# //--------------------------------------------------------------

# __device__ uint_fast32_t magicbit3D_masks32_encode[6] = 
# { 0x000003ff, 0, 0x30000ff, 0x0300f00f, 0x30c30c3, 0x9249249 };
# __device__ uint_fast64_t magicbit3D_masks64_encode[6] =
# { 0x1fffff, 0x1f00000000ffff, 0x1f0000ff0000ff,
#   0x100f00f00f00f00f, 0x10c30c30c30c30c3, 0x1249249249249249 };

# __device__ MORTON inline morton3D_SplitBy3bits(COORD a) {
#     const MORTON* masks = (sizeof(MORTON) <= 4) ?
#                            reinterpret_cast<const MORTON*>(magicbit3D_masks32_encode) :
#                            reinterpret_cast<const MORTON*>(magicbit3D_masks64_encode);
#     MORTON x = ((MORTON)a) & masks[0];
#     if (sizeof(MORTON) == 8) { x = (x | (uint_fast64_t)x << 32) & masks[1]; }
#     x = (x | x << 16) & masks[2];
#     x = (x | x << 8)  & masks[3];
#     x = (x | x << 4)  & masks[4];
#     x = (x | x << 2)  & masks[5];
#     return x;
# }

# __device__ MORTON inline m3D_e_magicbits(COORD x, COORD y, COORD z) {
#     return morton3D_SplitBy3bits(x) |
#           (morton3D_SplitBy3bits(y) << 1) |
#           (morton3D_SplitBy3bits(z) << 2);
# }

# //-----------------------------------------------------------------
# // Component 3: BVH accelerating structure / tree search algorithm
# //-----------------------------------------------------------------
# #define LEFT 0
# #define RIGHT 1
# #define ROOT -2
# #define MAX_COLLISIONS 32
# #define MAX_STACK_PTRS 64
# BVH_PREPROCESSOR_DIRECTIVE

# struct BVHNode
# {
#     AABB bounds;
#     BVHNode *childLeft, *childRight;
# #if COMPILE_NON_ESSENTIAL
#     BVHNode *parent, *self;
#     int idxSelf, idxChildL, idxChildR, isLeafChildL, isLeafChildR;
# #endif
#     int triangleID;
#     int atomic;
#     int rangeLeft, rangeRight;
# };
# struct CollisionList
# {
#     uint32_t hits[MAX_COLLISIONS];
#     int count;
# };

# __global__ void bytesInBVHNode(int &x){ x = sizeof(BVHNode); }
# __global__ void bytesInCollisionList(int &x){ x = sizeof(CollisionList); }

# typedef BVHNode* NodePtr;

# /* Auxiliary function */

# __global__ void kernelMortonCode(const float* __restrict__ vertices,
#                                  const int* __restrict__ triangles,
#                                  const float* __restrict__ minval,
#                                  const float* __restrict__ half_delta,
#                                  const float* __restrict__ inv_delta,
#                                  MORTON* __restrict__ morton,
#                                  int nTriangles)
# {
#     COORD vC[3];
#     const int i = blockIdx.x * blockDim.x + threadIdx.x;
#     if (i < nTriangles) {
#         const float *v0 = &vertices[3*triangles[3*i]],
#                     *v1 = &vertices[3*triangles[3*i+1]],
#                     *v2 = &vertices[3*triangles[3*i+2]];
#         //normalise centroid vertices (convert from real to integer)
#         //scale each dimension to use up to 21 bits
#         for (int c = 0; c < 3; c++) {
#             float centroid = ((v0[c] + v1[c] + v2[c]) / 3.0 - minval[c]);
#             vC[c] = static_cast<COORD>((centroid + half_delta[c]) * inv_delta[c]);
#         }
#         //- compute morton code
#         morton[i] = m3D_e_magicbits(vC[0], vC[1], vC[2]);
#     }
# }

# __device__ void computeTriangleBounds(const float *triangleVerts, AABB &box)
# {
#     const float *v0 = &triangleVerts[0],
#                 *v1 = &triangleVerts[3],
#                 *v2 = &triangleVerts[6];
#     if (v0[0] > v1[0]) {
#         if (v0[0] > v2[0]) { box.xMin = min(v1[0], v2[0]); box.xMax = v0[0]; }
#         else { box.xMin = v1[0]; box.xMax = v2[0]; }
#     }
#     else { // v1 >= v0
#         if (v1[0] > v2[0]) { box.xMax = v1[0]; box.xMin = min(v0[0], v2[0]); }
#         else { box.xMax = v2[0]; box.xMin = v0[0]; }
#     }
#     if (v0[1] > v1[1]) {
#         if (v0[1] > v2[1]) { box.yMin = min(v1[1], v2[1]); box.yMax = v0[1]; }
#         else { box.yMin = v1[1]; box.yMax = v2[1]; }
#     }
#     else {
#         if (v1[1] > v2[1]) { box.yMin = min(v0[1], v2[1]); box.yMax = v1[1]; }
#         else { box.yMin = v0[1]; box.yMax = v2[1]; }
#     }
#     if (v0[2] > v1[2]) {
#         if (v0[2] > v2[2]) { box.zMin = min(v1[2], v2[2]); box.zMax = v0[2]; }
#         else { box.zMin = v1[2]; box.zMax = v2[2]; }
#     }
#     else {
#         if (v1[2] > v2[2]) { box.zMax = v1[2]; box.zMin = min(v0[2], v2[2]); }
#         else { box.zMax = v2[2]; box.zMin = v0[2]; }
#     }
# }

# __device__ bool inline overlap(const AABB &queryBox, const BVHNode *candidate) {
#     const AABB &tBox = candidate->bounds;
#     if (queryBox.xMin > tBox.xMax || queryBox.xMax < tBox.xMin)
#         return false;
#     if (queryBox.yMin > tBox.yMax || queryBox.yMax < tBox.yMin)
#         return false;
#     if (queryBox.zMin > tBox.zMax || queryBox.zMax < tBox.zMin)
#         return false;
#     return true;
# }

# /* BVH implementation & related code */

# __device__ bool inline isLeaf(const BVHNode *node) {
#     return node->triangleID >= 0;
# }

# __device__ MORTON inline highestBit(int i, MORTON *morton)
# {   //find the highest differing bit between two keys: morton[i]
#     //and morton[i+1]. In practice, an XOR operation suffices.
#     return morton[i] ^ morton[i+1];
# }

# __device__ void bvhUpdateParent(BVHNode* node, BVHNode* internalNodes,
#                                 BVHNode *leafNodes, MORTON *morton, int nNodes)
# {
#     //This is a recursive function. It sets the parent node
#     //bounding box and traverses to the root node.

#     //allow only one thread to process a node
#     //  => for leaf nodes: always go through
#     //  => for internal nodes: only when both children have been discovered
#     if (atomicAdd(&node->atomic, 1) != 1)
#         return;
# #ifdef COMPILE_NON_ESSENTIAL
#     node->self = node;
# #endif
#     if (! isLeaf(node))
#     {   //expand bounds using children's axis-aligned bounding boxes
#         const BVHNode *dL = node->childLeft, //descendants
#                       *dR = node->childRight;
#         node->bounds.xMin = min(dL->bounds.xMin, dR->bounds.xMin);
#         node->bounds.xMax = max(dL->bounds.xMax, dR->bounds.xMax);
#         node->bounds.yMin = min(dL->bounds.yMin, dR->bounds.yMin);
#         node->bounds.yMax = max(dL->bounds.yMax, dR->bounds.yMax);
#         node->bounds.zMin = min(dL->bounds.zMin, dR->bounds.zMin);
#         node->bounds.zMax = max(dL->bounds.zMax, dR->bounds.zMax);
#     }
#     /* Deduce parent node index based on split properties described in
#        Ciprian Apetrei, "Fast and Simple Agglomerative LBVH Construction",
#        EG UK Computer Graphics & Visual Computing, 2014
#     */
#     int left = node->rangeLeft, right = node->rangeRight;
#     BVHNode *parent;
#     if (left == 0 || (right != nNodes - 1 &&
#         highestBit(right, morton) < highestBit(left - 1, morton)))
#     {
#         parent = &internalNodes[right];
#         parent->childLeft = node;
#         parent->rangeLeft = left;
# #ifdef COMPILE_NON_ESSENTIAL
#         parent->idxChildL = node->idxSelf;
#         parent->isLeafChildL = isLeaf(node);
#         node->parent = parent;
# #endif
#     }
#     else
#     {
#         parent = &internalNodes[left - 1];
#         parent->childRight = node;
#         parent->rangeRight = right;
# #ifdef COMPILE_NON_ESSENTIAL
#         parent->idxChildR = node->idxSelf;
#         parent->isLeafChildR = isLeaf(node);
#         node->parent = parent;
# #endif
#     }
#     if (left == 0 && right == nNodes - 1)
#     {   //current node represents the root,
#         //set left child in last internal node to root
#         internalNodes[nNodes - 1].childLeft = node;
#         node->triangleID = ROOT;
#         return;
#     }
#     bvhUpdateParent(parent, internalNodes, leafNodes, morton, nNodes);
# }

# __device__ bool inline bvhInsert(CollisionList &collisions, int value)
# {
#     //insert value into the hits[] array. Returned value indicates
#     //if buffer is full (true => not enough room for two elements).
#     collisions.hits[collisions.count++] = static_cast<uint32_t>(value);
#     return (collisions.count < MAX_COLLISIONS - 1)? false : true;
# }

# __device__ void bvhTraverse(
#            const AABB& queryBox, NodePtr &bvhNode,
#            NodePtr* &stackPtr, CollisionList &hits)
# {
#     //traverse nodes starting from the root iteratively
#     NodePtr node(bvhNode);
#     bool bufferFull(false);
#     do
#     {
#         //check each child node for overlap.
#         NodePtr childL = node->childLeft;
#         NodePtr childR = node->childRight;
#         bool overlapL = overlap(queryBox, childL);
#         bool overlapR = overlap(queryBox, childR);

#         //query overlaps a leaf node => report collision
#         if (overlapL && isLeaf(childL))
#             bufferFull = bvhInsert(hits, childL->triangleID);

#         if (overlapR && isLeaf(childR))
#             bufferFull |= bvhInsert(hits, childR->triangleID);

#         //query overlaps an internal node => traverse
#         bool traverseL = (overlapL && !isLeaf(childL));
#         bool traverseR = (overlapR && !isLeaf(childR));

#         if (!traverseL && !traverseR)
#             node = *--stackPtr; //pop
#         else
#         {
#             node = (traverseL) ? childL : childR;
#             if (traverseL && traverseR)
#                 *stackPtr++ = childR; //push
#         }
#     }
#     while (node != NULL && !bufferFull);
#     //when buffer is full, the input/output argument `bvhNode` is
#     //assigned a non-NULL NodePtr to permit resumption after the
#     //triangle candidates in the hits buffer have been tested.
#     bvhNode = node;
# }

# __global__ void kernelBVHReset(const float* __restrict__ vertices,
#                                const int* __restrict__ triangles,
#                                BVHNode* __restrict__ internalNodes,
#                                BVHNode* __restrict__ leafNodes,
#                                int* __restrict__ sortedObjectIDs, int nNodes)
# {
#    //reset parameters for internal and leaf nodes
#     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
#     if (i >= nNodes)
#         return;
#     //set triangle attributes in leaf
#     int t;
#     float triangleVerts[9];
#     leafNodes[i].triangleID = t = sortedObjectIDs[i];
#     for(int j = 0; j < 3; j++) {
#         int v = triangles[3*t+j];
#         for (int k = 0; k < 3; k++) {
#             triangleVerts[3*j+k] = vertices[3*v+k];
#         }
#     }
#     computeTriangleBounds(triangleVerts, leafNodes[i].bounds);

#     leafNodes[i].atomic = 1;
#     leafNodes[i].rangeLeft = i;
#     leafNodes[i].rangeRight = i;
# #ifdef COMPILE_NON_ESSENTIAL
#     leafNodes[i].idxSelf = i;
#     internalNodes[i].parent = NULL;
#     internalNodes[i].idxSelf = i;
# #endif
#     internalNodes[i].triangleID = -1;
#     internalNodes[i].atomic = 0;//first thread passes through
#     internalNodes[i].childLeft = internalNodes[i].childRight = NULL;
#     internalNodes[i].rangeLeft = internalNodes[i].rangeRight = -1;
#     if (nNodes == 1)
#     {
#         internalNodes[0].bounds = leafNodes[0].bounds;
#         internalNodes[0].childLeft = &leafNodes[0];
#     }
# }

# __global__ void kernelBVHConstruct(BVHNode *internalNodes, BVHNode *leafNodes,
#                                    MORTON *morton, int nNodes)
# {   //construct binary radix tree (Apetrei, 2014)
#     //select and update current node's parent in a bottom-up manner
#     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
#     if (i >= nNodes)
#         return;
#     //do this only for leaf nodes, information will propagate upward
#     bvhUpdateParent(&leafNodes[i], internalNodes, leafNodes, morton, nNodes);
# }

# __device__ void bvhFindCollisions1(const float* vertices,
#                                    const int* triangles,
#                                    const float* rayFrom,
#                                    const float* rayTo,
#                                    const AABB* rayBox,
#                                    const NodePtr bvhRoot,
#                                    CollisionList &collisions,
#                                    int* detected,
#                                    int rayIdx)
# {
#     NodePtr stack[MAX_STACK_PTRS];
#     NodePtr* stackPtr = stack;
#     *stackPtr++ = NULL;
#     NodePtr nextNode(bvhRoot);
#     do {
#         //find potential collisions (subset of triangles to test for)
#         //- if collisions buffer is full and there are more nodes
#         //  remaining to check, the returned nextNode won't be NULL
#         //- importantly, the stack content will persist in memory
#         collisions.count = 0;
#         bvhTraverse(rayBox[rayIdx], nextNode, stackPtr, collisions);

#         //check for actual intersections with the triangles found so far
#         int candidate = 0;
#         while (! detected[rayIdx] && (candidate < collisions.count)) {
#             int triangleID = collisions.hits[candidate++];
#             checkRayTriangleIntersection1(vertices, triangles, rayFrom, rayTo,
#                                           detected, rayIdx, triangleID);
#         }
#     }
#     while ((detected[rayIdx] == 0) && (nextNode != NULL));
# }

# // This version checks all candidates for the nearest intersection
# __device__ void bvhFindCollisions2(const float* vertices,
#                                    const int* triangles,
#                                    const float* rayFrom,
#                                    const float* rayTo,
#                                    const AABB* rayBox,
#                                    const NodePtr bvhRoot,
#                                    CollisionList &collisions,
#                                    int* intersectTriangle,
#                                    float* baryT,
#                                    float* baryU,
#                                    float* baryV,
#                                    int rayIdx)
# {
#     NodePtr stack[MAX_STACK_PTRS];
#     NodePtr* stackPtr = stack;
#     *stackPtr++ = NULL;
#     NodePtr nextNode(bvhRoot);
#     do {
#         collisions.count = 0;
#         bvhTraverse(rayBox[rayIdx], nextNode, stackPtr, collisions);

#         int candidate = 0;
#         while (candidate < collisions.count) {
#             int triangleID = collisions.hits[candidate++];
#             checkRayTriangleIntersection2(vertices, triangles, rayFrom, rayTo,
#                                           intersectTriangle, baryT, baryU,
#                                           baryV, rayIdx, triangleID);
#         }
#     }
#     while (nextNode != NULL);
# }

# // This version attempts to count unique ray-surface intersections
# __device__ void bvhFindCollisions3(const float* vertices,
#                                    const int* triangles,
#                                    const float* rayFrom,
#                                    const float* rayTo,
#                                    const AABB* rayBox,
#                                    const NodePtr bvhRoot,
#                                    CollisionList &collisions,
#                                    InterceptDistances &interceptDists,
#                                    int* detected,
#                                    int rayIdx)
# {
#     NodePtr stack[MAX_STACK_PTRS];
#     NodePtr* stackPtr = stack;
#     *stackPtr++ = NULL;
#     NodePtr nextNode(bvhRoot);

#     interceptDists.count = 0;
#     for (int i = 0; i < MAX_INTERSECTIONS; i++) {
#         interceptDists.t[i] = -1;
#     }
#     do {
#         collisions.count = 0;
#         bvhTraverse(rayBox[rayIdx], nextNode, stackPtr, collisions);

#         int candidate = 0;
#         while (candidate < collisions.count) {
#             int triangleID = collisions.hits[candidate++];
#             checkRayTriangleIntersection3(vertices, triangles, rayFrom, rayTo,
#                                           interceptDists, detected, rayIdx, triangleID);
#         }
#     }
#     while (nextNode != NULL);
# }

# // This version returns ray-surface intersection results via `detected`
# __global__ void kernelBVHIntersection1(const float* __restrict__ vertices,
#                                        const int* __restrict__ triangles,
#                                        const float* __restrict__ rayFrom,
#                                        const float* __restrict__ rayTo,
#                                        const BVHNode* __restrict__ internalNodes,
#                                        const AABB* __restrict__ rayBox,
#                                        CollisionList* __restrict__ raytriBoxHitIDs,
#                                        int* __restrict__ detected,
#                                        int numTriangles, int numRays)
# {
#     __shared__ NodePtr bvhRoot;
#     __shared__ int stride;
#     if (threadIdx.x == 0) {
#         bvhRoot = internalNodes[numTriangles-1].childLeft;
#         stride = gridDim.x * blockDim.x;
#     }
#     __syncthreads();

#     int threadStartIdx = blockIdx.x * blockDim.x + threadIdx.x;
#     int bufferIdx = threadStartIdx;
#     //iterate if numRays exceeds dimension of thread-block
#     for (int idx = threadStartIdx; idx < numRays; idx += stride) {
#         if (idx < numRays) {
#             //access thread-specific collision array
#             CollisionList &collisions = raytriBoxHitIDs[bufferIdx];
#             bvhFindCollisions1(vertices, triangles, rayFrom, rayTo, rayBox,
#                                bvhRoot, collisions, detected, idx);
#         }
#     }
# }

# // This version returns results via `intersectTriangle` and barycentric coordinates
# __global__ void kernelBVHIntersection2(const float* __restrict__ vertices,
#                                        const int* __restrict__ triangles,
#                                        const float* __restrict__ rayFrom,
#                                        const float* __restrict__ rayTo,
#                                        const BVHNode* __restrict__ internalNodes,
#                                        const AABB* __restrict__ rayBox,
#                                        CollisionList* __restrict__ raytriBoxHitIDs,
#                                        int* __restrict__ intersectTriangle,
#                                        float* __restrict__ baryT,
#                                        float* __restrict__ baryU,
#                                        float* __restrict__ baryV,
#                                        int numTriangles, int numRays)
# {
#     __shared__ NodePtr bvhRoot;
#     __shared__ int stride;
#     if (threadIdx.x == 0) {
#         bvhRoot = internalNodes[numTriangles-1].childLeft;
#         stride = gridDim.x * blockDim.x;
#     }
#     __syncthreads();

#     int threadStartIdx = blockIdx.x * blockDim.x + threadIdx.x;
#     int bufferIdx = threadStartIdx;

#     for (int idx = threadStartIdx; idx < numRays; idx += stride) {
#         if (idx < numRays) {
#             CollisionList &collisions = raytriBoxHitIDs[bufferIdx];
#             bvhFindCollisions2(vertices, triangles, rayFrom, rayTo, rayBox,
#                                bvhRoot, collisions, intersectTriangle,
#                                baryT, baryU, baryV, idx);
#         }
#     }
# }

# // This version counts number of unique surface intersections (limited to < 32)
# __global__ void kernelBVHIntersection3(const float* __restrict__ vertices,
#                                        const int* __restrict__ triangles,
#                                        const float* __restrict__ rayFrom,
#                                        const float* __restrict__ rayTo,
#                                        const BVHNode* __restrict__ internalNodes,
#                                        const AABB* __restrict__ rayBox,
#                                        CollisionList* __restrict__ raytriBoxHitIDs,
#                                        InterceptDistances* __restrict__ rayInterceptDists,
#                                        int* __restrict__ detected,
#                                        int numTriangles, int numRays)
# {
#     __shared__ NodePtr bvhRoot;
#     __shared__ int stride;
#     if (threadIdx.x == 0) {
#         bvhRoot = internalNodes[numTriangles-1].childLeft;
#         stride = gridDim.x * blockDim.x;
#     }
#     __syncthreads();

#     int threadStartIdx = blockIdx.x * blockDim.x + threadIdx.x;
#     int bufferIdx = threadStartIdx;

#     for (int idx = threadStartIdx; idx < numRays; idx += stride) {
#         if (idx < numRays) {
#             CollisionList &collisions = raytriBoxHitIDs[bufferIdx];
#             InterceptDistances &interceptDists = rayInterceptDists[bufferIdx];
#             bvhFindCollisions3(vertices, triangles, rayFrom, rayTo, rayBox,
#                                bvhRoot, collisions, interceptDists, detected, idx);
#         }
#     }
# }
# """

# def get_cuda_template():
#     return cuda_template
