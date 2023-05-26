## Issues: GPU implementation of a ray-surface intersection algorithm in CUDA

Issue number, followed by the commit where issue is found
- Description
- Resolution
<hr>

1. SHA 2175b498a14bcb32
- Numerical error due to internal float32 representation may be significant
  when working with vertices and rays expressed in [UTM coordinates](
  https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system).
  This may produce unexpected results due to rounding. Refer to details given
  in ```scripts/gpu_ray_surface_intersect.py```, see ```PyGpuRSI.translate_data```
  doc string.
- For the CUDA command line API (see ```gpu_ray_surface_intersect``` usage in
  [sec. 2.3](doc/gpu-rsi-doc.pdf#subsection.2.3)), the caller is responsible
  for centering the input data to minimise its effective range. For the python
  API (see ```PyGpuRSI.test``` method), translation will be performed
  automatically. The minimum spatial coordinates with respect to the first
  supplied surface will be subtracted from the vertices and rays. This offset
  remains unchanged in subsequent calls and the same adjustments will be made
  internally each time, until the PyGpuRSI object goes out of scope.

2. SHA 77cf088a4525e117
- An exception "illegal memory access was encountered in
  gpu_ray_surface_intersect.cu at line 270" is thrown when the surface
  contains only 1 triangle. See test case supplied by Ronan Danno in
  [```issues/0002```](issues/0002/test_case.md).
- This most likely is due to the construction of the binary radix tree.
  The bounding volume hierarchy traversal algorithm expects at least one split
  node where the left and right child nodes are defined. For this corner case,
  these are undefined at the root node. This problem disappears if the mesh
  contains multiple (>= 2) triangles. For simplicity, this condition is
  enforced in ```gpu_ray_surface_intersect.cu main()```. When
  ```nTriangles = readData(fileTriangles, h_triangles, 3, quietMode);```
  equals one, three coincident vertices (a degenerate triangle) are added to
  the host memory vector ```h_triangles```.

3. SHA 1edbd3e39f2f61e7
- The byte size in the cudaMalloc statement for d_interceptDists should
  depend on sizeof(InterceptDistances), not sizeof(CollisionList). This was
  a copy and paste error. Fortuitously, both objects have the same memory
  footprint, as uint32 and float (under CUDA) generally take up 4 bytes.
  So, this correction is more about semantics or the intent of the statement.
- Change int sz_interceptDists(gridXLambda * blockX * sizeof(CollisionList))
  to int sz_interceptDists(gridXLambda * blockX * sizeof(InterceptDistances))

4. SHA 6ecfbc8e057cd16f
- CUDA program reports spurious intersecting points on rare occasions where
  the ray runs parallel to, or lies in, the plane of a mesh triangle. In this
  situation, the determinant (det) in the Moller-Trumbore algorithm ought to
  be zero, however it remains above the zero threshold (EPSILON is currently
  fixed to 1e-6) due to rounding errors. Instead of aborting the test, it
  proceeds to calculate the (u,v) barycentric coordinates using precarious,
  unstable values. This produces a bogus intersection that deviates from the
  line segment. Refer to full analysis in [```issues/0004```](issues/0004/test_case.md)
- Instead of using a fixed absolute tolerance, epsilon is scaled by the length
  of the triangle edges (vAB, vAC) and ray direction vector (vPQ) to counter
  noise amplification effects. This increases robustness when the mesh
  triangles are very large or highly variable in size.
