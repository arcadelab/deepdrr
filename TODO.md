
# Features
<!-- - Use winding order in renderer -->
<!-- - Confirm mesh cutout -->
<!-- - Morph targets -->
- Mesh priorities (is necessary?)

# Optimizations
<!-- - On gpu sort -->
<!-- - On gpu ray generation -->
- Mesh instancing
- On gpu ray from and to gen
- Save memory by merging same-material mesh raycast hits
- Save memory by using signed distances to represent front/back hits
- Data stay on GPU
- Reuse tree for non blend meshes
- Use rasterization method
- Fast mode rasterization
- Free all new allocs
- On GPU morph targets