# Ops
- Implement in PelvisVR
- Cleanup
- Add tests
- Ask about     from pycuda.autoinit import context
- Get rid of macros
- chmod 777 /dev/dri/renderD128

# Features
<!-- - Use winding order in renderer -->
<!-- - Confirm mesh cutout -->
<!-- - Morph targets -->
<!-- - Min/max alpha -->
<!-- - Integrate API for meshes and volumes -->
- Mesh priorities (is necessary?)
- Fix attenuate outside volume
- scatter?
- better zero mesh handling
- zero volume handling

# Optimizations
<!-- - On gpu sort -->
<!-- - On gpu ray generation -->
- Mesh instancing
- Primitive merging
- On gpu ray from and to gen
- Save memory by merging same-material mesh raycast hits
- Save memory by using signed distances to represent front/back hits
- Data stay on GPU
- Reuse tree for non blend meshes
- Use rasterization method
- Fast mode rasterization
- Free all new allocs
- On GPU morph targets