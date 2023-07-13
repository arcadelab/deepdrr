# TODO
- Add dep on pycuda with gl support
- Add tests
- Mesh priorities (is necessary?)
- Scatter?
- Zero mesh handling
- Zero volume handling
- Primitive merging
- Save memory by using signed distances to represent front/back hits
- Free all new allocs

<!-- - Return peeling array up to 8 -->
<!-- - Support multi material -->
<!-- - Support more than 8 peels -->
<!-- - Zero copy buffers to cuda -->

# Questions
- Worth having a high-density mode, renders much faster, to handle nearly all cases?
- Was deepdrr not locked to fixed resolution before?

# Features
<!-- - Use winding order in renderer -->
<!-- - Confirm mesh cutout -->
<!-- - Morph targets -->
<!-- - Min/max alpha -->
<!-- - Integrate API for meshes and volumes -->
<!-- - Fix attenuate outside volume -->


# Optimizations
<!-- - On gpu sort -->
<!-- - On gpu ray generation -->
<!-- - Mesh instancing -->
<!-- - Save memory by merging same-material mesh raycast hits -->
<!-- - On gpu ray from and to gen -->

<!-- - Data stay on GPU -->
<!-- - Reuse tree for non blend meshes -->
<!-- - Use rasterization method -->
<!-- - Fast mode rasterization -->
<!-- - On GPU morph targets -->

# Ops
- Remove todo, .idea, .vscode
- Implement in PelvisVR
- Cleanup
- Ask about     from pycuda.autoinit import context
- Get rid of macros
- chmod 777 /dev/dri/renderD128
