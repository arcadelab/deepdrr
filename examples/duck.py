import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
# sudo apt-get install llvm-6.0 freeglut3 freeglut3-dev
from deepdrr.pyrender import Mesh, Scene, Viewer
import deepdrr.pyrender as pyrender

from deepdrr.pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer, RenderFlags
from io import BytesIO
import numpy as np
import trimesh
import requests
import time
from deepdrr.pyrender.constants import DRRMode


duck_source = "https://github.com/KhronosGroup/glTF-Sample-Models/raw/master/2.0/Duck/glTF-Binary/Duck.glb"

# duck = trimesh.load("./models/stress80k.glb")
# duck = trimesh.load(BytesIO(requests.get(duck_source).content), file_type='glb')
# duckmesh = Mesh.from_trimesh(list(duck.geometry.values())[0])
duckmesh = Mesh.from_trimesh(trimesh.load("./models/suzanne_stress.stl"))
# duckmesh = Mesh.from_trimesh(trimesh.load("./models/suzanne.stl"))
scene = Scene(ambient_light=np.array([1.0, 1.0, 1.0, 1.0]), bg_color=[100, 100, 100])

duckmesh_pose = np.array([
    [0.0, 0.0, -1.0, 0.0],
    [1.0, 0.0,  0.0, 0.0],
    [0.0, 1.0,  0.0, 0.0],
    [0.0, 0.0,  0.0, 1.0],
])

scene.add(duckmesh, pose=duckmesh_pose)


cam = PerspectiveCamera(yfov=(np.pi / 3.0))
# cam_pose = np.array([
#     [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 1.5],
#     [1.0, 0.0,           0.0,           0.0],
#     [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 1.4],
#     [0.0,  0.0,           0.0,          1.0]
# ])
# cam_pose = np.array([
#     [1.0, 0.0, 0.0, 0.0],
#     [0.0, 1.0, 0.0, 100.0],
#     [0.0, 0.0, -1.0, -300.0],
#     [0.0, 0.0, 0.0, 1.0]
# ])
import random

# random betwee -1 and 1
randval = random.uniform(-1, 1)

cam_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, randval],
    [0.0, 0.0, -1.0, -2.0],
    [0.0, 0.0, 0.0, 1.0]
])

scene.add(cam, pose=cam_pose)

r = pyrender.OffscreenRenderer(viewport_width=640,
                                viewport_height=480,
                                point_size=1.0,
                                max_dual_peel_layers=8)
zfar = 10
def render():
    # color, depth = r.render(scene, drr_mode=DRRMode.ERROR)

    # error_mask = np.abs(color) > 1e-6
    # color, depth = r.render(scene, drr_mode=DRRMode.DENSITY)
    # # color, depth = r.render(scene, drr_mode=DRRMode.DENSITY)
    # # color, depth = r.render(scene, drr_mode=DRRMode.DENSITY)
    # # color, depth = r.render(scene, drr_mode=DRRMode.DENSITY)
    
    # color[error_mask] = 0

    # ims = r.render(scene, drr_mode=DRRMode.DENSITY, flags=RenderFlags.RGBA, zfar=zfar)
    ims = r.render(scene, drr_mode=DRRMode.BACKDIST, flags=RenderFlags.RGBA, zfar=zfar)
    # color, depth = r.render(scene, drr_mode=DRRMode.FRONTDIST)



    return ims


ims = render()


N=1000
start_time = time.perf_counter()
for i in range(N):
    ims = render()

print("FPS: ", N / (time.perf_counter() - start_time))

exit()

r.delete()

# set all values less than 0 to 0
# color[color < 0] = 0
# color = np.abs(color)



# print(f"{color.shape=} {color.dtype=}")

# print(f"{np.amin(color)=} {np.amax(color)=}")
# print(f"{np.unique(color, return_counts=True)=}")

# print(f"{depth.shape=} {depth.dtype=}")

# print(f"{np.amin(depth)=} {np.amax(depth)=}")
# print(f"{np.unique(depth, return_counts=True)=}")

# save to file
import cv2
# remapped = np.interp(color[:,:,::-1], (1, 5), (0, 255)).astype(np.uint8)
# front = color[:,:,0]
# back = color[:,:,1]

# print(f"{front.shape=} {front.dtype=}")

# print(f"{np.amin(front)=} {np.amax(front)=}")
# print(f"{np.unique(front, return_counts=True)=}")


# print(f"{back.shape=} {back.dtype=}")

# print(f"{np.amin(back)=} {np.amax(back)=}")
# print(f"{np.unique(back, return_counts=True)=}")

# remapped = np.interp(front, (np.amin(front[front>-zfar+.001]), np.amax(front)), (0, 255)).astype(np.uint8)
# remapped_depth = np.interp(back, (np.amin(front[front>-zfar+.001]), np.amax(back)), (0, 255)).astype(np.uint8)
# # remapped_depth = np.interp(back, (np.amin(back), np.amax(back)), (0, 255)).astype(np.uint8)
# cv2.imwrite('asdfsa.png', remapped)
# cv2.imwrite('asdfsa_depth.png', remapped_depth)

# print("done")

# ims = [color[:,:,0], color[:,:,1], depth[:,:,0], depth[:,:,1], dfsa[:,:,0], dfsa[:,:,1]]
ims = [x for im in ims for x in [im[:,:,0], im[:,:,1]] ]
ims[0][ims[1]!=0] = 0 # for density error

for i, im in enumerate(ims):
    print(f"")
    print(f"{i=}")
    print(f"{im.shape=} {im.dtype=}")
    print(f"{np.amin(im)=} {np.amax(im)=}")
    print(f"{np.unique(im, return_counts=True)=}")

    above_zfar = im[im>-zfar+.001]
    if len(above_zfar) == 0:
        above_zfar = im
    remapped = np.interp(im, (np.amin(above_zfar), np.amax(im)), (0, 255)).astype(np.uint8)
    cv2.imwrite(f'asdfsa{i}.png', remapped)
