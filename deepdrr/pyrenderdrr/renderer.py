import numpy as np
from OpenGL.GL import *
from pathlib import Path
from pyrender.constants import (RenderFlags, ProgramFlags, BufFlags)
from pyrender.shader_program import ShaderProgramCache
from .material import DRRMaterial
from ..utils.cuda_utils import check_cudart_err, format_cudart_err
from cuda import cudart
import sys

class DRRMode:
    NONE = 0
    DENSITY = 1
    DIST = 2
    SEG = 3
    MESH_SUB = 4

GL_COLOR_ATTACHMENT_LIST = [
    GL_COLOR_ATTACHMENT0,
    GL_COLOR_ATTACHMENT1,
    GL_COLOR_ATTACHMENT2,
    GL_COLOR_ATTACHMENT3,
    GL_COLOR_ATTACHMENT4,
    GL_COLOR_ATTACHMENT5,
    GL_COLOR_ATTACHMENT6,
    GL_COLOR_ATTACHMENT7,
]


class Renderer(object):
    """Class for handling all rendering operations on a scene.

    Note
    ----
    This renderer relies on the existence of an OpenGL context and
    does not create one on its own.

    Parameters
    ----------
    viewport_width : int
        Width of the viewport in pixels.
    viewport_height : int
        Width of the viewport height in pixels.
    point_size : float, optional
        Size of points in pixels. Defaults to 1.0.
    """

    def __init__(self, viewport_width, viewport_height, point_size=1.0, num_peel_passes=None, num_mesh_mesh_layers=None, prim_unqiue_materials=None):
        self.dpscale = 1

        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.point_size = point_size
        self.num_peel_passes = num_peel_passes
        self.num_mesh_mesh_layers = num_mesh_mesh_layers
        self.prim_unqiue_materials = prim_unqiue_materials

        assert self.num_peel_passes is not None, "num_peel_passes must be set"
        assert self.num_peel_passes > 0, "num_peel_passes must be > 0"
        assert self.num_mesh_mesh_layers is not None, "num_mesh_mesh_layers must be set"
        assert self.num_mesh_mesh_layers > 0, "num_mesh_mesh_layers must be > 0"
        assert self.prim_unqiue_materials is not None, "prim_unqiue_materials must be set"
        assert self.prim_unqiue_materials >= 0, "prim_unqiue_materials must be >= 0"

        # Optional framebuffer for offscreen renders
        self._main_fb = None
        self._main_cb = None
        self._main_db = None
        self._fb_initialized = False
        self._main_fb_dims = (None, None)
        self.g_peelTexId = None
        self.g_peelFboIds = None
        self.g_densityTexId = None
        self.g_densityFboId = None
        self.g_peelTexSubId = None
        self.g_peelFboSubIds = None

        self.subtractive_reg_ims = None
        self.mesh_sub_reg_ims = None
        self.additive_reg_ims = None
        # self.additive_reg_im = None

        # Shader Program Cache
        d = Path(__file__).resolve().parent
        shader_dir = d / 'shaders'
        self._program_cache = ShaderProgramCache(shader_dir=shader_dir)
        self._meshes = set()
        self._mesh_textures = set()
        self._texture_alloc_idx = 0

    @property
    def viewport_width(self):
        """int : The width of the main viewport, in pixels.
        """
        return self._viewport_width

    @viewport_width.setter
    def viewport_width(self, value):
        self._viewport_width = self.dpscale * value

    @property
    def viewport_height(self):
        """int : The height of the main viewport, in pixels.
        """
        return self._viewport_height

    @viewport_height.setter
    def viewport_height(self, value):
        self._viewport_height = self.dpscale * value

    @property
    def point_size(self):
        """float : The size of screen-space points, in pixels.
        """
        return self._point_size

    @point_size.setter
    def point_size(self, value):
        self._point_size = float(value)

    def render(self, scene, flags, seg_node_map=None, drr_mode=DRRMode.NONE, zfar=0, mat=None, mat_idx=None, layer_id=None, tex_idx=None):
        self._update_context(scene, flags)

        if drr_mode == DRRMode.DIST:
            for i in range(self.num_peel_passes):
                retval = self._forward_pass(scene, flags, seg_node_map=seg_node_map, drr_mode=drr_mode, zfar=zfar, peelnum=i, mat=mat, mat_idx=mat_idx, layer_id=layer_id, tex_idx=tex_idx)
        elif drr_mode == DRRMode.MESH_SUB:
            retval = self._forward_pass(scene, flags, seg_node_map=seg_node_map, drr_mode=drr_mode, zfar=zfar, peelnum=None, mat=mat, mat_idx=mat_idx, layer_id=layer_id, tex_idx=tex_idx)
        elif drr_mode == DRRMode.DENSITY:
            retval = self._forward_pass(scene, flags, seg_node_map=seg_node_map, drr_mode=drr_mode, zfar=zfar, peelnum=None, mat=mat, mat_idx=mat_idx, layer_id=layer_id, tex_idx=tex_idx)
        elif drr_mode == DRRMode.SEG:
            retval = self._forward_pass(scene, flags, seg_node_map=seg_node_map, drr_mode=drr_mode, zfar=zfar, peelnum=None, mat=mat, mat_idx=mat_idx, layer_id=layer_id, tex_idx=tex_idx)
        else:
            raise NotImplementedError

        return retval

    def delete(self):
        """Free all allocated OpenGL resources.
        """
        # Free shaders
        self._program_cache.clear()

        # Free meshes
        for mesh in self._meshes:
            for p in mesh.primitives:
                p.delete()

        # Free textures
        for mesh_texture in self._mesh_textures:
            mesh_texture.delete()

        self._meshes = set()
        self._mesh_textures = set()
        self._texture_alloc_idx = 0

        self._delete_main_framebuffer()
        # self._delete_shadow_framebuffer()

    def __del__(self):
        try:
            self.delete()
        except Exception:
            pass

    ###########################################################################
    # Rendering passes
    ###########################################################################

    def _forward_pass(self, scene, flags, seg_node_map=None, drr_mode=DRRMode.NONE, zfar=None, peelnum=None, mat=None, mat_idx=None, layer_id=None, tex_idx=None):
        # Set up viewport for render
        self._configure_forward_pass_viewport(flags, drr_mode=drr_mode, peelnum=peelnum, mat_idx=mat_idx, layer_id=layer_id)

        # Clear it
        if drr_mode == DRRMode.DIST:
            glClearColor(-zfar, -zfar, -zfar, -zfar)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        elif drr_mode == DRRMode.DENSITY:
            glClearColor(0, 0, 0, 0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        elif drr_mode == DRRMode.MESH_SUB:
            # glClearColor(0, 0, 0, 0)
            # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            # Don't clear-- we are rendering directly onto previous layer
            pass
        elif drr_mode == DRRMode.SEG:
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            

        glDisable(GL_MULTISAMPLE)

        # Set up camera matrices
        V, P = self._get_camera_matrices(scene)

        program = None
        # Now, render each object in sorted order
        for node in self._sorted_mesh_nodes(scene):
            mesh = node.mesh

            # Skip the mesh if it's not visible
            if not mesh.is_visible:
                continue

            # If SEG, set color
            if drr_mode == DRRMode.SEG:
                if seg_node_map is None:
                    color = (255)
                else:
                    if not hasattr(mesh, "originmesh") or mesh.originmesh is None or mesh.originmesh not in seg_node_map:
                        continue
                    color = seg_node_map[mesh.originmesh]
                if not isinstance(color, (list, tuple, np.ndarray)):
                    color = np.repeat(color, 3)
                else:
                    color = np.asanyarray(color)
                color = color / 255.0

            for primitive in mesh.primitives:
                if not isinstance(primitive.material, DRRMaterial):
                    continue
                if drr_mode == DRRMode.DENSITY and not primitive.material.additive:
                    continue
                if drr_mode == DRRMode.DIST and not primitive.material.subtractive:
                    continue
                if mat is not None and primitive.material.drrMatName != mat:
                    continue
                if layer_id is not None and primitive.material.layer != layer_id:
                    continue

                # First, get and bind the appropriate program
                program = self._get_primitive_program(
                    primitive, flags, ProgramFlags.USE_MATERIAL, drr_mode=drr_mode, peelnum=peelnum
                )
                program._bind()

                # Set the camera uniforms
                program.set_uniform('V', V)
                program.set_uniform('P', P)
                program.set_uniform(
                    'cam_pos', scene.get_pose(scene.main_camera_node)[:3, 3]
                )
                if drr_mode == DRRMode.SEG:
                    program.set_uniform('color', color)

                # Finally, bind and draw the primitive
                self._bind_and_draw_primitive(
                    primitive=primitive,
                    pose=scene.get_pose(node),
                    program=program,
                    flags=flags,
                    drr_mode=drr_mode,
                    zfar=zfar,
                    peelnum=peelnum,
                    tex_idx=tex_idx,
                )
                self._reset_active_textures()
        # Unbind the shader and flush the output
        if program is not None:
            program._unbind()
        # glFlush() # TODO: I don't think this is needed for offscreen

        # if peelnum == self.max_dual_peel_layers-1 or drr_mode == DRRMode.DENSITY:
        #     return self._read_main_framebuffer(scene, flags, drr_mode=drr_mode, front=front)
        # return []

        if drr_mode == DRRMode.SEG:
            return self._read_main_framebuffer(scene, flags)

    def _bind_and_draw_primitive(self, primitive, pose, program, flags, drr_mode=DRRMode.NONE, zfar=3, peelnum=0, tex_idx=None):
        # Set model pose matrix
        program.set_uniform('M', pose)

        # Bind mesh buffers
        primitive._bind()

        # Bind mesh material
        material = primitive.material
        assert isinstance(primitive.material, DRRMaterial), "Material must be DRRMaterial"

        if drr_mode == DRRMode.DIST:
            if peelnum > 0:
                glActiveTexture(GL_TEXTURE0 + 0)
                glBindTexture(GL_TEXTURE_RECTANGLE, self.g_peelTexId[peelnum - 1])
                program.set_uniform('DepthBlenderTex', 0)
                glActiveTexture(GL_TEXTURE0)

            program.set_uniform('MaxDepth', float(zfar))

            glEnable(GL_BLEND)
            glBlendEquation(GL_MAX)
            glBlendFunc(GL_ONE, GL_ONE)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glDisable(GL_CULL_FACE)
        elif drr_mode == DRRMode.DENSITY:
            density = material.density
            assert density is not None, "Density must be set for DRRMode.DENSITY"
            assert isinstance(density, float), "Density must be float"
            if density < 0:
                density = 0
            program.set_uniform('density', float(density))

            glEnable(GL_BLEND)
            glBlendEquation(GL_FUNC_ADD)
            glBlendFunc(GL_ONE, GL_ONE)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glDisable(GL_CULL_FACE)
        elif drr_mode == DRRMode.SEG:
            glEnable(GL_BLEND)
            glBlendFunc(GL_ONE, GL_ZERO)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
        elif drr_mode == DRRMode.MESH_SUB:
            density = material.density
            assert density is not None, "Density must be set for DRRMode.DENSITY"
            assert isinstance(density, float), "Density must be float"
            if density < 0:
                raise ValueError("Density must be >= 0")
                # density = 0
            program.set_uniform('density', float(density))

            assert tex_idx is not None, "tex_idx must be set for DRRMode.MESH_SUB"

            glActiveTexture(GL_TEXTURE0 + 0)
            glBindTexture(GL_TEXTURE_RECTANGLE, self.g_peelTexSubId[tex_idx])
            program.set_uniform('DepthBlenderTex', 0)
            glActiveTexture(GL_TEXTURE0)

            program.set_uniform('MaxDepth', float(zfar))

            glEnable(GL_BLEND)
            glBlendEquation(GL_FUNC_ADD)
            glBlendFunc(GL_ONE, GL_ONE)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glDisable(GL_CULL_FACE)
        else:
            raise NotImplementedError

        # Render mesh
        n_instances = 1
        if primitive.poses is not None:
            n_instances = len(primitive.poses)

        if primitive.indices is not None:
            glDrawElementsInstanced(
                primitive.mode, primitive.indices.size, GL_UNSIGNED_INT,
                ctypes.c_void_p(0), n_instances
            )
        else:
            glDrawArraysInstanced(
                primitive.mode, 0, len(primitive.positions), n_instances
            )

        # Unbind mesh buffers
        primitive._unbind()

    def _sorted_mesh_nodes(self, scene):
        cam_loc = scene.get_pose(scene.main_camera_node)[:3, 3]
        solid_nodes = []
        trans_nodes = []
        for node in scene.mesh_nodes:
            mesh = node.mesh
            if mesh.is_transparent:
                trans_nodes.append(node)
            else:
                solid_nodes.append(node)

        # TODO BETTER SORTING METHOD
        trans_nodes.sort(
            key=lambda n: -np.linalg.norm(scene.get_pose(n)[:3, 3] - cam_loc)
        )
        solid_nodes.sort(
            key=lambda n: -np.linalg.norm(scene.get_pose(n)[:3, 3] - cam_loc)
        )

        return solid_nodes + trans_nodes

    def _sorted_nodes_by_distance(self, scene, nodes, compare_node):
        nodes = list(nodes)
        compare_posn = scene.get_pose(compare_node)[:3, 3]
        nodes.sort(key=lambda n: np.linalg.norm(
            scene.get_pose(n)[:3, 3] - compare_posn)
                   )
        return nodes

    ###########################################################################
    # Context Management
    ###########################################################################

    def _update_context(self, scene, flags):

        # Update meshes
        scene_meshes = scene.meshes

        # Add new meshes to context
        for mesh in scene_meshes - self._meshes:
            for p in mesh.primitives:
                p._add_to_context()

        # Remove old meshes from context
        for mesh in self._meshes - scene_meshes:
            for p in mesh.primitives:
                p.delete()

        self._meshes = scene_meshes.copy()

        # Update mesh textures
        mesh_textures = set()
        for m in scene_meshes:
            for p in m.primitives:
                mesh_textures |= p.material.textures

        # Add new textures to context
        for texture in mesh_textures - self._mesh_textures:
            texture._add_to_context()

        # Remove old textures from context
        for texture in self._mesh_textures - mesh_textures:
            texture.delete()

        self._mesh_textures = mesh_textures.copy()

    ###########################################################################
    # Texture Management
    ###########################################################################

    def _bind_texture(self, texture, uniform_name, program):
        """Bind a texture to an active texture unit and return
        the texture unit index that was used.
        """
        tex_id = self._get_next_active_texture()
        glActiveTexture(GL_TEXTURE0 + tex_id)
        texture._bind()
        program.set_uniform(uniform_name, tex_id)

    def _get_next_active_texture(self):
        val = self._texture_alloc_idx
        self._texture_alloc_idx += 1
        return val

    def _reset_active_textures(self):
        self._texture_alloc_idx = 0

    ###########################################################################
    # Camera Matrix Management
    ###########################################################################

    def _get_camera_matrices(self, scene):
        main_camera_node = scene.main_camera_node
        if main_camera_node is None:
            raise ValueError('Cannot render scene without a camera')
        P = main_camera_node.camera.get_projection_matrix(
            width=self.viewport_width, height=self.viewport_height
        )
        pose = scene.get_pose(main_camera_node)
        V = np.linalg.inv(pose)  # V maps from world to camera
        return V, P

    ###########################################################################
    # Shader Program Management
    ###########################################################################

    def _get_primitive_program(self, primitive, flags, program_flags, drr_mode=DRRMode.NONE, peelnum=0):
        vertex_shader = None
        fragment_shader = None
        geometry_shader = None
        defines = {}

        if drr_mode == DRRMode.DIST:
            if peelnum == 0:
                vertex_shader = 'dual_peeling_init_vertex.glsl'
                fragment_shader = 'dual_peeling_init_fragment.glsl'
            else:
                vertex_shader = 'dual_peeling_peel_vertex.glsl'
                fragment_shader = 'dual_peeling_peel_fragment.glsl'
        elif drr_mode ==  DRRMode.DENSITY:
            vertex_shader = 'density.vert'
            fragment_shader = 'density.frag'
        elif drr_mode == DRRMode.MESH_SUB:
            vertex_shader = 'density_between.vert'
            fragment_shader = 'density_between.frag'
        elif drr_mode == DRRMode.SEG:
            vertex_shader = 'segmentation.vert'
            fragment_shader = 'segmentation.frag'
        else:
            raise NotImplementedError

        # Set up vertex buffer DEFINES
        bf = primitive.buf_flags
        buf_idx = 1
        if bf & BufFlags.NORMAL:
            defines['NORMAL_LOC'] = buf_idx
            buf_idx += 1
        if bf & BufFlags.TANGENT:
            defines['TANGENT_LOC'] = buf_idx
            buf_idx += 1
        if bf & BufFlags.TEXCOORD_0:
            defines['TEXCOORD_0_LOC'] = buf_idx
            buf_idx += 1
        if bf & BufFlags.TEXCOORD_1:
            defines['TEXCOORD_1_LOC'] = buf_idx
            buf_idx += 1
        if bf & BufFlags.COLOR_0:
            defines['COLOR_0_LOC'] = buf_idx
            buf_idx += 1
        if bf & BufFlags.JOINTS_0:
            defines['JOINTS_0_LOC'] = buf_idx
            buf_idx += 1
        if bf & BufFlags.WEIGHTS_0:
            defines['WEIGHTS_0_LOC'] = buf_idx
            buf_idx += 1
        defines['INST_M_LOC'] = buf_idx

        program = self._program_cache.get_program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
            geometry_shader=geometry_shader,
            defines=defines
        )

        if not program._in_context():
            program._add_to_context()

        return program

    ###########################################################################
    # Viewport Management
    ###########################################################################

    def _configure_forward_pass_viewport(self, flags, drr_mode=DRRMode.NONE, peelnum=None, mat_idx=None, layer_id=None):
        self._configure_main_framebuffer()

        if drr_mode == DRRMode.DENSITY:
            glBindFramebuffer(GL_FRAMEBUFFER, self.g_densityFboId[layer_id * self.prim_unqiue_materials + mat_idx])
        elif drr_mode == DRRMode.SEG:
            # glBindFramebuffer(GL_FRAMEBUFFER, self._main_fb)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
        elif drr_mode == DRRMode.MESH_SUB:
            glBindFramebuffer(GL_FRAMEBUFFER, self.g_densityFboId[layer_id * self.prim_unqiue_materials + mat_idx])
        elif drr_mode == DRRMode.DIST:
            glBindFramebuffer(GL_FRAMEBUFFER, self.g_peelFboIds[peelnum])
        else:
            raise NotImplementedError

        glDrawBuffer(GL_COLOR_ATTACHMENT_LIST[0])

        glViewport(0, 0, self.viewport_width, self.viewport_height)
        glDisable(GL_DEPTH_TEST)
        glDepthFunc(GL_ALWAYS)
        glDepthRange(0.0, 1.0)

    ###########################################################################
    # Framebuffer Management
    ###########################################################################

    def _configure_main_framebuffer(self):
        def listify(x):
            if isinstance(x, (np.ndarray)):
                return x
            return np.array([x])

        # If mismatch with prior framebuffer, delete it
        if (self._fb_initialized and
                self.viewport_width != self._main_fb_dims[0] or
                self.viewport_height != self._main_fb_dims[1]):
            self._delete_main_framebuffer()

        # If framebuffer doesn't exist, create it
        if not self._fb_initialized:
            self._fb_initialized = True

            # seg textures
            self._main_cb, self._main_db = glGenRenderbuffers(2)
            
            glBindRenderbuffer(GL_RENDERBUFFER, self._main_cb)
            glRenderbufferStorage(
                GL_RENDERBUFFER, GL_RGBA,
                self.viewport_width, self.viewport_height
            )
            
            glBindRenderbuffer(GL_RENDERBUFFER, self._main_db)
            glRenderbufferStorage(
                GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                self.viewport_width, self.viewport_height
            )

            self._main_fb = glGenFramebuffers(1)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
            glFramebufferRenderbuffer(
                GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                GL_RENDERBUFFER, self._main_cb
            )
            glFramebufferRenderbuffer(
                GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                GL_RENDERBUFFER, self._main_db
            )

            # output depth textures for peeling 
            self.g_peelTexId = listify(glGenTextures(self.num_peel_passes))
            self.g_peelFboIds = listify(glGenFramebuffers(self.num_peel_passes))

            for i in range(self.num_peel_passes):
                glBindTexture(GL_TEXTURE_RECTANGLE, self.g_peelTexId[i])
                glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA32F, self.viewport_width, self.viewport_height, 0, GL_RGBA, GL_FLOAT, None)

            for i in range(self.num_peel_passes):
                glBindFramebuffer(GL_FRAMEBUFFER, self.g_peelFboIds[i])
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT_LIST[0], GL_TEXTURE_RECTANGLE, self.g_peelTexId[i], 0)

            # input depth textures for mesh-mesh subtraction TODO: reuse ones above
            self.g_peelTexSubId = listify(glGenTextures(self.num_peel_passes*2)) 
            self.g_peelFboSubIds = listify(glGenFramebuffers(self.num_peel_passes*2))

            for i in range(self.num_peel_passes*2):
                glBindTexture(GL_TEXTURE_RECTANGLE, self.g_peelTexSubId[i])
                glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RG32F, self.viewport_width, self.viewport_height, 0, GL_RG, GL_FLOAT, None)

            for i in range(self.num_peel_passes*2):
                glBindFramebuffer(GL_FRAMEBUFFER, self.g_peelFboSubIds[i])
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT_LIST[0], GL_TEXTURE_RECTANGLE, self.g_peelTexSubId[i], 0)

            # additive output textures
            self.g_densityTexId = listify(glGenTextures(self.num_mesh_mesh_layers * self.prim_unqiue_materials))
            self.g_densityFboId = listify(glGenFramebuffers(self.num_mesh_mesh_layers * self.prim_unqiue_materials))

            for i in range(self.num_mesh_mesh_layers * self.prim_unqiue_materials):
                glBindTexture(GL_TEXTURE_RECTANGLE, self.g_densityTexId[i])
                glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RG32F, self.viewport_width, self.viewport_height, 0, GL_RG, GL_FLOAT, None)

            for i in range(self.num_mesh_mesh_layers * self.prim_unqiue_materials):
                glBindFramebuffer(GL_FRAMEBUFFER, self.g_densityFboId[i])
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT_LIST[0], GL_TEXTURE_RECTANGLE, self.g_densityTexId[i], 0)

            self._main_fb_dims = (self.viewport_width, self.viewport_height)

            self.subtractive_reg_ims = []
            for tex_idx in range(self.num_peel_passes):
                reg_img = check_cudart_err(
                    cudart.cudaGraphicsGLRegisterImage(
                        int(self.g_peelTexId[tex_idx]),
                        GL_TEXTURE_RECTANGLE,
                        cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly,
                    )
                )
                self.subtractive_reg_ims.append(reg_img)

            self.mesh_sub_reg_ims = []
            for tex_idx in range(self.num_peel_passes*2):
                reg_img = check_cudart_err(
                    cudart.cudaGraphicsGLRegisterImage(
                        int(self.g_peelTexSubId[tex_idx]),
                        GL_TEXTURE_RECTANGLE,
                        cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
                    )
                )
                self.mesh_sub_reg_ims.append(reg_img)
            

            self.additive_reg_ims = []
            for tex_idx in range(self.num_mesh_mesh_layers * self.prim_unqiue_materials):
                reg_img = check_cudart_err(
                    cudart.cudaGraphicsGLRegisterImage(
                        int(self.g_densityTexId[tex_idx]),
                        GL_TEXTURE_RECTANGLE,
                        cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly,
                    )
                )
                self.additive_reg_ims.append(reg_img)
                

    def _delete_main_framebuffer(self):

        if self._main_fb is not None:
            glDeleteFramebuffers(1, [self._main_fb])
        if self._main_cb is not None:
            glDeleteRenderbuffers(1, [self._main_cb])
        if self._main_db is not None:
            glDeleteRenderbuffers(1, [self._main_db])

        self._main_fb = None
        self._main_cb = None
        self._main_db = None

        if self.additive_reg_ims is not None:
            for reg_img in self.additive_reg_ims:
                check_cudart_err(cudart.cudaGraphicsUnregisterResource(reg_img))

            self.additive_reg_ims = None

        if self.subtractive_reg_ims is not None:
            for reg_img in self.subtractive_reg_ims:
                check_cudart_err(cudart.cudaGraphicsUnregisterResource(reg_img))

            self.subtractive_reg_ims = None

        if self.mesh_sub_reg_ims is not None:
            for reg_img in self.mesh_sub_reg_ims:
                check_cudart_err(cudart.cudaGraphicsUnregisterResource(reg_img))

            self.mesh_sub_reg_ims = None
        

        if self.g_peelTexId is not None:
            glDeleteTextures(self.num_peel_passes, self.g_peelTexId)
            self.g_peelTexId = None
        if self.g_peelFboIds is not None:
            glDeleteFramebuffers(self.num_peel_passes, self.g_peelFboIds)
            self.g_peelFboIds = None
        if self.g_densityTexId is not None:
            glDeleteTextures(self.num_mesh_mesh_layers * self.prim_unqiue_materials, self.g_densityTexId)
            self.g_densityTexId = None
        if self.g_densityFboId is not None:
            glDeleteFramebuffers(self.num_mesh_mesh_layers * self.prim_unqiue_materials, self.g_densityFboId)
            self.g_densityFboId = None
        if self.g_peelTexSubId is not None:
            glDeleteTextures(self.num_peel_passes*2, self.g_peelTexSubId)
            self.g_peelTexSubId = None
        if self.g_peelFboSubIds is not None:
            glDeleteFramebuffers(self.num_peel_passes*2, self.g_peelFboSubIds)
            self.g_peelFboSubIds = None

        self._fb_initialized = False
        self._main_fb_dims = (None, None)


    def _read_main_framebuffer(self, scene, flags):
        width, height = self._main_fb_dims[0], self._main_fb_dims[1]

        # Bind framebuffer and blit buffers
        # glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
        # glBlitFramebuffer(
        #     0, 0, width, height, 0, 0, width, height,
        #     GL_COLOR_BUFFER_BIT, GL_LINEAR
        # )
        # glBlitFramebuffer(
        #     0, 0, width, height, 0, 0, width, height,
        #     GL_DEPTH_BUFFER_BIT, GL_NEAREST
        # )
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb)

        # Read depth
        # depth_buf = glReadPixels(
        #     0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT
        # )
        # depth_im = np.frombuffer(depth_buf, dtype=np.float32)
        # depth_im = depth_im.reshape((height, width))
        # depth_im = np.flip(depth_im, axis=0)
        # inf_inds = (depth_im == 1.0)
        # depth_im = 2.0 * depth_im - 1.0
        # z_near = scene.main_camera_node.camera.znear
        # z_far = scene.main_camera_node.camera.zfar
        # noninf = np.logical_not(inf_inds)
        # if z_far is None:
        #     depth_im[noninf] = 2 * z_near / (1.0 - depth_im[noninf])
        # else:
        #     depth_im[noninf] = ((2.0 * z_near * z_far) /
        #                         (z_far + z_near - depth_im[noninf] *
        #                         (z_far - z_near)))
        # depth_im[inf_inds] = 0.0

        # # Resize for macos if needed
        # if sys.platform == 'darwin':
        #     depth_im = self._resize_image(depth_im)

        # if flags & RenderFlags.DEPTH_ONLY:
        #     return depth_im

        # Read color
        if flags & RenderFlags.RGBA:
            color_buf = glReadPixels(
                0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE
            )
            color_im = np.frombuffer(color_buf, dtype=np.uint8)
            color_im = color_im.reshape((height, width, 4))
        else:
            color_buf = glReadPixels(
                0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE
            )
            color_im = np.frombuffer(color_buf, dtype=np.uint8)
            color_im = color_im.reshape((height, width, 3))
        color_im = np.flip(color_im, axis=0)

        # Resize for macos if needed
        if sys.platform == 'darwin':
            color_im = self._resize_image(color_im, True)

        return color_im