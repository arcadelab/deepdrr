"""PBR renderer for Python.

Author: Matthew Matl
"""
import sys
import numpy as np
import PIL

from .constants import (RenderFlags, TextAlign, GLTF, BufFlags, TexFlags,
                        ProgramFlags, DEFAULT_Z_FAR, DEFAULT_Z_NEAR,
                        SHADOW_TEX_SZ, MAX_N_LIGHTS, DRRMode)
from .shader_program import ShaderProgramCache
from .material import MetallicRoughnessMaterial, SpecularGlossinessMaterial
from .light import PointLight, SpotLight, DirectionalLight
from .font import FontCache
from .utils import format_color_vector

from OpenGL.GL import *


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

    def __init__(self, viewport_width, viewport_height, point_size=1.0, max_dual_peel_layers=4):
        self.dpscale = 1
        # Scaling needed on retina displays
        if sys.platform == 'darwin':
            self.dpscale = 2

        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.point_size = point_size
        self.max_dual_peel_layers = max_dual_peel_layers

        # Optional framebuffer for offscreen renders
        self._main_fb = None
        self._main_cb = None
        self._main_db = None
        self._main_fb_ms = None
        self._main_cb_ms = None
        self._main_db_ms = None
        self._main_fb_dims = (None, None)
        self._shadow_fb = None
        self._latest_znear = DEFAULT_Z_NEAR
        self._latest_zfar = DEFAULT_Z_FAR
        self.g_dualDepthTexId = None
        self.g_dualPeelingSingleFboId = None
        self.g_dualPeelingFboIds = None
        self.g_densityTexId = None
        self.g_densityFboId = None

        # Shader Program Cache
        self._program_cache = ShaderProgramCache()
        self._font_cache = FontCache()
        self._meshes = set()
        self._mesh_textures = set()
        self._shadow_textures = set()
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

    def render(self, scene, flags, seg_node_map=None, drr_mode=DRRMode.NONE, zfar=0):
        self._update_context(scene, flags)

        if drr_mode != DRRMode.DENSITY:
            for i in range(self.max_dual_peel_layers):
                retval = self._forward_pass(scene, flags, seg_node_map=seg_node_map, drr_mode=drr_mode, zfar=zfar, peelnum=i, front=True)
        else:
            retval = self._forward_pass(scene, flags, seg_node_map=seg_node_map, drr_mode=drr_mode, zfar=zfar, peelnum=0)

        self._latest_znear = scene.main_camera_node.camera.znear
        self._latest_zfar = scene.main_camera_node.camera.zfar

        return retval


    def delete(self):
        """Free all allocated OpenGL resources.
        """
        # Free shaders
        self._program_cache.clear()

        # Free fonts
        self._font_cache.clear()

        # Free meshes
        for mesh in self._meshes:
            for p in mesh.primitives:
                p.delete()

        # Free textures
        for mesh_texture in self._mesh_textures:
            mesh_texture.delete()

        for shadow_texture in self._shadow_textures:
            shadow_texture.delete()

        self._meshes = set()
        self._mesh_textures = set()
        self._shadow_textures = set()
        self._texture_alloc_idx = 0

        self._delete_main_framebuffer()
        self._delete_shadow_framebuffer()

    def __del__(self):
        try:
            self.delete()
        except Exception:
            pass

    ###########################################################################
    # Rendering passes
    ###########################################################################

    def _forward_pass(self, scene, flags, seg_node_map=None, drr_mode=DRRMode.NONE, zfar=0, peelnum=0, front=True):
        # Set up viewport for render
        self._configure_forward_pass_viewport(flags, drr_mode=drr_mode, peelnum=peelnum, front=front)

        # Clear it
        # if bool(flags & RenderFlags.SEG):
        #     glClearColor(0.0, 0.0, 0.0, 1.0)
        #     if seg_node_map is None:
        #         seg_node_map = {}
        # else:
        #     glClearColor(*scene.bg_color)
        if drr_mode != DRRMode.DENSITY:
            glClearColor(-zfar, -zfar, -zfar, -zfar)
        else:
            glClearColor(0, 0, 0, 0)


        # glClear(GL_COLOR_BUFFER_BIT) # TODO
        # glClear(GL_DEPTH_BUFFER_BIT) # TODO
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # if not bool(flags & RenderFlags.SEG):
        #     glEnable(GL_MULTISAMPLE)
        # else:
        #     glDisable(GL_MULTISAMPLE)
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
            if bool(flags & RenderFlags.SEG):
                if node not in seg_node_map:
                    continue
                color = seg_node_map[node]
                if not isinstance(color, (list, tuple, np.ndarray)):
                    color = np.repeat(color, 3)
                else:
                    color = np.asanyarray(color)
                color = color / 255.0

            for primitive in mesh.primitives:

                # First, get and bind the appropriate program
                program = self._get_primitive_program(
                    primitive, flags, ProgramFlags.USE_MATERIAL, drr_mode=drr_mode, peelnum=peelnum
                )
                program._bind()

                # Set the camera uniforms
                program.set_uniform('V', V)
                program.set_uniform('P', P)
                program.set_uniform(
                    'cam_pos', scene.get_pose(scene.main_camera_node)[:3,3]
                )
                if bool(flags & RenderFlags.SEG):
                    program.set_uniform('color', color)

                # # Next, bind the lighting
                # if not (flags & RenderFlags.DEPTH_ONLY or flags & RenderFlags.FLAT or
                #         flags & RenderFlags.SEG):
                #     self._bind_lighting(scene, program, node, flags)

                # Finally, bind and draw the primitive
                self._bind_and_draw_primitive(
                    primitive=primitive,
                    pose=scene.get_pose(node),
                    program=program,
                    flags=flags,
                    drr_mode=drr_mode,
                    zfar=zfar,
                    peelnum=peelnum,
                    front=front
                )
                self._reset_active_textures()

        # Unbind the shader and flush the output
        if program is not None:
            program._unbind()
        # glFlush() # TODO: I don't think this is needed for offscreen

        # if peelnum == self.max_dual_peel_layers-1 or drr_mode == DRRMode.DENSITY:
        #     return self._read_main_framebuffer(scene, flags, drr_mode=drr_mode, front=front)
        return []

        # # If doing offscreen render, copy result from framebuffer and return
        # if flags & RenderFlags.OFFSCREEN:
        #     return self._read_main_framebuffer(scene, flags)
        # else:
        #     raise ValueError('TODO')
        #     glFlush() # Maybe?
        #     return


    def _bind_and_draw_primitive(self, primitive, pose, program, flags, drr_mode=DRRMode.NONE, zfar=3, peelnum=0, front=True):
        # Set model pose matrix
        program.set_uniform('M', pose)

        # Bind mesh buffers
        primitive._bind()

        # Bind mesh material
        material = primitive.material

        if peelnum > 0:
            glActiveTexture(GL_TEXTURE0 + 0)
            glBindTexture(GL_TEXTURE_RECTANGLE, self.g_dualDepthTexId[peelnum-1])
            program.set_uniform('DepthBlenderTex', 0)
            glActiveTexture(GL_TEXTURE0)

        program.set_uniform('MaxDepth', float(zfar))
        

        if drr_mode == DRRMode.BACKDIST:
            glEnable(GL_BLEND)
            glBlendEquation(GL_MAX)
            glBlendFunc(GL_ONE, GL_ONE)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glDisable(GL_CULL_FACE)
        elif drr_mode == DRRMode.DENSITY:
            program.set_uniform('density', float(primitive.density))
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
        cam_loc = scene.get_pose(scene.main_camera_node)[:3,3]
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
            key=lambda n: -np.linalg.norm(scene.get_pose(n)[:3,3] - cam_loc)
        )
        solid_nodes.sort(
            key=lambda n: -np.linalg.norm(scene.get_pose(n)[:3,3] - cam_loc)
        )

        return solid_nodes + trans_nodes

    def _sorted_nodes_by_distance(self, scene, nodes, compare_node):
        nodes = list(nodes)
        compare_posn = scene.get_pose(compare_node)[:3,3]
        nodes.sort(key=lambda n: np.linalg.norm(
            scene.get_pose(n)[:3,3] - compare_posn)
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

    def _get_text_program(self):
        program = self._program_cache.get_program(
            vertex_shader='text.vert',
            fragment_shader='text.frag'
        )

        if not program._in_context():
            program._add_to_context()

        return program

    def _compute_max_n_lights(self, flags):
        max_n_lights = [MAX_N_LIGHTS, MAX_N_LIGHTS, MAX_N_LIGHTS]
        n_tex_units = glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS)

        # Reserved texture units: 6
        #   Normal Map
        #   Occlusion Map
        #   Emissive Map
        #   Base Color or Diffuse Map
        #   MR or SG Map
        #   Environment cubemap

        n_reserved_textures = 6
        n_available_textures = n_tex_units - n_reserved_textures

        # Distribute textures evenly among lights with shadows, with
        # a preference for directional lights
        n_shadow_types = 0
        if flags & RenderFlags.SHADOWS_DIRECTIONAL:
            n_shadow_types += 1
        if flags & RenderFlags.SHADOWS_SPOT:
            n_shadow_types += 1
        if flags & RenderFlags.SHADOWS_POINT:
            n_shadow_types += 1

        if n_shadow_types > 0:
            tex_per_light = n_available_textures // n_shadow_types

            if flags & RenderFlags.SHADOWS_DIRECTIONAL:
                max_n_lights[0] = (
                    tex_per_light +
                    (n_available_textures - tex_per_light * n_shadow_types)
                )
            if flags & RenderFlags.SHADOWS_SPOT:
                max_n_lights[1] = tex_per_light
            if flags & RenderFlags.SHADOWS_POINT:
                max_n_lights[2] = tex_per_light

        return max_n_lights

    def _get_primitive_program(self, primitive, flags, program_flags, drr_mode=DRRMode.NONE, peelnum=0):
        vertex_shader = None
        fragment_shader = None
        geometry_shader = None
        defines = {}

        if drr_mode != DRRMode.DENSITY:
            if peelnum == 0:
                vertex_shader = 'dual_peeling_init_vertex.glsl'
                fragment_shader = 'dual_peeling_init_fragment.glsl'
            else:
                vertex_shader = 'dual_peeling_peel_vertex.glsl'
                fragment_shader = 'dual_peeling_peel_fragment.glsl'
        else:
            vertex_shader = 'density.vert'
            fragment_shader = 'density.frag'

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

        # Set up shadow mapping defines
        if flags & RenderFlags.SHADOWS_DIRECTIONAL:
            defines['DIRECTIONAL_LIGHT_SHADOWS'] = 1
        if flags & RenderFlags.SHADOWS_SPOT:
            defines['SPOT_LIGHT_SHADOWS'] = 1
        if flags & RenderFlags.SHADOWS_POINT:
            defines['POINT_LIGHT_SHADOWS'] = 1
        max_n_lights = self._compute_max_n_lights(flags)
        defines['MAX_DIRECTIONAL_LIGHTS'] = max_n_lights[0]
        defines['MAX_SPOT_LIGHTS'] = max_n_lights[1]
        defines['MAX_POINT_LIGHTS'] = max_n_lights[2]

        # Set up vertex normal defines
        if program_flags & ProgramFlags.VERTEX_NORMALS:
            defines['VERTEX_NORMALS'] = 1
        if program_flags & ProgramFlags.FACE_NORMALS:
            defines['FACE_NORMALS'] = 1

        # Set up material texture defines
        if bool(program_flags & ProgramFlags.USE_MATERIAL):
            tf = primitive.material.tex_flags
            if tf & TexFlags.NORMAL:
                defines['HAS_NORMAL_TEX'] = 1
            if tf & TexFlags.OCCLUSION:
                defines['HAS_OCCLUSION_TEX'] = 1
            if tf & TexFlags.EMISSIVE:
                defines['HAS_EMISSIVE_TEX'] = 1
            if tf & TexFlags.BASE_COLOR:
                defines['HAS_BASE_COLOR_TEX'] = 1
            if tf & TexFlags.METALLIC_ROUGHNESS:
                defines['HAS_METALLIC_ROUGHNESS_TEX'] = 1
            if tf & TexFlags.DIFFUSE:
                defines['HAS_DIFFUSE_TEX'] = 1
            if tf & TexFlags.SPECULAR_GLOSSINESS:
                defines['HAS_SPECULAR_GLOSSINESS_TEX'] = 1
            if isinstance(primitive.material, MetallicRoughnessMaterial):
                defines['USE_METALLIC_MATERIAL'] = 1
            elif isinstance(primitive.material, SpecularGlossinessMaterial):
                defines['USE_GLOSSY_MATERIAL'] = 1

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

    def _configure_forward_pass_viewport(self, flags, drr_mode=DRRMode.NONE, peelnum=0, front=True):
        self._configure_main_framebuffer()

        if drr_mode == DRRMode.DENSITY:
            glBindFramebuffer(GL_FRAMEBUFFER, self.g_densityFboId)
        else:
            glBindFramebuffer(GL_FRAMEBUFFER, self.g_dualPeelingFboIds[peelnum])

        glDrawBuffer(GL_COLOR_ATTACHMENT_LIST[0])

        glViewport(0, 0, self.viewport_width, self.viewport_height)
        glDisable(GL_DEPTH_TEST)
        glDepthFunc(GL_ALWAYS)
        glDepthRange(0.0, 1.0)


    ###########################################################################
    # Framebuffer Management
    ###########################################################################

    def _configure_shadow_framebuffer(self):
        if self._shadow_fb is None:
            self._shadow_fb = glGenFramebuffers(1)

    def _delete_shadow_framebuffer(self):
        if self._shadow_fb is not None:
            glDeleteFramebuffers(1, [self._shadow_fb])

    def _configure_main_framebuffer(self):
        # If mismatch with prior framebuffer, delete it
        if (self._main_fb is not None and
                self.viewport_width != self._main_fb_dims[0] or
                self.viewport_height != self._main_fb_dims[1]):
            self._delete_main_framebuffer()

        # If framebuffer doesn't exist, create it
        if self._main_fb is None:
            self.g_dualDepthTexId = glGenTextures(self.max_dual_peel_layers)
            # self.g_dualPeelingSingleFboId = glGenFramebuffers(1)
            self.g_dualPeelingFboIds = glGenFramebuffers(self.max_dual_peel_layers)



            for i in range(self.max_dual_peel_layers):
                glBindTexture(GL_TEXTURE_RECTANGLE, self.g_dualDepthTexId[i])
                glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA32F, self.viewport_width, self.viewport_height, 0, GL_RGBA, GL_FLOAT, None)
                # print(f"self.g_dualDepthTexId[{i}] = {self.g_dualDepthTexId[i]} {self.viewport_width} {self.viewport_height}")


            # glBindFramebuffer(GL_FRAMEBUFFER, self.g_dualPeelingSingleFboId)
            for i in range(self.max_dual_peel_layers):
                glBindFramebuffer(GL_FRAMEBUFFER, self.g_dualPeelingFboIds[i])
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT_LIST[0], GL_TEXTURE_RECTANGLE, self.g_dualDepthTexId[i], 0)
                # glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT_LIST[i], GL_TEXTURE_RECTANGLE, self.g_dualDepthTexId[i], 0)


            self.g_densityTexId = glGenTextures(1)
            self.g_densityFboId = glGenFramebuffers(1)
            
            glBindTexture(GL_TEXTURE_RECTANGLE, self.g_densityTexId)
            glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RG32F, self.viewport_width, self.viewport_height, 0, GL_RG, GL_FLOAT, None)

            glBindFramebuffer(GL_FRAMEBUFFER, self.g_densityFboId)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT_LIST[0], GL_TEXTURE_RECTANGLE, self.g_densityTexId, 0)            

            # Generate standard buffer
            self._main_cb, self._main_db = glGenRenderbuffers(2)

            glBindRenderbuffer(GL_RENDERBUFFER, self._main_cb)
            glRenderbufferStorage(
                GL_RENDERBUFFER, GL_RGBA32F,
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

            # Generate multisample buffer
            self._main_cb_ms, self._main_db_ms = glGenRenderbuffers(2)
            glBindRenderbuffer(GL_RENDERBUFFER, self._main_cb_ms)
            glRenderbufferStorageMultisample(
                GL_RENDERBUFFER, 4, GL_RGBA32F,
                self.viewport_width, self.viewport_height
            )
            glBindRenderbuffer(GL_RENDERBUFFER, self._main_db_ms)
            glRenderbufferStorageMultisample(
                GL_RENDERBUFFER, 4, GL_DEPTH_COMPONENT24,
                self.viewport_width, self.viewport_height
            )
            self._main_fb_ms = glGenFramebuffers(1)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb_ms)
            glFramebufferRenderbuffer(
                GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                GL_RENDERBUFFER, self._main_cb_ms
            )
            glFramebufferRenderbuffer(
                GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                GL_RENDERBUFFER, self._main_db_ms
            )

            self._main_fb_dims = (self.viewport_width, self.viewport_height)

    def _delete_main_framebuffer(self):
        if self._main_fb is not None:
            glDeleteFramebuffers(2, [self._main_fb, self._main_fb_ms])
        if self._main_cb is not None:
            glDeleteRenderbuffers(2, [self._main_cb, self._main_cb_ms])
        if self._main_db is not None:
            glDeleteRenderbuffers(2, [self._main_db, self._main_db_ms])
        if self.g_dualDepthTexId is not None:
            glDeleteTextures(2, self.g_dualDepthTexId)
            self.g_dualDepthTexId = None # TODO: needed?
        if self.g_dualPeelingSingleFboId is not None:
            glDeleteFramebuffers(1, self.g_dualPeelingSingleFboId)
            self.g_dualPeelingSingleFboId = None #TODO: needed?
        if self.g_dualPeelingFboIds is not None:
            glDeleteFramebuffers(self.max_dual_peel_layers, self.g_dualPeelingFboIds)
            self.g_dualPeelingFboIds = None
        if self.g_densityTexId is not None:
            glDeleteTextures(1, [self.g_densityTexId])
            self.g_densityTexId = None
        if self.g_densityFboId is not None:
            glDeleteFramebuffers(1, [self.g_densityFboId])
            self.g_densityFboId = None

        self._main_fb = None
        self._main_cb = None
        self._main_db = None
        self._main_fb_ms = None
        self._main_cb_ms = None
        self._main_db_ms = None
        self._main_fb_dims = (None, None)

    # def _read_main_framebuffer(self, scene, flags, drr_mode=DRRMode.NONE, front=True):
    #     width, height = self._main_fb_dims[0], self._main_fb_dims[1]

    #     # Bind framebuffer and blit buffers
    #     glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb_ms)
    #     glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
    #     glBlitFramebuffer(
    #         0, 0, width, height, 0, 0, width, height,
    #         GL_COLOR_BUFFER_BIT, GL_LINEAR
    #     )
    #     glBlitFramebuffer(
    #         0, 0, width, height, 0, 0, width, height,
    #         GL_DEPTH_BUFFER_BIT, GL_NEAREST
    #     )
    #     glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb)

    #     ims = []

    #     numbufs = self.max_dual_peel_layers
    #     if drr_mode == DRRMode.DENSITY:
    #         glBindFramebuffer(GL_READ_FRAMEBUFFER, self.g_densityFboId)
    #         glReadBuffer(GL_COLOR_ATTACHMENT_LIST[0])

    #         color_buf = glReadPixels(
    #             0, 0, width, height, GL_RGB, GL_FLOAT
    #         )
    #         color_im = np.frombuffer(color_buf, dtype=np.float32)
    #         color_im = color_im.reshape((height, width, 3))
    #         color_im = np.flip(color_im, axis=0)
    #         ims.append(color_im)

    #     else:
    #         for i in range(numbufs):
    #             bufferidx = i + (0 if front else self.max_dual_peel_layers)
    #             glBindFramebuffer(GL_READ_FRAMEBUFFER, self.g_dualPeelingFboIds[bufferidx])
    #             glReadBuffer(GL_COLOR_ATTACHMENT_LIST[0])
    #             # glReadBuffer(GL_COLOR_ATTACHMENT_LIST[bufferidx])
    #             print(f"Reading buffer {bufferidx}")

    #             color_buf = glReadPixels(
    #                 0, 0, width, height, GL_RGBA, GL_FLOAT
    #             )
    #             color_im = np.frombuffer(color_buf, dtype=np.float32)
    #             color_im = color_im.reshape((height, width, 4))
    #             color_im = np.flip(color_im, axis=0)
    #             ims.append(color_im)

    #     return ims

