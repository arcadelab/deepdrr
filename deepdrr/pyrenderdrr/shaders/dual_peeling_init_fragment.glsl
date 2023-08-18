//--------------------------------------------------------------------------------------
// Order Independent Transparency with Dual Depth Peeling
//
// Author: Louis Bavoil
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

uniform vec3 cam_pos;
uniform float MaxDepth;

in vec3 frag_position;

void main(void)
{
	// gl_FragColor.xy = vec2(-gl_FragCoord.z, gl_FragCoord.z);
    // float mult = gl_FrontFacing ? -1 : 1;

    float dist = length(frag_position-cam_pos);

    if (!gl_FrontFacing) {
        gl_FragColor.rgba = vec4(-dist, dist, -MaxDepth, -MaxDepth);
    } else {
        gl_FragColor.rgba = vec4(-MaxDepth, -MaxDepth, -dist, dist);
    }
}
