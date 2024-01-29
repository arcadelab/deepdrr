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

    float fragDepth = length(frag_position-cam_pos);

    // if (!gl_FrontFacing) {
    //     gl_FragColor.rgba = vec4(-fragDepth, fragDepth, -MaxDepth, -MaxDepth);
    // } else {
    //     gl_FragColor.rgba = vec4(-MaxDepth, -MaxDepth, -fragDepth, fragDepth);
    // }

    if (!gl_FrontFacing) {
        gl_FragData[0].rgba = vec4(-MaxDepth, -MaxDepth, -MaxDepth, fragDepth);
    } else {
        gl_FragData[0].rgba = vec4(-MaxDepth, fragDepth, -MaxDepth, -MaxDepth);
    }
}
