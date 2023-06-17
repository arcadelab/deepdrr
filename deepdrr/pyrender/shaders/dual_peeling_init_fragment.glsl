//--------------------------------------------------------------------------------------
// Order Independent Transparency with Dual Depth Peeling
//
// Author: Louis Bavoil
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

uniform vec3 cam_pos;

in vec3 frag_position;

void main(void)
{
	// gl_FragColor.xy = vec2(-gl_FragCoord.z, gl_FragCoord.z);
    float dist = length(frag_position-cam_pos);
    gl_FragColor.xy = vec2(-dist, dist);
}
