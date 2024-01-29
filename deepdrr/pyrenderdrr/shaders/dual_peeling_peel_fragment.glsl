//--------------------------------------------------------------------------------------
// Order Independent Transparency with Dual Depth Peeling
//
// Author: Louis Bavoil
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#extension ARB_draw_buffers : require

uniform sampler2DRect DepthBlenderTex;
// uniform samplerRECT FrontBlenderTex;

// #define MAX_DEPTH 10.0
uniform float MaxDepth;

// vec4 ShadeFragment();

uniform vec3 cam_pos;

in vec3 frag_position;

// void main(void)
// {
// 	// gl_FragColor.xy = vec2(-gl_FragCoord.z, gl_FragCoord.z);
//     float dist = length(frag_position-cam_pos);
//     gl_FragColor.xy = vec2(-dist, dist);
// }


void main(void)
{
    float fragDepth = length(frag_position-cam_pos);
	vec4 depthBlender = texture2DRect(DepthBlenderTex, gl_FragCoord.xy).xyzw;

    float nearestDepthFront = depthBlender.x;
	float farthestDepthFront = depthBlender.y;
    float nearestDepthAway = depthBlender.z;
	float farthestDepthAway = depthBlender.w;

    float near = -MaxDepth;
    float far = -MaxDepth;
    
    if (!gl_FrontFacing) {
        if (-fragDepth < nearestDepthAway) {
            near = -fragDepth;
        }
        if (fragDepth < farthestDepthAway) {
            far = fragDepth;
        }
        gl_FragData[0].rgba = vec4(-MaxDepth, -MaxDepth, near, far);
        return;
    } else {
        if (-fragDepth < nearestDepthFront) {
            near = -fragDepth;
        }
        if (fragDepth < farthestDepthFront) {
            far = fragDepth;
        }
        gl_FragData[0].rgba = vec4(near, far, -MaxDepth, -MaxDepth);
        return;
    }
}
