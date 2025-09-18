#version 330 core

uniform vec3 cam_pos;
uniform float MaxDepth;
uniform float density;

in vec3 frag_position;

void main(void)
{
    float mult=!gl_FrontFacing?-1:1;
    gl_FragColor.rgba=vec4(length(frag_position-cam_pos)*mult*density,mult,0,0);
}
