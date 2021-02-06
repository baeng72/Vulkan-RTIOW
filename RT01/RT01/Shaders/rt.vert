#version 450
layout (location=0) in vec3 aPos;
layout(location = 1) in vec2 inUV;
layout (location = 0) out vec2 outUV;

out gl_PerVertex 
{
	vec4 gl_Position;
};

void main() 
{
	outUV = inUV;
	gl_Position = vec4(outUV * 2.0f + -1.0f, 0.0f, 1.0f);
}
