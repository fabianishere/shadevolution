FRESNEL_BASELINE = '''
float Fresnel (float th, float n) {
    float cosi = cos (th);
    float R = 1.0f;
    float n12 = 1.0f / n;
    float sint = n12 * sqrt (1 - (cosi * cosi));

    if (sint < 1.0f) {
        float cost = sqrt (1.0 - (sint * sint));
        float r_ortho = (cosi - n * cost) / (cosi + n * cost);
        float r_par = (cost - n * cosi) / (cost + n * cosi);
        R = (r_ortho * r_ortho + r_par * r_par) / 2;
    }
    return R;
}
'''


def create_program(ctx, fragment=FRESNEL_BASELINE):
    """
    Construct a OpenGL program for the Fresnel shader.

    :param ctx: The OpenGL context to use.
    :param fragment: The code to append to the fragment shader.
    :return: The OpenGL program.
    """
    return ctx.program(
        vertex_shader=create_vertex_shader(),
        fragment_shader=create_fragment_shader(fragment)
    )


def create_vertex_shader():
    """
    Construct a vertex shader for use with the Fresnel function.

    :return: The total vertex shader implementation.
    """
    return '''
        #version 410 core
        in vec2 aTex;
        in vec3 aPos;
        in vec3 aNormal;

        out vec3 FragPos;
        out vec3 Normal;

        uniform mat4 model;
        uniform mat4 normalModel;
        uniform mat4 mvp;

        void main()
        {
            FragPos = vec3(model * vec4(aPos, 1.0));
            Normal = vec3(normalModel * vec4(aNormal, 1.0));

            gl_Position = mvp * vec4(aPos, 1.0);
        }
    '''


def create_fragment_shader(fresnel):
    """
    Construct a fragment shader with the specified Fresnel function implementation.

    :param fresnel: The source code of the Fresnel function.
    :return: The total shader implementation.
    """
    return f'''
        #version 410 core
        in vec3 Normal;
        in vec3 FragPos;

        out vec4 FragColor;

        uniform vec3 lightPos;
        uniform vec3 lightColor;
        uniform vec3 objectColor;

        {fresnel}

        void main()
        {{
            float ambientStrength = 0.1;
            vec3 ambient = ambientStrength * lightColor;

            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            float fresnelStrength = 15;
            vec3 fresnel = fresnelStrength * Fresnel(diff, 1.25) * lightColor;

            vec3 result = (ambient + fresnel) * objectColor;
            FragColor = vec4(result, 1.0);
        }}
    '''
