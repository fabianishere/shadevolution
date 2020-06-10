import moderngl
import numpy as np
import pywavefront
from moderngl_window.opengl.vao import VAO


def load_crate():
    """
    Load the crate model we want to render in a VAO object.
    """
    scene = pywavefront.Wavefront('resources/scenes/crate.obj')

    vaos = []
    for _, mat in scene.materials.items():
        vbo = np.array(mat.vertices, dtype='f4')
        vao = VAO(mat.name, mode=moderngl.TRIANGLES)
        vao.buffer(vbo, '2f 3f 3f', ['aTex', 'aNormal', 'aPos'])
        vaos.append(vao)
    return vaos[0]
