from pathlib import Path

import moderngl
import moderngl_window as mglw
from moderngl_window.scene import KeyboardCamera
from pyrr import Matrix44

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 projection;

void main()
{
    gl_Position = projection * vec4(aPos, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
}
"""


class Game(mglw.WindowConfig):
    resource_dir = (Path(__file__).parent.parent / 'resources').resolve()
    gl_version = (3, 2)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )
        self.camera = mglw.scene.Camera(
            fov=75.0,
            aspect_ratio=self.wnd.aspect_ratio,
            near=0.1,
            far=1000.0,
        )
        self.scene = self.load_scene('scenes/crate.obj')

    def render(self, time, frametime):
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        translation = Matrix44.from_translation((0, 0, -1.5))
        rotation = Matrix44.from_eulers((0, 0, 0))
        model_matrix = translation * rotation
        camera_matrix = self.camera.matrix * model_matrix

        self.scene.draw(
            projection_matrix=self.camera.projection.matrix,
            camera_matrix=camera_matrix,
            time=time,
        )

    def resize(self, width: int, height: int):
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)


if __name__ == '__main__':
    mglw.run_window_config(Game)
