import io
from pathlib import Path

import moderngl
import moderngl_window as mglw
import numpy as np
import pywavefront
from moderngl_window.opengl.vao import VAO
from pyrr import Matrix44, Vector3

import shadevolution.fresnel as fresnel
import shadevolution.plot as plot


class App(mglw.WindowConfig):
    resource_dir = (Path(__file__).parent.parent / 'resources').resolve()
    gl_version = (4, 1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reference_program = fresnel.create_program(self.ctx)
        self._init_model()
        self.duration_renderer = plot.FigureRenderer(self.ctx, xlabel='Time', ylabel='Frame Duration [ns]',
                                                figsize=(600, 200))
        self.duration_line = plot.Line()
        self.duration_renderer.add(self.duration_line)

        self.error_renderer = plot.FigureRenderer(self.ctx, xlabel='Time', ylabel='Error [%]',
                                            figsize=(600, 200))
        self.error_line = plot.Line()
        self.error_renderer.add(self.error_line)

        self.light_pos = Vector3([0.0, 0.0, 2.0], dtype='f4')
        self.light_color = Vector3([1.0, 1.0, 1.0], dtype='f4')
        self.object_color = Vector3([1.0, 0.5, 0.31], dtype='f4')


    def render(self, time, frametime):
        """
         Run the render loop of the application.
         :param self: The application to render.
         :param time: The current time.
         :param frametime: Delta time from last frame in seconds
         """

        self.ctx.clear()
        self.ctx.blend_func = self.ctx.DEFAULT_BLENDING
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)

        translation = Matrix44.from_translation((0, 0, 0), dtype='f4')
        rotation = Matrix44.from_eulers((0, 0, 0.5 * time), dtype='f4')
        model = translation * rotation

        view = Matrix44.look_at(
            (0, 0, 2),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            dtype='f4',
        )

        projection = Matrix44.perspective_projection(fovy=80, aspect=self.wnd.aspect_ratio, near=1.0, far=100.0,
                                                     dtype='f4')

        projection_left = Matrix44.from_translation((-0.5, 0.25, 0), dtype='f4') * projection
        projection_right = Matrix44.from_translation((0.5, 0.25, 0), dtype='f4') * projection

        query_left = self._render_variant(self.reference_program, model, view, projection_left)
        query_right = self._render_variant(self.reference_program, model, view, projection_right)

        projection_bottom_left = Matrix44.from_translation((-0.5, -0.6, 0), dtype='f4') * \
                                 Matrix44.from_scale((1.2, 0.5, 1), dtype='f4') * projection

        projection_bottom_right = Matrix44.from_translation((0.5, -0.6, 0), dtype='f4') * \
                                 Matrix44.from_scale((1.2, 0.5, 1), dtype='f4') * projection

        self.duration_line.append(time, query_right.elapsed)
        self.duration_renderer.render(projection_bottom_left * view)
        self.error_renderer.render(projection_bottom_right * view)

    def _render_variant(self, program, model, view, projection):
        """
        Render a model with the given program to screen.
        :param program: The shader program to use.
        :param model: The model matrix.
        :param view: The view matrix.
        :param projection: The projection matrix.
        """
        mvp = projection * view * model

        program['normalModel'].write(model.T.inverse)
        program['model'].write(model)
        program['mvp'].write(mvp)
        program['lightPos'].write(self.light_pos)
        program['lightColor'].write(self.light_color)
        program['objectColor'].write(self.object_color)

        query = self.ctx.query(samples=True, time=True)
        with query:
            self.vao.render(program)
        return query

    def _init_model(self):
        """
        Initialize the model we want to render.
        """
        scene = pywavefront.Wavefront('resources/scenes/crate.obj')

        vaos = []
        for _, mat in scene.materials.items():
            vbo = np.array(mat.vertices, dtype='f4')
            vao = VAO(mat.name, mode=moderngl.TRIANGLES)
            vao.buffer(vbo, '2f 3f 3f', ['aTex', 'aNormal', 'aPos'])
            vaos.append(vao)
        self.vao = vaos[0]


if __name__ == '__main__':
    mglw.run_window_config(App)
