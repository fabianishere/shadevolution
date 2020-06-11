import moderngl
import numpy as np
from pyrr import Matrix44

from shadevolution import models, fresnel, shader, plot


class Evaluator:
    """
    An evaluator that runs a genetic algorithm using OpenGL for the fitness evaluation.
    """
    gl_version = (4, 1)

    def __init__(self, window, size=(2048, 2048), repeat=32):
        """
        Construct an evaluator instance.

        :param window: The window to perform the evaluation in.
        :param size: The size of the framebuffer to render in.
        """
        self.wnd = window
        self.ctx = window.ctx
        self.size = size
        self.repeat = repeat

        self.reference_program = fresnel.create_program(self.ctx)
        self.fbo = self.ctx.simple_framebuffer(size, 4)
        self.vao = models.load_crate()
        self.fresnel = fresnel.Fresnel(self.ctx, self.vao)

        self.reporter = plot.VisualizationReporter()

    def __del__(self):
        self.vao.release()
        self.fbo.release()

    def determine_baseline(self):
        """
        Determine the baseline value of the original shader.

        :return: The baseline as a Numpy array representing the framebuffer.
        """
        view = self._prepare_view()
        projection = self._prepare_projection(aspect=16/9, fovy=40)

        program = self.reference_program
        res = []

        try:
            for i in range(self.repeat):
                model = self._prepare_model(0.05 * i)

                # Render in our framebuffer object
                self.fbo.use()
                self.fbo.clear()
                self.fresnel.render(program, model, view, projection)

                raw = self.fbo.read(components=3, dtype='f4')
                fb = np.frombuffer(raw, dtype='f4')
                fb = fb.reshape((len(fb) // 3, 3))
                res.append(fb)

                # Render demonstration
                self.wnd.use()
                self._render_window(program, model, view, projection)
                self.wnd.swap_buffers()
        finally:
            # Make sure we render again to the window
            self.wnd.use()

        return res

    def eval(self, individual, genesis, baseline):
        """
        Evaluate the specified individual.

        :param individual: The individual to evaluate.
        :param genesis: The genesis tree.
        :param baseline: The baseline to compute the error against.
        :return: The score of the individual.
        """
        name = 'Fresnel'
        params = [('th', 'float'), ('n', 'float')]

        source = shader.write(name, params, individual)
        view = self._prepare_view()
        projection = self._prepare_projection(aspect=16/9, fovy=40)

        frame_durations = []
        errors = []

        try:
            program = fresnel.create_program(self.ctx, source)

            for i in range(self.repeat):
                model = self._prepare_model(0.05 * i)

                # Render in our framebuffer object
                self.fbo.use()
                self.fbo.clear()

                query = self.fresnel.render(program, model, view, projection)
                frame_durations.append(query.elapsed)

                # self.duration_line.append(self.count, query.elapsed)
                # self.count += 1

                raw = self.fbo.read(components=3, dtype='f4')
                fb = np.frombuffer(raw, dtype='f4')
                fb = fb.reshape((len(fb) // 3, 3))

                err = np.linalg.norm(fb - baseline[i])
                errors.append(err)

                # Render demonstration
                self.wnd.use()
                self._render_window(program, model, view, projection)
                self.wnd.swap_buffers()

                self.reporter.report(query.elapsed, err)
        except Exception as e:
            print(e)
            # Make sure we can render again to the window
            self.wnd.use()
            return 35000000, 10000

        return np.mean(frame_durations), np.mean(errors)

    def _render_window(self, program, model, view, projection):
        """
        Render a demo of the current shaders at work to screen.
        """
        self.ctx.clear()
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)

        projection_left = Matrix44.from_translation((-0.5, 0, 0), dtype='f4') * projection
        projection_right = Matrix44.from_translation((0.5, 0, 0), dtype='f4') * projection

        self.fresnel.render(self.reference_program, model, view, projection_left)
        self.fresnel.render(program, model, view, projection_right)

    @staticmethod
    def _prepare_model(rot):
        """
        Prepare the model matrix.

        :param rot: The rotation to apply.
        """
        translation = Matrix44.from_translation((0, 0, 0), dtype='f4')
        rotation = Matrix44.from_eulers((0, 0, rot), dtype='f4')
        return translation * rotation

    @staticmethod
    def _prepare_view(x=2, y=2, z=2):
        """
        Prepare the view matrix.
        """
        return Matrix44.look_at(
            (x, y, z),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            dtype='f4',
        )

    @staticmethod
    def _prepare_projection(aspect, fovy=60):
        """
        Prepare the projection matrix.
        """
        return Matrix44.perspective_projection(
            fovy=fovy, aspect=aspect, near=1.0, far=100.0,
            dtype='f4'
        )
