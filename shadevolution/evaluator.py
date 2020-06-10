import sys

import numpy as np
from pyrr import Matrix44

from shadevolution import models, fresnel, shader


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

        self.fbo = self.ctx.simple_framebuffer(size)

        self.vao = models.load_crate()
        self.fresnel = fresnel.Fresnel(self.ctx, self.vao)

    def __del__(self):
        self.vao.release()
        self.fbo.release()

    def determine_baseline(self):
        """
        Determine the baseline value of the original shader.

        :return: The baseline as a Numpy array representing the framebuffer.
        """
        model, view, projection = self._prepare_mvp(self.size[0] / self.size[1])

        # Render in our framebuffer object
        self.fbo.use()

        program = fresnel.create_program(self.ctx)
        self.fresnel.render(program, model, view, projection)

        # Make sure we render again to the window
        self.wnd.use()

        raw = self.fbo.read(components=4, dtype='f4')
        return np.frombuffer(raw, dtype='f4')

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
        model, view, projection = self._prepare_mvp(self.size[0] / self.size[1])

        # Render in our framebuffer object
        self.fbo.use()

        try:
            program = fresnel.create_program(self.ctx, source)

            frame_durations = []

            for _ in range(self.repeat):
                query = self.fresnel.render(program, model, view, projection)
                frame_durations.append(query.elapsed)

                # raw = self.fbo.read(components=4, dtype='f4')
                # fb = np.frombuffer(raw, dtype='f4')

            self.wnd.use()
            self.ctx.clear()
            self.fresnel.render(program, model, view, projection)
            self.wnd.swap_buffers()
        except Exception as e:
            # Make sure we render again to the window
            self.wnd.use()

            # print(e)
            # diff = shader.diff(name, params, genesis, individual)
            # sys.stdout.writelines(diff)
            return False,
        return np.mean(frame_durations),

    @staticmethod
    def _prepare_mvp(aspect):
        """
        Prepare the model view projection matrices.

        :return: A tuple containing the three matrices.
        """
        translation = Matrix44.from_translation((0, 0, 0), dtype='f4')
        rotation = Matrix44.from_eulers((0, 0, 0), dtype='f4')
        model = translation * rotation
        view = Matrix44.look_at(
            (0, 0, 2),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            dtype='f4',
        )
        projection = Matrix44.perspective_projection(
            fovy=80, aspect=aspect, near=1.0, far=100.0,
            dtype='f4'
        )

        return model, view, projection
