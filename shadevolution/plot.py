import collections

import matplotlib.backends.backend_agg as backend
import matplotlib.pyplot as plt
import moderngl
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from moderngl_window.opengl.vao import VAO

sns.set()
sns.set_style("white")
plt.style.use('dark_background')


class FigureRenderer:
    """
    A class for managing a Matplotlib figure.
    """

    def __init__(self, ctx, title='', xlabel='', ylabel='', figsize=(500, 500)):
        """
        Construct a figure.

        :param ctx: The OpenGL context to which the figure is bound.
        """
        self.ctx = ctx
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.program = self.ctx.program(
            vertex_shader='''
               #version 330

               uniform mat4 mvp;
               in vec3 aPos;
               in vec2 aTex;
               out vec3 pos;
               out vec2 tex;

               void main() {
                   gl_Position = mvp * vec4(aPos, 1.0);
                   pos = aPos;
                   tex = aTex;
               }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D Texture;
                in vec2 tex;

                out vec4 FragColor;

                void main() {
                    FragColor = texture(Texture, tex);
                }
            ''',
        )

        self.vao = VAO('plot', mode=moderngl.TRIANGLES)
        self.vao.buffer(np.array([
            0.5, 0.5, 1.0,
            -0.5, 0.5, 1.0,
            -0.5, -0.5, 1.0,
            0.5, -0.5, 1.0,
        ], dtype='f4'), '3f', ['aPos'])
        self.vao.buffer(np.array([
            1.0, 0.0,
            0.0, 0.0,
            0.0, 1.0,
            1.0, 1.0
        ], dtype='f4'), '2f', ['aTex'])
        self.vao.index_buffer(np.array([
            2, 1, 0,
            3, 2, 0
        ]))

        self.fig = Figure(figsize=(figsize[0] / 72, figsize[1] / 72))
        self.fig.patch.set_alpha(0.0)

        self.canvas = backend.FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.lines = []

        self._update()
        self.stale = True
        self.texture = None

    def _update(self):
        self.ax.clear()
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.patch.set_alpha(0.0)
        for line in self.lines:
            self.ax.plot(line.x, line.y, 'r-')
        self.ax.autoscale_view(True, True, True)
        self.fig.tight_layout()
        self.stale = True

    def add(self, line):
        self.lines.append(line)
        line.fig = self

    def __del__(self):
        """
        Release the resources of the figure.
        """
        self.program.release()
        self.vao.release()

    def render(self, mvp):
        """
        Render the plot.
        :param mvp: The model view projection matrix to utilize.
        """
        if self.stale:
            self.canvas.draw_idle()
            s, size = self.canvas.print_to_buffer()
            if self.texture:
                self.texture.release()
                self.texture = None
            self.texture = self.ctx.texture(size, 4, s)
            self.stale = False

        self.texture.use()
        self.program['mvp'].write(mvp)
        self.vao.render(self.program)


class Line:
    """
    Represents a 2 dimensional line.
    """

    def __init__(self, fmt='r-', maxlen=100):
        self.x = collections.deque(maxlen=maxlen)
        self.y = collections.deque(maxlen=maxlen)
        self.fig = None

    def append(self, x, y):
        """
        Append an observation to the line.
        :param x: The x value of the observation.
        :param y: The y value of the observation.
        :return:
        """
        self.x.append(x)
        self.y.append(y)

        if self.fig:
            self.fig._update()
