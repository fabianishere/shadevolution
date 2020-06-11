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


class OpenGLFigureRenderer:
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
        self.series = {}

        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.patch.set_alpha(0.0)

        self._update()
        self.stale = True
        self.texture = None

    def _update(self):
        for series, line in self.series.items():
            line.set_data(series.x, series.y)
        self.ax.relim()
        self.ax.autoscale(tight=True)
        self.stale = True

    def add(self, line):
        self.series[line] = self.ax.plot(line.x, line.y, line.fmt)[0]
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
            self.canvas.draw()
            s, size = self.canvas.print_to_buffer()
            if self.texture:
                self.texture.release()
                self.texture = None
            self.texture = self.ctx.texture(size, 4, s)
            self.stale = False

        self.texture.use()
        self.program['mvp'].write(mvp)
        self.vao.render(self.program)


class VisualizationReporter:
    """
    A reporter that reports the statistics as a visualization.
    """

    def __init__(self, maxlen=100):
        plt.ion()
        plt.show(block=False)

        self.count = 0
        self.index = collections.deque(maxlen=maxlen)
        self.elapsed_live = collections.deque(maxlen=maxlen)
        self.elapsed_mean = collections.deque(maxlen=maxlen)
        self.elapsed_min = collections.deque(maxlen=maxlen)
        self.elapsed_max = collections.deque(maxlen=maxlen)
        self.error_live = collections.deque(maxlen=maxlen)
        self.error_mean = collections.deque(maxlen=maxlen)
        self.error_min = collections.deque(maxlen=maxlen)
        self.error_max = collections.deque(maxlen=maxlen)

        self.fig, self.axs = plt.subplots(2, sharex=True)
        self._setup_elapsed()
        self._setup_error()
        self.fig.tight_layout()

    def __del__(self):
        plt.ioff()

    def _setup_elapsed(self):
        ax = self.axs[0]
        line = ax.plot([], [], '-', label='Live')[0]
        line_mean = ax.plot([], [], '-', label='Mean')[0]
        line_min = ax.plot([], [], '-', label='Min')[0]
        line_max = ax.plot([], [], '-', label='Max')[0]
        self.elapsed_plot = ax, (line, line_mean, line_min, line_max)
        ax.set_ylabel('Frame Duration [ns]')
        ax.legend()

    def _update_elapsed(self):
        ax, lines = self.elapsed_plot
        lines[0].set_data(self.index, self.elapsed_live)
        lines[1].set_data(self.index, self.elapsed_mean)
        lines[2].set_data(self.index, self.elapsed_min)
        lines[3].set_data(self.index, self.elapsed_max)
        ax.relim()
        ax.autoscale(tight=True)

    def _setup_error(self):
        ax = self.axs[1]
        line = ax.plot([], [], '-', label='Live')[0]
        line_mean = ax.plot([], [], '-', label='Mean')[0]
        line_min = ax.plot([], [], '-', label='Min')[0]
        line_max = ax.plot([], [], '-', label='Max')[0]
        self.error_plot = ax, (line, line_mean, line_min, line_max)
        ax.set_ylabel('L2 Error')

    def _update_error(self):
        ax, lines = self.error_plot
        lines[0].set_data(self.index, self.error_live)
        lines[1].set_data(self.index, self.error_mean)
        lines[2].set_data(self.index, self.error_min)
        lines[3].set_data(self.index, self.error_max)
        ax.relim()
        ax.autoscale(tight=True)

    def report(self, elapsed, error):
        self.index.append(self.count)

        # Elapsed
        self.elapsed_live.append(elapsed)
        if self.elapsed_mean:
            cur = self.elapsed_mean[-1]
            new = (cur * self.count + elapsed) / (self.count + 1)
            self.elapsed_mean.append(new)
        else:
            self.elapsed_mean.append(elapsed)

        if self.elapsed_min:
            self.elapsed_min.append(min(self.elapsed_min[-1], elapsed))
        else:
            self.elapsed_min.append(elapsed)

        if self.elapsed_max:
            self.elapsed_max.append(max(self.elapsed_max[-1], elapsed))
        else:
            self.elapsed_max.append(elapsed)

        # Error
        self.error_live.append(error)
        if self.error_mean:
            cur = self.error_mean[-1]
            new = (cur * self.count + error) / (self.count + 1)
            self.error_mean.append(new)
        else:
            self.error_mean.append(error)

        if self.error_min:
            self.error_min.append(min(self.error_min[-1], error))
        else:
            self.error_min.append(error)

        if self.error_max:
            self.error_max.append(max(self.error_max[-1], error))
        else:
            self.error_max.append(error)

        self.count += 1

        self._update_elapsed()
        self._update_error()
        plt.draw()


class Series:
    """
    Represents a 2-dimensional series.
    """

    def __init__(self, fmt='r-', maxlen=100):
        self.x = collections.deque(maxlen=maxlen)
        self.y = collections.deque(maxlen=maxlen)
        self.fmt = fmt
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
