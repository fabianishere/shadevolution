import collections

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_style("white")
plt.style.use('dark_background')

# TkAgg crashes on Windows
if matplotlib.get_backend() == 'TkAgg':
    matplotlib.use('Qt5Agg')


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
