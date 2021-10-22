"""
Realtime Plotter

The Realtime Plotter expects to be constantly given values to plot in realtime.
It assumes the values are an array and plots different indices at different
colors according to the spectral colormap.
"""
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

from visualization.util import buffered_axis_limits


class TrajPlotter(object):

	def __init__(self, fig, gs, time_window=500, fontdict=None, \
					labels=None, alphas=None, x0=None):
		self._fig = fig
		self._gs = gridspec.GridSpec(1, 1, self._fig)
		self._ax = plt.subplot(self._gs[0])

		self._time_window = time_window
		self._labels = labels
		self._alphas = alphas
		self._init = False
		self.Xinit = x0
		self._fontdict  = fontdict
		self._labelsize = 16

		if self._labels:
			self.init(len(self._labels))

		self._fig.canvas.draw()
		self._fig.canvas.flush_events()

	def init(self, data_len):
		"""
		Initialize plots based off the length of the data array.
		"""
		self._t = 0
		self._data_len = data_len
		self._xi = np.empty((0, data_len))
		self._xi_dot = np.empty((0, data_len))

		cm = plt.get_cmap('ocean')
		self._plots = []
		for i in range(self.Xinit.shape[-1]):
			color = cm(1.0 * i / data_len)
			alpha = self._alphas[i] if self._alphas is not None else 1.0
			self._ax.plot(self.Xinit[0, i], self.Xinit[1, i], 'ko', markersize=20,  linewidth=2.5)
			self._ax.plot([], [], color=color, alpha=alpha, markersize=2, linewidth=2.5, label=f'CLF Corrected Traj {i+1}')

		self._ax.set_xlabel('$\\xi$', fontdict=self._fontdict)
		self._ax.set_ylabel('$\\dot{\\xi}$', fontdict=self._fontdict)

		self._ax.xaxis.set_tick_params(labelsize=self._labelsize)
		self._ax.yaxis.set_tick_params(labelsize=self._labelsize)

		self._ax.legend(loc='upper left', bbox_to_anchor=(0, 1.15))
		self._ax.set_title('Trajectory Corrections Plotter')

		self._init = True

	def update(self, xi, xi_dot):
		"""
		Update the plots with new data x. Assumes x is a one-dimensional array.
		"""
		xi = np.ravel([xi])
		xi_dot = np.ravel([xi_dot])

		if not self._init:
			self.init(xi.shape[0])

		assert xi.shape[0] == self._data_len
		xi = xi.reshape((1, self._data_len))

		self._t += 1
		self._xi = np.append(self._xi, xi, axis=0)
		self._xi_dot = np.append(self._xi_dot, xi, axis=0)

		# t, tw = self._t, self._time_window
		# t0, tf = (0, t) if t < tw else (t - tw, t)
		# for i in range(self._data_len):
		cm = plt.get_cmap('ocean')

		for j in range(self.Xinit.shape[-1]):
			color = cm(1.0 * j / self.Xinit.shape[-1])
			# self._ax.plot(Xinit[0, j], Xinit[1, j], 'ko', markersize=20,  linewidth=2.5)
			self._ax.plot(self._xi, self._xi_dot, color=color, markersize=2,\
				linewidth=2.5, label=f'CLF Corrected Traj {j+1}')
		# self._ax.plot(self._xi, self._xi_dot)

		x_min, x_max =np.amin(self._xi), np.amax(self._xi)
		self._ax.set_xlim(buffered_axis_limits(x_min, x_max, buffer_factor=1.25))

		y_min, y_max = np.amin(self._xi_dot), np.amax(self._xi_dot)
		self._ax.set_ylim(buffered_axis_limits(y_min, y_max, buffer_factor=1.25))

		self.draw()

	def draw(self):
		for plot in self._plots:
			self._ax.draw_artist(plot)
		self._fig.canvas.flush_events()
