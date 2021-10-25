"""
Realtime Plotter

The Realtime Plotter expects to be constantly given values to plot in realtime.
It assumes the values are an array and plots different indices at different
colors according to the spectral colormap.
"""
import time
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

from visualization.util import buffered_axis_limits


class TrajPlotter(object):

	def __init__(self, fig, gs, fontdict=None, pause_time=1e-3, \
					labels=None, x0=None):
		self._fig = fig
		self._gs = gridspec.GridSpec(1, 1, self._fig)
		plt.ion()
		self._ax = plt.subplot(self._gs[0])

		self._labels = labels
		self._init = False
		self.Xinit = x0
		self._fontdict  = fontdict
		self._labelsize = 16
		self.pause_time = pause_time

		if self._labels:
			self.init(x0.shape[-1])

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

		D = self.Xinit.shape[-1]

		# Plot it
		cm = plt.get_cmap('viridis')
		colors = ['red', 'magenta', 'orange']
		self._plots = [np.nan for _ in range(self.Xinit.shape[-1])]
		for i in range(D):
			self._plots[i] = [self._ax.plot(self.Xinit[0, i], self.Xinit[1, i], 'o', color=colors[i], markersize=10)[0]]
			self._plots[i] += [self._ax.plot([], [], color=colors[i], linewidth=2.5, label=f'Corrected Traj {i+1}')[0]]

		# Show me the attractor
		self.targ, = self._ax.plot([0],[0],'g*',markersize=15,linewidth=3, label='Target')

		self._ax.set_xlabel('$\\xi$', fontdict=self._fontdict)
		self._ax.set_ylabel('$\\dot{\\xi}$', fontdict=self._fontdict)

		self._ax.xaxis.set_tick_params(labelsize=self._labelsize)
		self._ax.yaxis.set_tick_params(labelsize=self._labelsize)

		self._ax.grid('on')
		self._ax.legend(loc='best')
		self._ax.set_title('CLF-Gaussian Trajectory Corrections', fontdict=self._fontdict)

		self._init = True

	def update(self, xi, xi_dot):
		"""
		Update the plots with new data x. Assumes x is a one-dimensional array.
		"""
		D = xi.shape[-1]  # number of conditions
		if not self._init:
			self.init(xi.shape[0])

		assert xi.shape[1] == self._data_len, f'xi of shape {xi.shape}has to be of shape {self._data_len}'
		assert xi_dot.shape[1] == self._data_len, f'xi_dot of shape {xi_dot.shape} has to be of shape {self._data_len}'

		self._t += 1
		self._xi = np.append(self._xi, xi[0].reshape(1, D), axis=0)
		self._xi_dot = np.append(self._xi_dot, xi[1].reshape(1, D), axis=0)

		cm = plt.get_cmap('ocean')

		idx=0
		for traj_plotter in self._plots:
			traj_plotter[-1].set_data(self._xi[:, idx], self._xi_dot[:, idx]) #

			x_min, x_max =np.amin(self._xi[:, idx]), np.amax(self._xi[:, idx])
			self._ax.set_xlim(buffered_axis_limits(x_min, x_max, buffer_factor=1.25))

			y_min, y_max = np.amin(self._xi_dot[:, idx]), np.amax(self._xi_dot[:, idx])
			self._ax.set_ylim(buffered_axis_limits(y_min, y_max, buffer_factor=1.25))

			idx+=1

		self.draw()
		time.sleep(self.pause_time)

	def draw(self):
		for plots in self._plots:
			for each_plot in plots:
				self._ax.draw_artist(each_plot)

		self._fig.canvas.draw()
		self._fig.canvas.flush_events()
