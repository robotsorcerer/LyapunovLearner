__author__ 		= "Lekan Molu (Microsoft Corp.)"
__copyright__ 	= "2018, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thomson (MIT), Pérez-Dattari, Rodrigo (TU Delft)"
__license__ 	= "MIT"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

class Visualizer():
	def __init__(self, fig=None, winsize=(12,7),
				labelsize=18, linewidth=6, fontdict=None,
				savedict=None, labels=None, data=None,
				alphas=None):
		"""
			Ad-hoc visualizer for Lyapunov Function's Level Sets
			and demos, and corrected trajectories.

			Inputs:
				fig: pyplot figure. If not passed, it's created.

				winsize: size of pyplot window (default is (16,9)).

				labelsize: size of plot x/y labels.

				linewidth: width of 2D lines.

				fontdict: fontweight and size for visualization.

				savedict: Hashable specifying path and filenames to save with.
						Keys:
						  save: True/False -- Whether to save th eplotted trajos.
						  savepath: path with which to save the chart.
						  savename: name of the file.

			 Copyright (c) Lekan Molux. https://scriptedonachip.com
			 2021.
		"""
		if winsize:
			self._fig = plt.figure(figsize=winsize)

		self.winsize=winsize

		self._gs = gridspec.GridSpec(1, 1, self._fig)
		self._ax = plt.subplot(self._gs[0])

		self._labels = labels
		self._alphas = alphas
		self._init = False

		self._labelsize = labelsize
		self.linewidth = linewidth
		self._fontdict  = fontdict
		self.savedict = savedict

		if self.savedict["save"] and not os.path.exists(self.savedict["savepath"]):
			os.makedirs(self.savedict["savepath"])

		if self._fontdict is None:
			self._fontdict = {'fontsize':12, 'fontweight':'bold'}

		if self._labels:
			self.init(data)
			self._init = True

		self._fig.canvas.draw()
		self._fig.canvas.flush_events()

	def init(self, data=None):
		"""
			Initialize plots based off the data array.
			Inputs:
				num_points: Number of points to use for the data spacing;
							This is useful for quiver plots and such.

		"""
		self.data=data

		offset = 30 # offset from trajectories
		X = [np.min(data[0,:])-offset, np.max(data[0,:])+offset]
		Y = [np.min(data[1,:])-offset, np.max(data[1,:])+offset]
		self.data_lims = [X, Y]

		cm = plt.get_cmap('rainbow')
		self._plots = []
		D = len(data)//2
		linestyles = ['.', '*']

		for i in range(D-1):
			color = cm(1.0 * i / D)
			alpha = self._alphas[i] if self._alphas is not None else 1.0
			label = self._labels[i] if self._labels is not None else str(i)
			self._plots.append(
				self._ax.plot([], [], color=color, alpha=alpha, label=label)[0]
			)
		self._ax.legend(loc='upper left')#, bbox_to_anchor=(0, 1.15))
		self._ax.grid('on')

	def init_demos(self, Xinit, save=False):
		data = self.data
		if not self._init:
			self.init(data)

		fig = plt.figure(figsize=self.winsize)
		ax1 = fig.add_subplot(1, 1, 1)

		cm = plt.get_cmap('ocean')
		for j in range(Xinit.shape[-1]-1):
			color = cm(2.0 * j / Xinit.shape[-1])
			ax1.plot(Xinit[0, j], Xinit[1, j], 'ko', markersize=10,  linewidth=2.5)
		ax1.plot(Xinit[0, Xinit.shape[-1]-1], Xinit[1, Xinit.shape[-1]-1], 'ko', markersize=10,  linewidth=2.5, label='Init Conditions')

		demo_handle, = ax1.plot(data[0, :], data[1, :], 'm.', label='Demos')
		demo_handle, = ax1.plot([0],[0],'g*',markersize=25,linewidth=3, label='Target')
		ax1.grid('on')
		ax1.legend(loc='best')
		ax1.set_xlabel('$\\xi$', fontdict=self._fontdict)
		ax1.set_ylabel('$\\dot{\\xi}$', fontdict=self._fontdict)
		ax1.set_title('WAM Robot Task Space Demos', fontdict=self._fontdict)

		if save:
			fig.savefig(join(self.savedict["savepath"],self.savedict["savename"]),
						bbox_inches='tight',facecolor='None')
		self._plots.append(demo_handle)


	def level_sets(self, Vxf,  cost_obj, num_points=None,
					 disp=False, levels = [], save=False,
					 odeFun=None):
		"""
			Computes the energy function of a Lyapunov Function's
			LevelSets.

			Inputs:
				Vxf: The Lyapunov function object. See detailed exnumeration
					 of its attributes in the root config folder.
				cost_obj: Cost function instance that was used in learning the
						level sets of the Lyapunov Function.
				num_points: Number of points to use for the data spacing;
							This is useful for quiver plots and such.
				levels: Level set of the Lyapunov function to plot
		"""
		# Infer from data if no specific number of points given
		if not num_points:
			num_points = self.data.shape[-1] # number of linear spacing between points

		x = np.linspace(self.data_lims[0][0], self.data_lims[0][1], num_points)
		y = np.linspace(self.data_lims[1][0], self.data_lims[1][1], num_points)

		X, Y = np.meshgrid(x, y, indexing='ij')
		xi_stacked = np.stack([np.ravel(X), np.ravel(Y)])

		V, dV = cost_obj.compute_lyapunov(xi_stacked, np.array(()), Vxf, nargout=2)

		if not np.any(levels):
			levels = np.arange(0, np.log(np.max(V)), 0.5)
			levels = np.exp(levels)
			if np.max(V) > 40:
				levels = np.round(levels)

		V = V.reshape(len(y), len(x))

		if disp:
			ax = self._fig.add_subplot(1, 1, 1)

			h1  = ax.contour(X, Y, V, levels, colors='k', origin='upper', linewidths=2)
			h2,  = ax.plot(0, 0, 'g*', markersize=20, linewidth=self.linewidth, label='Target Attractor')
			#if odefun available, integrate dynamics and add quiver plots
			if odeFun is not None:
				(nr,nc) = X.shape;
				d = Vxf['d']; x1 = data[:d, :];  x2 = data[d:, :]
				dx = np.empty((nr, nc, 2))

				for i in range(nr):
					for j in range(nc):
						dx[i, j, :] = odefun(xd[:,nc])
				# Plot the quiver plot
				x1 = data[:d, :]
				dx, dy = np.meshgrid(x1[0,:], x1[1,:], indexing='ij')
				h3,  = ax.quiver(X, Y, dx, dy, dx, angles='xy')

			self._plots.append(h2)
			ax.set_title('Learned Lyapunov Function\'s ROA', fontdict=self._fontdict)

			ax.set_xlabel('$\\xi$', fontsize=12, fontweight='bold')
			ax.set_ylabel('$\\dot{\\xi}$', fontsize=12, fontweight='bold')

			ax.xaxis.set_tick_params(labelsize=self._labelsize)
			ax.yaxis.set_tick_params(labelsize=self._labelsize)

			ax.legend(loc='best')

			handles = [h1, h2]
			if odeFun:
				handles.extend([h3])
			self._fig.tight_layout()

			if save:
				self._fig.savefig(join(self.savedict["savepath"],self.savedict["savename"]),
							bbox_inches='tight',facecolor='None')

		return handles

	def draw(self):
		self._ax.draw_artist(self._ax.patch)
		for plot in self._plots:
			self._ax.draw_artist(plot)
		self._fig.canvas.flush_events()

	def corrected_trajos(self, trajos, save=True):
		# Plot the corrected trajectories
		x_hist = np.stack(trajos.x_hist)
		xd_hist = np.stack(trajos.xd_hist)
		t_hist = np.stack(trajos.t_hist)
		xT = trajos.XT
		Xinit = trajos.Xinit

		f = plt.figure(figsize=(self.winsize))
		plt.clf()
		f.tight_layout()

		# _labelsize = 18
		nbSPoint = x_hist.shape[-1]
		cm = plt.get_cmap('ocean')
		ax = f.gca()
		ax.grid('on')
		colors =['r', 'magenta', 'cyan']

		# plot the target attractors
		ax.plot(xT[0], xT[1], 'g*',markersize=20,linewidth=1.5, label='Target Attractor')

		for j in range(nbSPoint):
			color = cm(1.0 * j / nbSPoint)
			ax.plot(Xinit[0, j], Xinit[1, j], 'ko', markersize=20,  linewidth=2.5)
			ax.plot(x_hist[:, 0, j], x_hist[:, 1, j], color=colors[j], markersize=2,\
				linewidth=2.5, label=f'CLF Corrected Traj {j}')

		ax.set_xlabel('$\\xi$', fontdict=self._fontdict)
		ax.set_ylabel('$\\dot{\\xi}$', fontdict=self._fontdict)

		ax.xaxis.set_tick_params(labelsize=self._labelsize)
		ax.yaxis.set_tick_params(labelsize=self._labelsize)

		ax.legend(loc='upper left')
		ax.set_title(f'Corrected Trajectories in the Interval: [{t_hist[0]:.2f}, {t_hist[-1]:.2f}] secs', fontdict=self._fontdict)

		if save:
			f.savefig(join(self.savedict["savepath"], f'corrected_traj_{trajos.model}.jpg'),
						bbox_inches='tight',facecolor='None')

		plt.show()
