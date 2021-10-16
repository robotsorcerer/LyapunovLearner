__author__ 		= "Lekan Molu (Microsoft Corp.)"
__copyright__ 	= "2018, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thomson (MIT), Pérez-Dattari, Rodrigo (TU Delft)"
__license__ 	= "MIT"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualizer():
    def __init__(self, fig=None, winsize=None,
                labelsize=18, linewidth=6, fontdict=None,
                block=False, savedict=None):
        """
            Ad-hoc visualizer for grids, grid partitions
            and HJI solutions

            fig: pyplot figure. If not passed, it's created
            ax: subfig of fig on which to plot figure
            winsize: size of pyplot window (default is (16,9))
            labelsize: size of plot x/y labels
            linewidth: width of 2D lines
            fontdict: fontweight and size for visualization
            block: whether to block screen after plot or not
        """
        self.linewidth = linewidth
        if fig is None:
            if winsize is None:
                winsize =(16, 9)
            self._fig = plt.figure(figsize=winsize)
        else:
            self._fig = fig
        self._fig.tight_layout()
        self.block=block

        self._labelsize = labelsize
        self._fontdict  = fontdict
        self._projtype = 'rectilinear'
        self.savedict = savedict

        if self.savedict["save"] and not os.path.exists(self.savedict["savepath"]):
            os.makedirs(self.savedict["savepath"])

        if self._fontdict is None:
            self._fontdict = {'fontsize':12, 'fontweight':'bold'}

    def level_sets(self, Vxf, D, cost_obj, num_points=300, data=None, \
                           disp=False, levels = np.array([]), odeFun=None):
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
        x = np.linspace(D[0][0], D[0][1], num_points)
        y = np.linspace(D[1][0], D[1][1], num_points)

        X, Y = np.meshgrid(x, y, indexing='ij')
        #X, Y = np.mgrid([x, y]) #, indexing='ij')
        xi_stacked = np.stack([np.ravel(X), np.ravel(Y)])

        V, dV = cost_obj.computeEnergy(xi_stacked, np.array(()), Vxf, nargout=2)

        if not levels.size:
            levels = np.arange(0, np.log(np.max(V)), 0.5)
            levels = np.exp(levels)
            if np.max(V) > 40:
                levels = np.round(levels)

        V = V.reshape(len(y), len(x))



        if disp:
            ax = self._fig.add_subplot(1, 1, 1)

            ax.contour(X, Y, V, levels, colors='k', origin='upper', linewidths=2)
            ax.plot(0, 0, 'g*', markersize=20, linewidth=self.linewidth, label='Target Attractor')
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
                ax.quiver(X, Y, dx, dy, dx, angles='xy')

            ax.set_title('Learned Lyapunov Function\'s ROA', fontdict=self._fontdict)

            ax.set_xlabel('Trajectory', fontsize=12, fontweight='bold')
            ax.set_ylabel('Dt(Trajectory)', fontsize=12, fontweight='bold')

            ax.xaxis.set_tick_params(labelsize=self._labelsize)
            ax.yaxis.set_tick_params(labelsize=self._labelsize)

            ax.legend(loc='best')

            self._fig.tight_layout()
            return ax
