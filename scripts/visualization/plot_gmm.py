__author__ 		= "Lekan Molu"
__copyright__ 	= "Lekan Molu, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thomson (MIT), Pérez-Dattari, Rodrigo (TU Delft)"
__license__ 	= "MIT"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Testing"

import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
plt.style.use('fivethirtyeight')

def plotGMM(mu, sigma, color, display_mode, *args):
    '''
    inputs will be mu, sigma, color, display_mode

     This function plots a representation of the components (means and
     covariance amtrices) of a Gaussian Mixture Model (GMM) or a
     Gaussian Mixture Regression (regress_gauss_mix).

     Inputs -----------------------------------------------------------------
       o mu:           D x K array representing the centers of the K GMM components.
       o sigma:        D x D x K array representing the covariance matrices of the
                       K GMM components.
       o color:        Color used for the representation
       o display_mode: Display mode (1 is used for a GMM, 2 is used for a regress_gauss_mix
                       with a 2D representation and 3 is used for a regress_gauss_mix with a
                       1D representation).

     Copyright (c) 2006 Sylvain Calinon, LASA Lab, EPFL, CH-1015 Lausanne,
                   Switzerland, http://lasa.epfl.ch

    Ported to Python by Lekan Ogunmolu & Rachael Thompson
                        patlekno@icloud.com
                        August 12, 2016
    '''
    nbData = mu.shape[1]
    if not args:
        lightcolor = color + np.asarray([0.6,0.6,0.6]) #remove *5/3
        lightcolor[np.where(lightcolor>1.0)] = 1.0
    else:
        lightcolor=args[0]
    if display_mode==1:
        nbDrawingSeg = 40
        t = np.arange(-pi, pi, nbDrawingSeg).T
        for j in range(nbData):
            stdev = LA.sqrtm(3.0*sigma[:,:,j])
            X = np.r_[np.cos(t), np.sin(t)] * stdev.real() + np.tile(mu[:,j].T, [nbDrawingSeg,1])
            if lightcolor:
                # patch(X(:,1), X(:,2), lightcolor, 'line', 'none'); #linewidth=2
                NotImplemented
        plt.plot(X[:,1], X[:,2],'c')
      plt.plot(mu[1,:], mu[2,:], 'cx', linewidth=2, markersize=6)
    elif display_mode==2:
      nbDrawingSeg = 40
      lightcolor=np.array([0.7, 0.7, 0])
      t = np.arange(-np.pi, np.pi, nbDrawingSeg).T
      for j in range(nbData):
        stdev = LA.sqrtm(3.0 * sigma[:,:,j]) #1.0->3.0
        X = np.array([np.cos(t), np.sin(t)]) * stdev.real() + np.tile(mu[:,j].T, [nbDrawingSeg,1])
        # patch(X(:,1), X(:,2), lightcolor, 'LineStyle', 'none');
      plt.hold(True) #hold on
      plt.plot(mu[1,:], mu[2,:], 'cx', lineWidth=3, color=color);
    elif display_mode==3:
      for j in range(nbData):
        ymax[j] = mu[2,j] + LA.sqrtm(3*sigma[1,1,j])
        ymin[j] = mu[2,j] - LA.sqrtm(3*sigma[1,1,j])
    #   patch([mu(1,1:end) mu(1,end:-1:1)], [ymax(1:end) ymin(end:-1:1)], lightcolor, 'LineStyle', 'none')
      plt.plot(mu[1,:], mu[2,:], '-', linewidth=3, color=color)

"""
The functions below are meant to replace the matlab patch function

See this: http://matplotlib.org/examples/api/patch_collection.html
And this: https://stackoverflow.com/questions/26935701/ploting-filled-polygons-in-python
"""
def polyMeshFromMeshGrid(self, X, Y, Z, *args, **kwargs):
    rows, cols = Z.shape
    tX, tY, tZ = np.transpose(X), np.transpose(Y), np.transpose(Z)
    rstride = kwargs.pop('rstride', 10)
    cstride = kwargs.pop('cstride', 10)

    if 'facecolors' in kwargs:
        fcolors = kwargs.pop('facecolors')
    else:
        color = np.array(colorConverter.to_rgba(kwargs.pop('color', 'b')))
        fcolors = None

    cmap = kwargs.get('cmap', None)
    norm = kwargs.pop('norm', None)
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    shade = kwargs.pop('shade', cmap is None)
    lightsource = kwargs.pop('lightsource', None)

    # Shade the data
    if shade and cmap is not None and fcolors is not None:
        fcolors = self._shade_colors_lightsource(Z, cmap, lightsource)

    polys = kwargs.pop('polys', [])
    normals = kwargs.pop('normals', [])
    #colset contains the data for coloring: either average z or the facecolor
    colset = kwargs.pop('colset', [])

    for rs in np.arange(0, rows-1, rstride):
        for cs in np.arange(0, cols-1, cstride):
            ps = []
            corners = []
            for a, ta in [(X, tX), (Y, tY), (Z, tZ)]:
                ztop = a[rs][cs:min(cols, cs+cstride+1)]
                zleft = ta[min(cols-1, cs+cstride)][rs:min(rows, rs+rstride+1)]
                zbase = a[min(rows-1, rs+rstride)][cs:min(cols, cs+cstride+1):]
                zbase = zbase[::-1]
                zright = ta[cs][rs:min(rows, rs+rstride+1):]
                zright = zright[::-1]
                corners.append([ztop[0], ztop[-1], zbase[0], zbase[-1]])
                z = np.concatenate((ztop, zleft, zbase, zright))
                ps.append(z)

            # The construction leaves the array with duplicate points, which
            # are removed here.
            ps = zip(*ps)
            lastp = np.array([])
            ps2 = []
            avgzsum = 0.0
            for p in ps:
                if p != lastp:
                    ps2.append(p)
                    lastp = p
                    avgzsum += p[2]
            polys.append(ps2)

            if fcolors is not None:
                colset.append(fcolors[rs][cs])
            else:
                colset.append(avgzsum / len(ps2))

            # Only need vectors to shade if no cmap
            if cmap is None and shade:
                v1 = np.array(ps2[0]) - np.array(ps2[1])
                v2 = np.array(ps2[2]) - np.array(ps2[0])
                normals.append(np.cross(v1, v2))

    return polys, normals, colset


def plot_surfaceMesh(self, polys, normals, colset, *args, **kwargs):
    had_data = self.has_data()

    rows, cols = Z.shape
    rstride = kwargs.pop('rstride', 10)
    cstride = kwargs.pop('cstride', 10)

    if 'facecolors' in kwargs:
        fcolors = kwargs.pop('facecolors')
    else:
        color = np.array(colorConverter.to_rgba(kwargs.pop('color', 'b')))
        fcolors = None

    cmap = kwargs.get('cmap', None)
    norm = kwargs.pop('norm', None)
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    linewidth = kwargs.get('linewidth', None)
    shade = kwargs.pop('shade', cmap is None)
    lightsource = kwargs.pop('lightsource', None)

    polyc = art3d.Poly3DCollection(polys, *args, **kwargs)

    if fcolors is not None:
        if shade:
            colset = self._shade_colors(colset, normals)
        polyc.set_facecolors(colset)
        polyc.set_edgecolors(colset)
    elif cmap:
        colset = np.array(colset)
        polyc.set_array(colset)
        if vmin is not None or vmax is not None:
            polyc.set_clim(vmin, vmax)
        if norm is not None:
            polyc.set_norm(norm)
    else:
        if shade:
            colset = self._shade_colors(color, normals)
        else:
            colset = color
        polyc.set_facecolors(colset)

    self.add_collection(polyc)

    x, y, z = [], [], [];
    for pol in polys:
      for p in pol:
        x.append(p[0])
        y.append(p[1])
        z.append(p[2])

    self.auto_scale_xyz(x, y, z, had_data)
    return polyc
