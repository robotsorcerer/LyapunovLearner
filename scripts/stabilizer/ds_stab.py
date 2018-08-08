from __future__ import print_function

__author__ 		= "Olalekan Ogunmolu"
__copyright__ 	= "2018, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thomson (MIT), Jethro Tan (PFN)"
__license__ 	= "MIT"
__maintainer__ 	= "Olalekan Ogunmolu"
__email__ 		= "patlekano@gmail.com"
__status__ 		= "Testing"

import time
import numpy as np
# from inspect import getfullargspec
from cost import Cost

def dsStabilizer(X, gmr_handle, Vxf, rho0, kappa0):
    # print('X: ', X, X.shape)
    """
    This function takes the position and generates the cartesian velocity
    """
    d = Vxf['d']
    if X.shape[0] == 2*d:
        Xd     = X[d:2*d,:]
        X      = X[:d,:]
    else:
        Xd, _, _ = gmr_handle(X)

    cost = Cost()
    V,Vx    = cost.computeEnergy(X,[],Vxf)

    norm_Vx = np.sum(Vx ** 2, axis=0)
    norm_x  = np.sum(X ** 2,axis=0)

    Vdot    = np.sum(Vx * Xd,axis=0)
    rho     = rho0 * (1-np.exp(-kappa0*norm_x)) * np.sqrt(norm_Vx)
    ind     = np.where((Vdot + rho) >= 0)[0]
    u       = Xd * 0

    # print('u: {}, Xd: {}, Vdot: {}, rho {}, ind: {}'.format(u.shape, Xd.shape, Vdot.shape, rho.shape, ind))
    if np.sum(ind)>0:  # we need to correct the unstable points
        lambder   = (Vdot[ind] + rho[ind]) / (norm_Vx[ind] + 1e-10)
        # print('lambder: {}, Vx: {}, u: {}'.format(np.tile(lambder,[d,1]).shape, Vx[:,ind].shape, u[:,ind].shape))
        u[:,ind]  = -np.tile(lambder,[d,1]) * Vx[:,ind]
        Xd[:,ind] = Xd[:,ind] + u[:,ind]

    # print('u: ', u)
    # time.sleep(40)

    return Xd, u
