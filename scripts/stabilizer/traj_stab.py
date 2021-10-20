__author__ 		= "Lekan Molu"
__copyright__ 	= "2018, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thompson (MIT), Jethro Tan (PFN)"
__license__ 	= "MIT"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Testing"

import sys
import numpy as np
from utils.utils import realmin

def stabilizer(X, gmr_handle, Vxf, rho0, kappa0, **kwargs):
    """
         Syntax:

               [Xd u] = stabilizer(x,gmr_handle,Vxf,rho0,kappa0,kwargs)

         For a given (unstable) dynamical system f, this function computes a
         corrective command u such that Xd = f + u becomes globally asymptotically
         stable. Note that f could be autonomous (i.e. xd = f(x)) or
         non-autonomous (i.e. xd = f(t,x)).

         Inputs -----------------------------------------------------------------
           o X:       If f is an autonomous DS, then X is d by N matrix
                      representing N different query point(s) (each column of X
                      corresponds to each query point). If f is a non-autonomous
                      function, then X is a (d+1) by N matrix. In this case the
                      last row of X corresponds to time, for example X(d+1,10)
                      corresponds to the time at the 10th query point.

           o gmr_handle: This is a function handle that evaluates either f(t,x) or
                        f(x).

           o Vxf:     A structure variable representing the energy function. This
                      structure should follow the format explained in learnEnergy.m

           o rho0, kappa0: These parameters impose minimum acceptable rate of decrease
                           in the energy function during the motion. It computes
                           this lower bound from the following class \mathcal{K}
                           function:
                                   rho(\|x\|) = rho0 * ( 1 - exp(-kappa0 * \|x\|) )
                           Please refer to page 8 of the paper for more information.

           o kwargs:  An optional variable that provides dt (integration time
                        step) to the function, i.e. kwargs{1} = dt, dt>0.
                        Providing dt is useful, especially when using large
                        integration time step. Note that our whole stability proof
                        is based on continuous space assumption. When using large
                        time step, the effect of discretization become more
                        dominant and could cause oscillation. Bt providing dt, we
                        could alleviate this issue.

         Outputs ----------------------------------------------------------------

           o Xd:       A d x N matrix providing the output velocity after stabilization,
                       i.e. Xd = f + u

           o u:        A d x N matrix corresponding to the stabilizing command that
                       were generated to ensure stability of the dynamical system.
                       When u(:,i) = 0, it means the function f is stable by
                       itself at that query point, and no stabilizing command was
                       necessary. Note: we provide u as an output just for
                       information, you do NOT need to add it to the output
                       velocity!

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%         Copyright (c) 2014 Mohammad Khansari, LASA Lab, EPFL,       %%%
        %%%          CH-1015 Lausanne, Switzerland, http://lasa.epfl.ch         %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

         The program is free for non-commercial academic use. Please contact the
         author if you are interested in using the software for commercial purposes.
         The software must not be modified or distributed without prior permission
         of the authors. Please acknowledge the authors in any academic publications
         that have made use of this code or part of it. Please use this BibTex
         reference:

         S.M. Khansari-Zadeh and A. Billard (2014), "Learning Control Lyapunov Function
         to Ensure Stability of Dynamical System-based Robot Reaching Motions."
         Robotics and Autonomous Systems, vol. 62, num 6, p. 752-765.

         To get latest update of the software please visit
                                  http://cs.stanford.edu/people/khansari/

         Please send your feedbacks or questions to:
                                  khansari_at_cs.stanford.edu
    """
    if not 'cost' in kwargs:
        error('User must supply the Control Lyapunov Function Cost.')
    cost = kwargs['cost']

    # print('X ', X)
    d = Vxf['d']
    if X.shape[0] == 2*d:
        Xd = X[d:2*d, :]
        X = X[:d, :]
    else:
        if 'time_varying' in kwargs and not kwargs['time_varying']:
            Xd, _, _ = gmr_handle(X)
        elif 'time_varying' and kwargs['time_varying']:
            t = X[d,:]
            X = X[d,:]
            Xd, _, _ = gmr_handle(t,X)
        else:
            disp('Unknown GMR function handle!')
            return

    V, Vx = cost.computeEnergy(X, np.array(()), Vxf)

    norm_Vx = np.sum(Vx**2, axis=0)
    norm_x = np.sum(X**2, axis=0)

    Vdot = np.sum(Vx * Xd, axis=0)
    rho = rho0 * (1-np.exp(-kappa0 * norm_x)) * np.sqrt(norm_Vx)

    # print(f'Vdot: {Vdot}, rho: {rho}')
    ind = np.nonzero((Vdot + rho)>=0)[0] # not sure of this indexing
    # print(f'ind: {ind}')
    u = np.zeros_like(Xd, dtype=np.float64)

    if np.sum(ind) > 0:
        lambder = np.expand_dims(np.divide(Vdot[ind] + rho[ind], norm_Vx[ind]), 0) #+ realmin) # sys issues bruh)
        # print(f'u[:, ind]: {u[:, ind].shape}')
        #', np.tile(lambder, [d, 1]): {np.tile(lambder, [d, 1]).shape}, Vx: {Vx[:, ind].shape}')
        u[:, ind] = -np.tile(lambder, [d, 1]) * np.squeeze(Vx[:, ind])
        Xd[:, ind] = Xd[:, ind] + u[:, ind]

    if 'dt' in kwargs:
        dt = kwargs['dt']
        Xn = X + np.dot(Xd, dt)
        Vn = cost.computeEnergy(Xn, np.array(()), Vxf)
        ind = (Vn >= V)
        i = 0

        while np.any(ind) and i < 10:
            alpha = V[ind]/Vn[ind]
            Xd[:,ind] = np.tile(alpha, [d, 1]) * Xd[:, ind] - \
                        np.tile(alpha * np.sum(Xd[:, ind] * \
                        Vx[:, ind], axis=0)/norm_Vx[ind], [d, 1])*Vx[:, ind]
            Xn = x + np.dot(Xd, dt)
            Vn = cost.computeEnergy(Xn, np.array(()), Vxf)
            ind = Vn >= V
            i = i + 1

    return Xd, u
