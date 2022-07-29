__author__ 		= "Lekan Molu"
__copyright__ 	= "Lekan Molu, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thomson (MIT), Pérez-Dattari, Rodrigo (TU Delft)"
__license__ 	= "MIT"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import numpy as np
import scipy.linalg as LA
import sys

realmin = sys.float_info.epsilon


def guess_init_lyap(data, Vxf0, b_initRandom=False):
    """
    This function guesses the initial lyapunov function
        Inputs:
            data: [D X N] data of demos
                   D = Data Dimension consisting of x and \dot{x} from
                   first order ode of dynamical system.
                   N = Number of points in demo.
            Vxf0: Hashable containing parameters that define the
                  Lyapunov Function.
                  Mu: Mean of Gaussian Prior.
                  P: Positive Symmetric Definite Matrix Used in Constructing
                  Lyapunov Function.
                  Priors: Priors of Gaussian Model.
            b_initRandom: Do we want a random initialization of the parameters of the
            Control Lyapunov Function?

     Copyright (c) Lekan Molux. https://scriptedonachip.com
     2021.
    """
    Vxf0['Mu'] = np.zeros(( Vxf0['d'], Vxf0['L'] + 1))
    Vxf0['P'] = np.zeros(( Vxf0['d'], Vxf0['d'], Vxf0['L'] + 1))

    if b_initRandom:
        lengthScale = np.sqrt(np.var(data[:Vxf0['d'], :].T, axis=0))
        lengthScale = np.ravel(lengthScale)
        '''
         If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
        '''
        #lengthScaleMatrix = LA.sqrtm(np.cov(np.var(data[:Vxf0['d'],:].T, axis=0), rowvar=False))
        Vxf0['Priors'] = np.random.rand(Vxf0['L']+1,1)

        for l in range(Vxf0['L']+1):
            tempMat = np.random.randn(Vxf0['d'], Vxf0['d'])
            Vxf0['Mu'][:,l] = np.random.randn(Vxf0['d']) #np.multiply(np.random.randn(Vxf0['d'],1), lengthScale)
            Vxf0['P'][:,:,l] = np.eye(Vxf0['d'])
    else:
        Vxf0['Priors'] = np.ones((Vxf0['L']+1, 1))
        Vxf0['Priors'] = Vxf0['Priors']/np.sum(Vxf0['Priors'])
        Vxf0['Mu'] = np.zeros((Vxf0['d'],  Vxf0['L']+1))

        for l in range(Vxf0['L']+1):
            Vxf0['P'][:,:,l] = np.eye(Vxf0['d'])

    Vxf0.update(Vxf0)

    return Vxf0
