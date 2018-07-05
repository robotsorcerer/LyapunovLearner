import numpy as np
import scipy.linalg as LA

def guess_init_lyap(data, Vxf0, b_initRandom=False):
    """
    This function guesses the initial lyapunov function
    """
    Vxf0['Mu']  =  np.zeros(( Vxf0['d'], Vxf0['L']+1 )) # will be 2x2
    Vxf0['P']   =  np.zeros(( Vxf0['d'], Vxf0['d'], Vxf0['L']+1)) # wil be 2x2x3

    if b_initRandom:
        lengthScale = np.sqrt(np.var(data[:Vxf0['d'],:].T, axis=0))
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
            Vxf0['P'][:,:,l] = tempMat #lengthScaleMatrix.dot((tempMat * tempMat.T)).dot(lengthScaleMatrix)
    else:
        Vxf0['Priors'] = np.ones((Vxf0['L']+1, 1))
        Vxf0['Priors'] = Vxf0['Priors']/np.sum(Vxf0['Priors'])
        Vxf0['Mu']     = np.zeros((Vxf0['d'], Vxf0['L']+1))
        Vxf0['P']      = np.zeros(( Vxf0[ 'd'], Vxf0['d'], Vxf0['L']+1)) # wil be 2x2x3
        for l in range(Vxf0['L']+1):
            Vxf0['P'][:,:,l] = np.eye((Vxf0['d']))

    Vxf0.update(Vxf0)

    return Vxf0


class BundleType(object):
    """
    This class bundles many fields, similar to a record or a mutable
    namedtuple.
    """
    def __init__(self, variables):
        for var, val in variables.items():
            object.__setattr__(self, var, val)

    # Freeze fields so new ones cannot be set.
    # def __setattr__(self, key, value):
    #     if not hasattr(self, key):
    #         raise AttributeError("%r has no attribute %s" % (self, key))
    #     object.__setattr__(self, key, value)
