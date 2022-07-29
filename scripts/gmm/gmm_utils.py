__author__ 		= "Lekan Molu"
__copyright__ 	= "Lekan Molu, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thomson (MIT), Pérez-Dattari, Rodrigo (TU Delft)"
__license__ 	= "MIT"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import numpy as np
import logging
logger = logging.getLogger(__name__)


def matlength(x):
  "find the max of a numpy matrix dims"
  return np.max(x.shape)

def stack_gmm_params(Vxf, options):
    """
        Transforms optimization parameters into a column vector
    """
    d = Vxf['d']
    if Vxf['L'] > 0:
        if options['optimizePriors']:
            p0 = np.vstack((np.expand_dims(np.ravel(Vxf['Priors']), axis=1),
                            np.expand_dims(Vxf['Mu'][:, 1:], axis=1).reshape(Vxf['L'] * d, 1)))
        else:
            p0 = Vxf['Mu'][:, 1:].reshape(Vxf['L'] * d, 1)
    else:
        p0 = np.array(())

    for k in range(Vxf['L']):
        p0 = np.vstack((p0, Vxf['P'][:, :, k].reshape(d ** 2, 1)))

    # For some weird reason, I cannot identify this bug so I manually add the last L+1 term outside the loop
    p0 = np.vstack((p0, Vxf['P'][:, :, Vxf['L']].reshape(d ** 2, 1)))

    return p0

def gauss_params_to_lyapunov(p, d, L, options):
    """
        Transform the column of Gaussian
        parameters into Gaussian Priors, Mu,
        and symmetric positive definite, P.
    """
    P = np.zeros((d, d, L + 1))
    optimizePriors = options['optimizePriors']
    if L == 0:
        Priors = 1
        Mu = np.zeros((d, 1))
        i_c = 0
    else:
        if optimizePriors:
            Priors = p[:L+1]
            i_c = L+1
        else:
            Priors = np.ones((L + 1, 1))
            i_c = 0

        Priors /= np.sum(Priors)
        mu_term1 = np.zeros((d, 1))
        p_idx = [i_c+x for x in range(d*L)]
        p_ind = p[p_idx].reshape(d, L)
        Mu = np.hstack((mu_term1, p_ind))
        i_c+=(d*L)

    for k in range(L):
        p_ind = np.arange(i_c+k*d**2, i_c+(k+1)*d**2, dtype=np.intp)
        P[:,:,k] = p[np.ix_(p_ind)].reshape(d, d)
    Vxf = dict(Priors=Priors,
               Mu=Mu,
               P=P,
               SOS=0)
    return Vxf

def gauss_regress_to_lyapunov(x, Priors, Mu, P):
    """
        Compute the Lyapunov Function from the
        Gaussian Mixture Regression Parameters.

        Given a dynamical system xdot = Ax, the Lyapunov
        function candidate is V(x) = x^T Px, where P is
        symmetric positive definite matrix.

        The time-derivative of V(x(t)) along the system trajectories
        is \dot{V} = \dot{x}^T P x + x^T P \dot{x}
                   = x^T (A^T P + PA) x
                   = -x^T Q x < 0.
                   Q is SPD.
                   The Lyapunov equation is A^TP + PA = -Q.
    """
    nbData = x.shape[1]
    d = x.shape[0]
    L = P.shape[2]-1
    "L is the number of WSAQFs in KZ's Paper."
    # Compute the influence of each GMM component, given input x
    for k in range(L):
        P_cur               = P[:, :, k]
        if k                == 0:
            "First term of the WSAQF term in the paper."
            V_k             = np.sum(x * (P_cur@x), axis=0)
            V               = Priors[k]*V_k
            Vx              = Priors[k]*((P_cur+P_cur.T)@x)
        else:
            x_tmp           = x - np.tile(Mu[:, k], [1, nbData])
            V_k             = np.sum((P_cur@x_tmp)*x, axis=0)
            V_k[V_k < 0]    = 0
            V               += Priors[k]*(V_k**2)
            Vx              += (np.tile(2*Priors[k]*V_k, [d, 1])*(P_cur@x_tmp + P_cur.T@x))

    return V, Vx
