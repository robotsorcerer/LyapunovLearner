from __future__ import print_function

__author__ 		= "Lekan Molu"
__copyright__ 	= "Lekan Molu, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thomson (MIT), Pérez-Dattari, Rodrigo (TU Delft)"
__license__ 	= "MIT"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Testing"

import numpy as np
import logging
logger = logging.getLogger(__name__)


def matlength(x):
  # find the max of a numpy matrix dims
  return np.max(x.shape)


def get_pdf(data, mu, sigma):
    if data.ndim == 1:
        nbVar, nbdata = 1, len(data)
    else:
        nbVar, nbdata = data.shape
    sigma_det = np.linalg.det(sigma)
    # else:
    #     nbVar, nbdata = 1, len(data)
    #     sigma = np.expand_dims(sigma, 1)
    #     sigma_det = np.linalg.norm(sigma, ord=2)

    data = data.T - np.tile(mu.T, [nbdata,1])
    prob = np.sum((data/sigma_det)*data, axis=1);
    # print('data: {}, prob pre: {}, sigma: {}, nbVar: {}'.format(data.shape, prob, sigma.shape, nbVar))
    prob = np.exp(-0.5*prob) / np.sqrt((2*np.pi)**nbVar *
                                       np.abs(sigma_det+1e-5))
    return prob


def regress_gauss_mix(Priors, Mu, Sigma, x, inp, out, nargout=3):
    nbData   = x.shape[-1] if x.ndim > 1 else 1
    nbVar    = Mu.shape[0]
    nbStates = Sigma.shape[2]//2

    # compute the influence of each GMM component, given input x
    Pxi = np.zeros((x.shape[0], nbStates))
    print('Pxi {} Priors: {}, Mu: {}, Sigma: {}, x: {}, nbVar: {}'
          .format(Pxi.shape, Priors.shape, Mu.shape, \
                  Sigma.shape, x.shape, nbVar))
    for i in range(nbStates):
        # print('Sigma[inp,inp,i]: ', Sigma[inp,inp,i].shape)
        gaussOutput = get_pdf(x, Mu[inp,i], Sigma[inp,inp,i])
        Pxi[:,i] = Priors[i] * gaussOutput

    # print('Pxi: {} Pxi tiled: {} '.format(Pxi.shape, \
    #       np.tile(np.sum(Pxi,axis=1) + \
    #             np.finfo(np.float32).min,\
    #             [nbStates,1]).T.shape))

    beta = np.divide(Pxi, np.tile(np.sum(Pxi,axis=1) + 1e-10, [nbStates, 1]).T)

    # Compute expected output distribution, given input x
    y = np.zeros((Pxi.shape[0], nbData))
    Sigma_y = np.zeros((Pxi.shape[0], Pxi.shape[0], nbData))
    # for 1D experiments, account for it in x
    if x.ndim < 2:
        x = np.expand_dims(x, -1)
    for i in range(nbData):
        # compute expected means y, given input x
        for j in range(nbStates):
            try:
                sigma_inv = np.linalg.inv(Sigma[inp, inp, j])
            except np.linalg.LinAlgError as e:
                logger.debug('LinAlgError: %s', e)
            yj_tmp = Mu[out, j] + Sigma[out, inp, j].dot(\
                        sigma_inv).dot(\
                        x[:,i]-Mu[inp, j])
            y[:,i] += beta[j,i] * yj_tmp
        # compute the expected covariance matrices Sigma_y, given input x
        for j in range(nbStates):
            # print('beta: {} | Sigma[inp, inp, j]: {}'
            #       .format(beta.shape, Sigma[inp, inp, j].shape))
            Sigmaj_y_tmp = Sigma[out, out, j] - (Sigma[out, inp, j].dot(np.linalg.inv(\
                                                Sigma[inp, inp, j])).dot(Sigma[inp, out, j]))
            Sigma_y[:,:,i] += Sigma_y[:,:,i] + (beta[j,i]**2) * Sigmaj_y_tmp

    return y, Sigma_y, beta


def stack_gmm_params(Vxf, options):
    # transforming optimization parameters into a column vector
    d = Vxf['d']
    if Vxf['L'] > 0:
        if options['optimizePriors']:
            p0 = np.vstack((np.expand_dims(np.ravel(Vxf['Priors']), axis=1),  # will be a x 1
                            np.expand_dims(Vxf['Mu'][:, 1:], axis=1).reshape(Vxf['L'] * d, 1)))
        else:
            p0 = Vxf['Mu'][:, 2:].reshape(Vxf['L'] * d, 1)  # print(p0) # p0 will be 4x1
    else:
        p0 = np.array(())

    for k in range(Vxf['L'] + 1):
        p0 = np.vstack((p0, Vxf['P'][k, :, :].reshape(d ** 2, 1)))

    return p0


def parameters_2_gmm(popt, d, L, options):
    # transforming the column of parameters into Priors, Mu, and P
    return gauss_params_to_lyapunov(popt, d, L, options)


def gauss_params_to_lyapunov(p, d, L, options):
    # transforming the column of parameters into Priors, Mu, and P
    P = np.zeros((L + 1, d, d))
    optimizePriors = options['optimizePriors']
    # print('options', optimizePriors)
    if L == 0:
        Priors = 1
        Mu = np.zeros((d, 1))
        i_c = 1
    else:
        if optimizePriors:  # options['optimizePriors']:
            Priors = p[:L + 1]
            i_c = L + 1
        else:
            Priors = np.ones((L + 1, 1))
            i_c = 0

        Priors = Priors / np.sum(Priors)
        Mu = np.hstack((np.zeros((d, 1)), np.transpose(np.reshape(p[[i_c + x for x in range(d * L)]], [L, d]))))
        i_c = i_c + d * L

    for k in range(L + 1):
        P[k, :, :] = np.transpose(p[range(i_c + k * (d ** 2), i_c + (k + 1) * (d ** 2))].reshape(d, d))

    Vxf = dict(Priors=Priors,
               Mu=Mu,
               P=P,
               SOS=0)

    return Vxf


def gauss_regress_to_lyapunov(x, Priors, Mu, P):
    # print('x.shape: ', x.shape)
    nbData = x.shape[1]
    d = x.shape[0]
    L = P.shape[0]-1
    # Compute the influence of each GMM component, given input x
    for k in range(L + 1):
        P_cur               = P[k, :, :]
        if k                == 0:
            V_k             = np.sum(x * (P_cur.dot(x)), axis=0)
            V               = Priors[k] * V_k
            Vx              = Priors[k] * ((P_cur+P_cur.T).dot(x))
        else:
            x_tmp           = x - np.tile(Mu[:, k], [nbData, 1]).T
            V_k             = np.sum(P_cur.dot(x_tmp)*x, axis=0)
            V_k[V_k < 0]    = 0
            Priors_reshaped = np.reshape(Priors[k], [1])
            V              += Priors_reshaped.dot(np.expand_dims(V_k ** 2, axis=0))
            temp            = (2 * Priors_reshaped).dot(np.expand_dims(V_k, axis=0))
            Vx              = Vx + np.tile(temp, [d, 1])*(P_cur.dot(x_tmp) + P_cur.T.dot(x))


    return V, Vx
