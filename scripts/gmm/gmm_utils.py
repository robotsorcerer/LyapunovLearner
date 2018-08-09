from __future__ import print_function

__author__ 		= "Olalekan Ogunmolu"
__copyright__ 	= "Olalekan Ogunmolu, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thomson (MIT), Jethro Tan (PFN)"
__license__ 	= "MIT"
__maintainer__ 	= "Olalekan Ogunmolu"
__email__ 		= "patlekano@gmail.com"
__status__ 		= "Testing"

import numpy as np
import logging
logger = logging.getLogger(__name__)

def matlength(x):
  # find the max of a numpy matrix dims
  return np.max(x.shape)

def gaussPDF(data, mu, sigma):
    if data.ndim > 1:
        nbVar, nbdata = data.shape
        sigma_det = np.linalg.det(sigma)
    else:
        nbVar, nbdata = 1, len(data)
        sigma = np.expand_dims(sigma, 1)
        sigma_det = np.linalg.norm(sigma, ord=2)

    data = data.T - np.tile(mu.T, [nbdata,1])
    prob = np.sum((data/sigma_det)*data, axis=1);
    # print('data: {}, prob pre: {}, sigma: {}, nbVar: {}'.format(data.shape, prob, sigma.shape, nbVar))
    prob = np.exp(-0.5*prob) / np.sqrt((2*np.pi)**nbVar *
                                       np.abs(sigma_det+1e-5))
    return prob

def GMR(Priors, Mu, Sigma, x, inp, out, nargout=3):
    nbData   = x.shape[-1]
    nbVar    = Mu.shape[0]
    nbStates = Sigma.shape[2]

    # Pxi = np.zeros_like(Priors)
    Pxi = np.zeros((x.shape[0], nbStates))

    # print('Pxi {} Priors: {}, Mu: {}, Sigma: {}, x: {} inp: {}, out: {}, nbVar: {}'
    #       .format(Pxi.shape, Priors.shape, Mu.shape, Sigma.shape, x.shape,
    #               inp.shape, out.shape, nbVar))

    # compute the influence of eacxh GMM component, given input x
    for i in range(nbStates):
        gaussOutput = gaussPDF(x, Mu[inp,i], Sigma[inp,inp,i])
        # print('Priors[i]: {} gaussPDF out: {} '.format(Priors[i], gaussOutput))
        if gaussOutput.ndim > 1:
            Pxi[:,i] = Priors[i] * gaussOutput
        else:
            Pxi[:,i] = Priors[i] * gaussOutput

    if gaussOutput.ndim > 1:
        beta = Pxi / np.tile(np.sum(Pxi,axis=1) + np.finfo(np.float32).min, [1,nbStates])
    else:
        beta = Pxi / np.tile(np.sum(Pxi,axis=1) + np.finfo(np.float32).min, [x.shape[0],2])

    #########################################################################
    y_tmp = np.zeros((nbData, nbData, nbStates))
    for j in range(nbStates):
        if gaussOutput.ndim > 1:
            y_tmp[:,:,j] = np.tile(Mu[out,j],[1,nbData])  + \
                            Sigma[out,inp,j]/(Sigma[inp,inp,j]).dot(x-np.tile(Mu[inp,j],[1,nbData]))
        else:
            # print('Mu[out,j]: {} | nbData: {}'.format(Mu[out,j].shape, nbData))
            # print('np.tile(Mu[out,j],[1,nbData]): {} | Sigma[out,inp,j]: {} | Sigma[inp,inp,j]: {} | x: {} |'
            #       ' np.tile(Mu[inp,j],[1,nbData]): {}'
            #       .format(np.tile(Mu[out,j],[1,nbData]).shape, Sigma[out,inp,j].shape,
            #               Sigma[inp,inp,j].shape, x, np.tile(Mu[inp,j],[1,nbData]).shape))
            y_tmp[:,:,j] = np.tile(Mu[out,j],[nbData, 1])  + \
                            Sigma[out,inp,j]/(Sigma[inp,inp,j]).dot(\
                            x-np.tile(Mu[inp,j],[nbData, 1]))

    beta_tmp = np.expand_dims(beta, 0)
    y_tmp2   = np.tile(beta_tmp,[len(out), 1, 1]) * y_tmp
    print('y_tmp2: ', y_tmp2.shape, ' y_tmp: ', y_tmp.shape)
    y        = np.sum(y_tmp2, axis=2)
    print('y: ', y.shape)
    ## Compute expected covariance matrices Sigma_y, given input x
    #########################################################################
    if nargout > 1:
        for j in range(nbStates):
            print('Sigma[out,out,j]: {} | Sigma[out,inp,j]: {} | Sigma[inp,inp,j]: {} | Sigma[inp,out,j]: {}'
                  .format(Sigma[out,out,j].shape, Sigma[out,inp,j].shape, Sigma[inp,inp,j].shape, Sigma[inp,out,j].shape))
            temp = Sigma[out,out,j] - (Sigma[out,inp,j]/(Sigma[inp,inp,j])  \
                                   * Sigma[inp,out,j])
            print('temp: ', temp.shape)
            Sigma_y_tmp[:,:,0,j] = None

        beta_tmp                = beta.reshape(1, 1, beta.shape)
        Sigma_y_tmp2            = np.tile(beta_tmp * beta_tmp, \
                                   [matlength(out), matlength(out), 1, 1]) * \
                                    np.tile(Sigma_y_tmp,[1, 1, nbData, 1])
        Sigma_y = np.sum(Sigma_y_tmp2, axis=3)

    return y, Sigma_y, beta

def gmm_2_parameters(Vxf, options):
    # transforming optimization parameters into a column vector
    d = Vxf['d']
    if Vxf['L'] > 0:
        if options['optimizePriors']:
            p0 = np.vstack((
                           np.expand_dims(np.ravel(Vxf['Priors']), axis=1), # will be a x 1
                           np.expand_dims(Vxf['Mu'][:,1:], axis=1).reshape(Vxf['L']*d,1)
                        ))
        else:
            p0 = Vxf['Mu'][:,2:].reshape(Vxf['L']*d, 1) #print(p0) # p0 will be 4x1
    else:
        p0 = []

    for k in range(Vxf['L']):
        p0 = np.vstack((
                      p0,
                      Vxf['P'][:,:,k+1].reshape(d**2,1)
                    ))
    # print('p0 in gmm: ', p0.shape)

    return p0

def parameters_2_gmm(popt, d, L, options):
    # transforming the column of parameters into Priors, Mu, and P
    return shape_DS(popt, d, L, options)

def shape_DS(p,d,L,options):
    # transforming the column of parameters into Priors, Mu, and P
    P = np.zeros((d,d,L+1))
    optimizePriors = options['optimizePriors']
    if L == 0:
        Priors = 1
        Mu = np.zeros((d,1))
        i_c = 1
    else:
        if options['optimizePriors']:
            Priors = p[:L+1]
            i_c = L+1
        else:
            Priors = np.ones((L+1,1))
            i_c = 0

        Priors = np.divide(Priors, np.sum(Priors))
        Mu = np.hstack((np.zeros((d,1)), p[[i_c+ x for x in range(d*L)]].reshape(d,L)))
        i_c = i_c+d*L+1

    for k in range(L):
        # print('p in range ', i_c+k*(d**2)-1, i_c+(k+1)*(d**2)-1, 'p shape ', p.shape)
        # P[:,:,k+1] = p[range(i_c+k*(d**2)-1,i_c+(k+1)*(d**2)-1)].reshape(d,d)
        P[:,:,k+1] = p[range(i_c+k*(d**2)-1,i_c+(k+1)*(d**2)-1)].reshape(d,d)

    Vxf           = dict(Priors = Priors,
                         Mu = Mu,
                         P = P,
                         SOS = 0)

    return Vxf

def gmr_lyapunov(x, Priors, Mu, P):
    # print('x.shape: ', x.shape)
    nbData = x.shape[1]
    d = x.shape[0]
    L = P.shape[-1]-1;


    # Compute the influence of each GMM component, given input x
    for k in range(L):
        P_cur               = P[:,:,k+1]
        x                   = x - np.expand_dims(x[:, -1], 1)     # subtract target from each x

        if k                == 0:
            V_k             = np.sum(x * (P_cur.dot(x)), axis=0)  # will be 1 x 10,000
            # V_k[V_k < 0]    = 0
            V               = Priors[k+1] * V_k                   # will be
            Vx              = Priors[k+1]*((P_cur+P_cur.T).dot(x))
        else:
            x_tmp           = x - np.tile(Mu[:,k+1], [nbData, 1]).T
            # print('x_tmp: ', x_tmp.shape, 'np.tile(Mu[:,k+1], [nbData, 1]).T: ', np.tile(Mu[:,k+1], [nbData, 1]).T.shape)
            V_k             = np.sum(P_cur.dot(x_tmp)*x, axis=0)
            V_k[V_k < 0]    = 0
            V              += Priors[k+1] * (V_k ** 2)
            temp            = (2 * Priors[k+1]) * (V_k)
            Vx              = Vx + np.tile(temp, [d,1])*(P_cur.dot(x_tmp) + P_cur.T.dot(x))

        # sanity check to be sure the lyapunov constraints are not being violated
        # print('V: ', len(np.unique(V[V<0])), ' Vx: ', len(np.unique(Vx[Vx>0])),  'x: ', x.shape)

    return V, Vx
