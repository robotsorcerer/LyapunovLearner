from __future__ import print_function
import numpy as np
import logging
logger = logging.getLogger(__name__)

def matlength(x):
  # find the max of a numpy matrix dims
  return np.max(x.shape)

def gaussPDF(data, mu, sigma):
    nbVar, nbdata = data.shape

    data = data.T - np.tile(mu.T, [nbdata,1])
    prob = np.sum((data/sigma)*data, axis=1);
    prob = np.exp(-0.5*prob) / np.sqrt((2*np.pi)**nbVar *
                                       (np.abs(np.linalg.det(sigma))+
                                        np.finifo(np.float64).min))
    return prob

def GMR(Priors, Mu, Sigma, x, inp, out, nargout=3):
    nbData = x.shape[-1]
    nbVar = Mu.shape[0]
    nbStates = Sigma.shape[2]

    Pxi = np.zeros_like(Priors)
    for i in range(nbStates):
        Pxi[:,i] = Priors[i] * gaussPDF(x, Mu[inp,i], Sigma[inp,inp,i])

    beta = Pxi / np.tile(np.sum(Pxi,axis=1) +
                         np.finfo(np.float32).min, [1,nbStates])
    #########################################################################
    for j in range(nbStates):
        y_tmp[:,:,j] = np.tile(Mu[out,j],[1,nbData]) \
                     + Sigma[out,inp,j]/(Sigma[inp,inp,j]).dot(x-np.tile(Mu[inp,j],[1,nbData]))

    beta_tmp = beta.reshape(1, beta.shape)
    y_tmp2 = np.tile(beta_tmp,[matlength(out), 1, 1]) * y_tmp
    y = np.sum(y_tmp2,axis=2)
    ## Compute expected covariance matrices Sigma_y, given input x
    #########################################################################
    if nargout > 1:
        for j in range(nbStates):
            Sigma_y_tmp[:,:,0,j] = Sigma[out,out,j] \
                                   - (Sigma[out,inp,j]/(Sigma[inp,inp,j])  \
                                   * Sigma[inp,out,j])

        beta_tmp = beta.reshape(1, 1, beta.shape)
        Sigma_y_tmp2 = np.tile(beta_tmp * beta_tmp, \
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
        #print('P [',k, ']', list(range(i_c+k*(d**2)-1,i_c+(k+1)*(d**2)-1)))
        P[:,:,k] = p[range(i_c+k*(d**2)-1,i_c+(k+1)*(d**2)-1)].reshape(d,d)
        #print(P[:,:,k])

    Vxf           = dict()
    Vxf['Priors'] = Priors
    Vxf['Mu']     = Mu
    Vxf['P']      = P
    Vxf['SOS']    = 0

    return Vxf

def gmr_lyapunov(x, Priors, Mu, P):
    # print('x.shape: ', x.shape)
    nbData = x.shape[1]
    d = x.shape[0]
    L = P.shape[2]-1;

    # Compute the influence of each GMM component, given input x
    for k in range(L):
        P_cur               = P[:,:,k+1]
        if k                == 0:
            V_k             = np.sum(x * (P_cur.dot(x)), axis=0)
            V               = Priors[k+1]*(V_k)
            Vx              = Priors[k+1]*((P_cur+P_cur.T).dot(x))
        else:
            x_tmp           = x - np.tile(Mu[:,k+1], [nbData, 1]).T
            V_k             = np.sum(P_cur.dot(x_tmp)*x, axis=0)
            V_k[V_k < 0]    = 0
            V              += Priors[k+1] * (V_k ** 2)
            temp            = (2 * Priors[k+1]) * (V_k)
            Vx              = Vx + np.tile(temp, [d,1])*(P_cur.dot(x_tmp) + P_cur.T.dot(x))

    return V, Vx
