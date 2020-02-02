from __future__ import print_function

__author__ 		= "Olalekan Ogunmolu"
__copyright__ 	= "2018, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thompson (MIT), Jethro Tan (PFN)"
__license__ 	= "MIT"
__maintainer__ 	= "Olalekan Ogunmolu"
__email__ 		= "patlekano@gmail.com"
__status__ 		= "Testing"

import numpy as np


def gaussPDF(data, mu, sigma):
    a = data.shape
    nbVar, nbdata = data.shape

    data = data.T - np.tile(mu.T, [nbdata, 1])
    prob = np.sum(np.linalg.lstsq(sigma, data.T)[0].T * data, axis=1)
    prob = np.exp(-0.5 * prob) / np.sqrt((2 * np.pi)**nbVar * np.abs(np.linalg.det(sigma) + 1e-300))

    return prob.T


def GMR(Priors, Mu, Sigma, x, inp, out, nargout=0):
    nbData = x.shape[1]
    nbVar = Mu.shape[0]
    nbStates = Sigma.shape[2]

    ## Fast matrix computation (see the commented code for a version involving
    ## one-by-one computation, which is easier to understand).
    ##
    ## Compute the influence of each GMM component, given input x
    #########################################################################
    Pxi = []
    for i in range(nbStates):
      Pxi.append(Priors[0, i] * gaussPDF(x, Mu[inp, i], Sigma[inp[0]:(inp[1] + 1), inp[0]:(inp[1] + 1), i]))

    Pxi = np.reshape(Pxi, [len(Pxi), -1]).T
    beta = Pxi / np.tile(np.sum(Pxi, axis=1) + 1e-300, [nbStates, 1]).T

    #########################################################################
    y_tmp = []
    for j in range(nbStates):
        a = np.tile(Mu[out, j], [nbData, 1]).T
        b = Sigma[out, inp[0]:(inp[1] + 1), j]
        c = x - np.tile(Mu[inp[0]:(inp[1] + 1), j], [nbData, 1]).T
        d = Sigma[inp[0]:(inp[1] + 1), inp[0]:(inp[1] + 1), j]
        e = np.linalg.lstsq(d, b.T)[0].T
        y_tmp.append(a + e.dot(c))

    y_tmp = np.reshape(y_tmp, [nbStates, len(out), nbData])

    beta_tmp = beta.T.reshape([beta.shape[1], 1, beta.shape[0]])
    y_tmp2 = np.tile(beta_tmp, [1, len(out), 1]) * y_tmp
    y = np.sum(y_tmp2, axis=0)
    ## Compute expected covariance matrices Sigma_y, given input x
    #########################################################################
    Sigma_y_tmp = []
    Sigma_y = []
    if nargout > 1:
        for j in range(nbStates):
            Sigma_y_tmp.append(Sigma[out,out,j] - (Sigma[out,inp,j]/(Sigma[inp,inp,j]) * Sigma[inp,out,j]))

        beta_tmp = beta.reshape(1, 1, beta.shape)
        Sigma_y_tmp2 = np.tile(beta_tmp * beta_tmp, [len(out), len(out), 1, 1]) * np.tile(Sigma_y_tmp, [1, 1, nbData, 1])
        Sigma_y = np.sum(Sigma_y_tmp2, axis=3)
    return y, Sigma_y, beta


def dsStabilizer(x, Vxf, rho0, kappa0, Priors_EM, Mu_EM, Sigma_EM, inp, output, cost, *args):
    d = Vxf['d']
    if x.shape[0] == 2*d:
        xd = x[d+1:2*d, :]
        x = x[:d, :]
    else:
        xd, _, _ = GMR(Priors_EM, Mu_EM, Sigma_EM, x, inp, output)
    V, Vx = cost.computeEnergy(x, np.array(()), Vxf)
    norm_Vx = np.sum(Vx * Vx, axis=0)
    norm_x = np.sum(x * x, axis=0)
    Vdot = np.sum(Vx * xd, axis=0)
    rho = rho0 * (1-np.exp(-kappa0 * norm_x)) * np.sqrt(norm_Vx)
    ind = Vdot + rho >= 0
    u = xd * 0

    if np.sum(ind) > 0:
        lambder = (Vdot[ind] + rho[ind]) / norm_Vx[ind]
        u[:, ind] = -np.tile(lambder, [d, 1]) * Vx[:, ind]
        xd[:, ind] = xd[:, ind] + u[:, ind]

    if args:
        dt = args[0]
        xn = x + np.dot(xd, dt)
        Vn = cost.computeEnergy(xn, np.array(()), Vxf)
        ind = Vn >= V
        i = 0

        while np.any(ind) and i < 10:
            alpha = V[ind]/Vn[ind]
            xd[:,ind] = np.tile(alpha, [d, 1]) * xd[:, ind] - \
                        np.tile(alpha * np.sum(xd[:, ind] * \
                        Vx[:, ind], axis=0)/norm_Vx[ind], [d, 1])*Vx[:, ind]
            xn = x + np.dot(xd, dt)
            Vn = cost.computeEnergy(xn, np.array(()), Vxf)
            ind = Vn >= V
            i = i + 1

    return xd, u
