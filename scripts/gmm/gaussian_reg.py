import numpy as np
import sys

realmin = sys.float_info.epsilon

# def gaussLogProbs():
#     "This from gmm.py"
#     self.sigma[i] += self.eta * np.eye(self.sigma[i].shape[-1])
#     mu, sigma = self.mu[i], self.sigma[i]
#     # logger.debug('sigma: {}\n'.format(sigma))
#     try:
#         L = scipy.linalg.cholesky(sigma, lower=True)
#     except LinAlgError as e:
#         logger.debug('LinAlgError: %s', e)
#         self.fail = True
#         # restart the for loop if sigma aint positive definite
#         logger.debug("sigma non-positive definiteness encountered; restarting")
#         break
#     logobs[:, i] -= np.sum(np.log(np.diag(L)))
#     diff = (data - mu).T
#     soln = scipy.linalg.solve_triangular(L, diff, lower=True)
#     logobs[:, i] -= 0.5*np.sum(soln**2, axis=0)

def gaussPDF(data, mu, sigma):
    """
     This function computes the Probability Density Function (PDF) of a
     multivariate Gaussian represented by means and covariance matrix.

     Inputs -----------------------------------------------------------------
       o Data:  D x N array representing N datapoints of D dimensions.
       o Mu:    D x K array representing the centers of the K GMM components.
       o Sigma: D x D x K array representing the covariance matrices of the
                K GMM components.
     Outputs ----------------------------------------------------------------
       o prob:  1 x N array representing the probabilities for the
                N datapoints.

     Copyright (c) 2006 Sylvain Calinon, LASA Lab, EPFL, CH-1015 Lausanne,
                   Switzerland, http://lasa.epfl.ch
    """
    global realmin
    nbVar, nbdata = data.shape

    data = data.T - np.tile(mu.T, [nbdata, 1])
    prob = np.sum(np.divide(data, sigma)*data, axis=1)

    temp = np.abs(np.linalg.det(sigma) + realmin )
    prob = np.divide(np.exp(-0.5 * prob), np.sqrt(((2*np.pi)**nbVar)@temp))

    return prob.T

def GMR(Priors, Mu, Sigma, x, traj, traj_deri):
    """

     This function performs Gaussian Mixture Regression (GMR), using the
     parameters of a Gaussian Mixture Model (GMM). Given partial input data,
     the algorithm computes the expected distribution for the resulting
     dimensions. By providing temporal values as inputs, it thus outputs a
     smooth generalized version of the data encoded in GMM, and associated
     constraints expressed by covariance matrices.

     Inputs -----------------------------------------------------------------
       o Priors:  1 x K array representing the prior probabilities of the K GMM
                  components.
       o Mu:      D x K array representing the centers of the K GMM components.
       o Sigma:   D x D x K array representing the covariance matrices of the
                  K GMM components.
       o x:       P x N array representing N datapoints of P dimensions.
       o in:      1 x P array representing the dimensions to consider as
                  inputs.
       o out:     1 x Q array representing the dimensions to consider as
                  outputs (D=P+Q).
     Outputs ----------------------------------------------------------------
       o y:       Q x N array representing the retrieved N datapoints of
                  Q dimensions, i.e. expected means.
       o Sigma_y: Q x Q x N array representing the N expected covariance
                  matrices retrieved.

     Copyright (c) 2006 Sylvain Calinon, LASA Lab, EPFL, CH-1015 Lausanne,
                   Switzerland, http://lasa.epfl.ch

     The program is free for non-commercial academic use.
     Please contact the authors if you are interested in using the
     software for commercial purposes. The software must not be modified or
     distributed without prior permission of the authors.
     Please acknowledge the authors in any academic publications that have
     made use of this code or part of it. Please use this BibTex reference:

     @article{Calinon06SMC,
       title="On Learning, Representing and Generalizing a Task in a Humanoid
         Robot",
       author="S. Calinon and F. Guenter and A. Billard",
       journal="IEEE Transactions on Systems, Man and Cybernetics, Part B.
         Special issue on robot learning by observation, demonstration and
         imitation",
       year="2006",
       volume="36",
       number="5"
     }
    """
    nbData = x.shape[1]
    nbVar = Mu.shape[0]
    nbStates = Sigma.shape[2]

    global realmin

    ## Fast matrix computation (see the commented code for a version involving
    ## one-by-one computation, which is easier to understand).
    ##
    ## Compute the influence of each GMM component, given output x
    #########################################################################
    Pxi = [np.nan for i in range(nbStates)]
    Priors = Priors.squeeze()
    for i in range(nbStates):
      Pxi[i] = Priors[i] * gaussPDF(x, Mu[traj, i], Sigma[traj[0], traj[0], i])

    butt = np.tile(np.sum(Pxi, 1) + realmin, [1, nbStates]) #np.sum(Pxi, 1) + realmin #
    beta = np.divide(Pxi, butt) # Or rely on numpy broadcasting

    #########################################################################
    y_tmp = [np.nan for i in range(nbStates)]
    for j in range(nbStates):
        y_tmp[i] = Mu[traj_deri, j] + \
                    np.divide(Sigma[traj_deri, traj, j], \
                    Sigma[traj, traj, j])@(x-Mu[traj, j])

    beta_tmp = np.reshape(beta, ((1,)+beta.shape)) #beta.T.reshape([beta.shape[1], 1, beta.shape[0]])
    y_tmp2   = np.reshape(beta_tmp, (1, len(traj_deri), 1)) * y_tmp
    y        = np.sum(y_tmp2, axis=2)
    ## Compute expected covariance matrices Sigma_y, given trajut x
    #########################################################################
    Sigma_y_tmp = [np.nan for i in range(nbStates)]
    for j in range(nbStates):
        Sigma_y_tmp[:,:,0,j] = Sigma[traj_deri,traj_deri,j] - \
                                np.divide(Sigma[traj_deri,traj,j], \
                                Sigma[traj,traj,j])@Sigma[traj,traj_deri,j]

    beta_tmp = np.reshape(beta, ((1, 1,)+ beta.shape))
    Sigma_y_tmp2 = np.tile(beta_tmp * beta_tmp, \
                    [len(traj_deri), len(traj_deri), 1, 1]) * \
                    np.tile(Sigma_y_tmp, [1, 1, nbData, 1])
    Sigma_y = np.sum(Sigma_y_tmp2, axis=3)

    return y, Sigma_y, beta
