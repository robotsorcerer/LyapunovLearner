import numpy as np
import sys
from utils.utils import realmin

def gaussPDF(data, mu, sigma):
    """
     This function computes the Probability Density Function (PDF) of a
     multivariate Gaussian represented by means and covariance matrix.

     Inputs -----------------------------------------------------------------
       o data:  D x N array representing N datapoints of D dimensions.
       o mu:    D x K array representing the centers of the K GMM components.
       o sigma: D x D x K array representing the covariance matrices of the
                K GMM components.
     Outputs ----------------------------------------------------------------
       o prob:  1 x N array representing the probabilities for the
                N datapoints.

     Copyright (c) 2021 Lekan Molux, Microsoft Corp
    """
    global realmin
    num_vars, num_data = data.shape

    #print('data: ', data, 'mu: ', mu)
    diff = data.T - np.tile(mu.T, [num_data, 1])
    #print('diff: ', diff, 'mu: ', mu)
    prob = np.sum( (diff@np.linalg.inv(sigma))*diff, axis=1)
    #print('prob: ', prob) # this is corect

    if not isinstance(sigma, np.ndarray):
        temp = np.abs(sigma) + realmin
    else:
        temp = np.abs(np.linalg.det(sigma)) + realmin
    butt = np.sqrt(((2*np.pi)**num_vars)*temp)
    #print('butt: ', butt) # butt is correct
    prob = np.divide(np.exp(-0.5 * prob), butt)
    #print('prob: ', prob) # this is corect

    return prob

def GMR(priors, mu, sigma, x, traj, traj_deri):
    """
     This function performs Gaussian Mixture Regression (GMR), using the
     parameters of a Gaussian Mixture Model (GMM). Given partial input data,
     the algorithm computes the expected distribution for the resulting
     dimensions. By providing temporal values as inputs, it thus outputs a
     smooth generalized version of the data encoded in GMM, and associated
     constraints expressed by covariance matrices.

     Inputs -----------------------------------------------------------------
       o priors:  1 x K array representing the prior probabilities of the K GMM
                  components.
       o mu:      D x K array representing the centers of the K GMM components.
       o sigma:   D x D x K array representing the covariance matrices of the
                  K GMM components.
       o x:       P x N array representing N datapoints of P dimensions.
       o in:      1 x P array representing the dimensions to consider as
                  inputs.
       o out:     1 x Q array representing the dimensions to consider as
                  outputs (D=P+Q).
     Outputs ----------------------------------------------------------------
       o y:       Q x N array representing the retrieved N datapoints of
                  Q dimensions, i.e. expected means.
       o sigma_y: Q x Q x N array representing the N expected covariance
                  matrices retrieved.

     Copyright (c) 2021 Lekan Molux, Microsoft Corp

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

    init_conds = x.shape[1]
    num_vars = mu.shape[0]
    num_clusters = sigma.shape[2]

    global realmin

    priors = priors.squeeze()
    Pxi = np.empty((init_conds, num_clusters)) #[np.nan for i in range(num_clusters)]

    for i in range(num_clusters):
        mu_loc, sigma_loc = mu[np.ix_(*[traj, [i]])].squeeze(), \
                        sigma[np.ix_(*[traj, traj, [i]])].squeeze()
        pdf = gaussPDF(x, mu_loc, sigma_loc)
        #print(f'pdf: {pdf}')
        Pxi[:,i] = priors[i] * pdf
    butt = np.tile(np.expand_dims(np.sum(Pxi, axis=1), 1)+realmin, [1, num_clusters])
    beta = np.divide(Pxi, butt)
    # this is correct
    #% Compute expected means y, given input x
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    y_tmp = np.empty((x.shape)+(num_clusters,))
    for j in range(num_clusters):
        sig1 = sigma[np.ix_(*[traj_deri, traj, [j]])].squeeze()
        sig2 = sigma[np.ix_(*[traj, traj, [j]])].squeeze()
        sig_scaled = sig1@np.linalg.inv(sig2) #np.divide(sig1, sig2)

        mu_term1 = mu[np.ix_(*[traj_deri, [j]])]
        mu_term1 = np.tile(mu_term1, [1, init_conds])
        #print('mu_sub ', mu_term1, mu_term1.shape)

        mu_term2 = mu[np.ix_(*[traj, [j]])]
        mu_term2 = np.tile(mu_term2, [1, init_conds])
        xdiff = (x-mu_term2)
        #print('xdiff ', xdiff)
        #print('sig_scaled ', sig_scaled)
        y_tmp[:,:,j] = mu_term1  + sig_scaled@xdiff
        #print(f'y_tmp[:,:,{j}] , {y_tmp[:,:,j]}, {y_tmp[:,:,j].shape}')

    beta_tmp = np.reshape(beta, ((1,)+beta.shape))
    temp = np.tile(beta_tmp,[len(traj_deri), 1, 1])
    y_tmp2 = temp * y_tmp
    y = np.sum(y_tmp2,axis=2)
    # correct unto here

    ## Compute expected covariance matrices sigma_y, given trajut x
    #########################################################################
    temp0 = sigma[np.ix_(*[traj_deri, traj_deri, [0]])].squeeze() - \
            sigma[np.ix_(*[traj_deri, traj, [0]])].squeeze()@\
            np.linalg.inv(sigma[np.ix_(*[traj, traj, [0]])].squeeze())

    sigma_y_tmp = np.empty(temp0.shape +(1,num_clusters))
    sigma_y_tmp[:,:,0, 0]  = temp0
    for j in range(1, num_clusters):
        sigma_y_tmp[:,:,0, j] = sigma[np.ix_(*[traj_deri, traj_deri, [j]])].squeeze() - \
                                sigma[np.ix_(*[traj_deri, traj, [j]])].squeeze()@\
                                np.linalg.inv(sigma[np.ix_(*[traj, traj, [j]])].squeeze())
        beta_tmp = np.reshape(beta, ((1, 1,)+ beta.shape))
        sigma_y_tmp2 = np.tile(beta_tmp * beta_tmp, \
                        [len(traj_deri), len(traj_deri), 1, 1]) * \
                        np.tile(sigma_y_tmp, [1, 1, init_conds, 1])
        sigma_y = np.sum(sigma_y_tmp2, axis=3)
        #print('sigma_y: ', sigma_y)
    return y, sigma_y, beta
