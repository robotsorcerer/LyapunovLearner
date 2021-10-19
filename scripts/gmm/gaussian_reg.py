import numpy as np
import sys

realmin = sys.float_info.epsilon

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

    diff = data.T - np.tile(mu.T, [num_data, 1])
    prob = np.sum(np.divide(diff, sigma)*diff, axis=1)

    if not isinstance(sigma, np.ndarray):
        temp = sigma + realmin
        butt = np.sqrt(((2*np.pi)**num_vars)*temp)
    else:
        temp = np.abs(np.linalg.det(sigma) + realmin )
        butt = np.sqrt(((2*np.pi)**num_vars)@temp)

    prob = np.divide(np.exp(-0.5 * prob), butt)

    return prob.T

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
        Pxi[:,i] = priors[i] * gaussPDF(x, mu[traj, i], sigma[traj[0], traj[0], i])
    butt = np.tile(np.expand_dims(np.sum(Pxi, axis=1), 1)+realmin, [1, num_clusters])
    beta = np.divide(Pxi, butt) #for xs in Pxi]).T

    #% Compute expected means y, given input x
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    y_tmp = np.empty((x.shape)+(num_clusters,))
    for j in range(num_clusters):
        sig_scaled = np.divide(sigma[np.ix_(*[traj_deri, traj, [j]])], \
                               sigma[np.ix_(*[traj, traj, [j]])])
        y_tmp[:,:,j] = np.tile(mu[np.ix_(*(traj_deri, [j]))], (1, init_conds)) \
                        + sig_scaled.squeeze()@(x-\
                        np.tile(mu[np.ix_(*(traj, [j]))], (1, init_conds)))
    beta_tmp = np.reshape(beta, ((1,)+beta.shape))
    temp = np.tile(beta_tmp,[len(traj_deri), 1, 1])
    y_tmp2 = temp * y_tmp
    y = np.sum(y_tmp2,axis=2)

    ## Compute expected covariance matrices sigma_y, given trajut x
    #########################################################################
    temp0 = sigma[np.ix_(*[traj_deri, traj_deri, [0]])].squeeze() - \
                np.divide(sigma[np.ix_(*[traj_deri, traj, [0]])], \
                          sigma[np.ix_(*[traj, traj, [0]])]).squeeze()

    sigma_y_tmp = np.empty(temp0.shape +(1,num_clusters))
    sigma_y_tmp[:,:,0, 0]  = temp0
    for j in range(1, num_clusters):
        sigma_y_tmp[:,:,0, j] = sigma[np.ix_(*[traj_deri, traj_deri, [j]])].squeeze() - \
                np.divide(sigma[np.ix_(*[traj_deri, traj, [j]])], \
                          sigma[np.ix_(*[traj, traj, [j]])]).squeeze()
        beta_tmp = np.reshape(beta, ((1, 1,)+ beta.shape))
        sigma_y_tmp2 = np.tile(beta_tmp * beta_tmp, \
                        [len(traj_deri), len(traj_deri), 1, 1]) * \
                        np.tile(sigma_y_tmp, [1, 1, init_conds, 1])
        sigma_y = np.sum(sigma_y_tmp2, axis=3)

    return y, sigma_y, beta
