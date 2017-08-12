import numpy as np
from .gauss_pdf import gaussPDF

def EM(data, priors0, mu0, sigma0):
    """
     This function learns the parameters of a Gaussian Mixture Model
     (GMM) using a recursive Expectation-Maximization (EM) algorithm, starting
     from an initial estimation of the parameters.

     Inputs -----------------------------------------------------------------
       o data:    D x N array representing N datapoints of D dimensions.
       o priors0: 1 x K array representing the initial prior probabilities
                  of the K GMM components.
       o mu0:     D x K array representing the initial centers of the K GMM
                  components.
       o sigma0:  D x D x K array representing the initial covariance matrices
                  of the K GMM components.
     Outputs ----------------------------------------------------------------
       o priors:  1 x K array representing the prior probabilities of the K GMM
                  components.
       o mu:      D x K array representing the centers of the K GMM components.
       o sigma:   D x D x K array representing the covariance matrices of the
                  K GMM components.

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
    ## Criterion to stop the EM iterative update
    loglik_threshold = 1e-10;

    ## Initialization of the parameters
    nbVar, nbdata = data.shape
    nbStates = sigma0.shape[2]
    loglik_old = -np.finfo(np.float64).max;
    nbStep = 0;

    mu = mu0;
    sigma = sigma0;
    priors = priors0;

    ## EM fast matrix computation (see the commented code for a version
    ## involving one-by-one computation, which is easier to understand)
    max_iter = 50000;
    num_iter = 0;
    while (1 and num_iter <= max_iter):
        num_iter +=  1
      # E-step
      for i in range(nbStates):
        # Compute probability p(x|i)
        Pxi[:,i] = gaussPDF(data, mu[:,i], sigma[:,:,i])

      # Compute posterior probability p(i|x)
      Pix_tmp = np.tile(priors,[nbdata ,1])*Pxi
      Pix = Pix_tmp / np.tile(np.sum(Pix_tmp,axis=1),[1, nbStates])
      #Compute cumulated posterior probability
      E = np.sum(Pix)
      ## M-step ################################
      for i in range(nbStates):
        # Update the priors
        priors[i] = np.divide(E[i], nbdata)
        # Update the centers
        mu[:,i] = data*Pix[:,i] / E[i]
        # Update the covariance matrices
        data_tmp1 = data - np.tile(mu[:,i], [1,nbdata])
        data_tmp2a = np.tile(data_tmp1.reshape(np.r_[nbVar, 1, nbdata]), [1, nbVar, 1])
        data_tmp2b = np.tile(data_tmp1.reshape(np.r_[1, nbVar, nbdata]), [nbVar, 1, 1])
        data_tmp2c = np.tile(Pix[:,i].reshape(np.r_[1, 1, nbdata]), [nbVar, nbVar, 1])
        sigma[:,:,i] = np.sum(data_tmp2a * data_tmp2b * data_tmp2c, axis=2) / E[i]
        ## Add a tiny variance to avoid numerical instability
        sigma[:,:,i] += 1e-5 * np.diag(
                                        np.ones((nbVar,1))
                                      )
      ## Stopping criterion ####################
      for i in range(nbStates):
        #Compute the new probability p(x|i)
        Pxi[:,i] = gaussPDF(data, mu[:,i], sigma[:,:,i])

      # Compute the log likelihood
      F = Pxi.dot(priors.T)
      F[F<np.finfo(np.float64).min)] = np.finfo(np.float64).min
      loglik = np.sum(log(F))
      # Stop the process depending on the increase of the log likelihood
      if np.abs((loglik/loglik_old)-1) < loglik_threshold:
        break

      loglik_old = loglik
      nbStep += 1

    # ## EM slow one-by-one computation (better suited to understand the
    # ## algorithm)
    # while 1
    #   ## E-step ################################
    #   for i=1:nbStates
    #     #Compute probability p(x|i)
    #     Pxi(:,i) = gaussPDF(data, mu(:,i), sigma(:,:,i));
    #   end
    #   #Compute posterior probability p(i|x)
    #   for j=1:nbdata
    #     Pix(j,:) = (priors.*Pxi(j,:))./(sum(priors.*Pxi(j,:))+realmin);
    #   end
    #   #Compute cumulated posterior probability
    #   E = sum(Pix);
    #   ## M-step ################################
    #   for i=1:nbStates
    #     #Update the priors
    #     priors(i) = E(i) / nbdata;
    #     #Update the centers
    #     mu(:,i) = data*Pix(:,i) / E(i);
    #     #Update the covariance matrices
    #     covtmp = zeros(nbVar,nbVar);
    #     for j=1:nbdata
    #       covtmp = covtmp + (data(:,j)-mu(:,i))*(data(:,j)-mu(:,i))'.*Pix(j,i);
    #     end
    #     sigma(:,:,i) = covtmp / E(i);
    #   end
    #   ## Stopping criterion ####################
    #   for i=1:nbStates
    #     #Compute the new probability p(x|i)
    #     Pxi(:,i) = gaussPDF(data, mu(:,i), sigma(:,:,i));
    #   end
    #   #Compute the log likelihood
    #   F = Pxi*priors';
    #   F(find(F<realmin)) = realmin;
    #   loglik = mean(log(F));
    #   #Stop the process depending on the increase of the log likelihood
    #   if abs((loglik/loglik_old)-1) < loglik_threshold
    #     break;
    #   end
    #   loglik_old = loglik;
    #   nbStep = nbStep+1;
    # end
    ## Add a tiny variance to avoid numerical instability ###note
    for i in range(nbStates):
      sigma[:,:,i] += 1e-5*np.diag(
                                    np.ones((nbVar,1))
                                    )

    return priors, mu, sigma, nbStep, loglik
