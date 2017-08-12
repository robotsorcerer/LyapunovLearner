import numpy as np
from .gauss_pdf import gaussPDF
from .. learn_energy import matlength

def GMR(Priors, Mu, Sigma, x, inp, out, nargout=3):
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
       o inp:      1 x P array representing the dimensions to consider as
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
     Ported to python by Lekan Ogunmolu & Rachel Thompson
                          patlekano@gmail.com
                          August 12, 2017
    """


    nbData = x.shape[1]
    nbVar = Mu.shape[0]
    nbStates = Sigma.shape[2]

    ## Fast matrix computation (see the commented code for a version involving
    ## one-by-one computation, which is easier to understand).
    ##
    ## Compute the influence of each GMM component, given input x
    #########################################################################
    for i in range(nbStates):
      Pxi[:,i] = Priors[i] * gaussPDF[x, Mu[inp,i], Sigma[inp,inp,i]);

    beta = Pxi / np.tile(np.sum(Pxi,axis=1) +
                         np.finfo(np.float64).min, [1,nbStates])

    # ind = find(sum(beta,2) == 0);
    # if ~isempty(ind)
    #     for i=1:nbStates
    #         tmp = x(:,ind)' - np.tile(Mu(in,i)',length(ind),1);
    #         score(i,:) = Priors(i)*(sum((tmp/Sigma(in,in,i)).*tmp, 2));
    #     end
    #     [i i]=min(score);
    #     beta=beta';
    #     beta(nbStates.*(ind-1)+i')=1;
    #     beta=beta';
    # end
    ## Compute expected means y, given input x
    #########################################################################
    for j in range(nbStates):
      y_tmp[:,:,j] = np.tile(Mu[out,j],[1,nbData]) +
                     Sigma[out,inp,j]/(Sigma[inp,inp,j]).dot(x-np.tile(Mu[inp,j],[1,nbData]))

    beta_tmp = beta.reshape(1 beta.shape)
    y_tmp2 = np.tile(beta_tmp,[matlength(out), 1, 1]) * y_tmp
    y = np.sum(y_tmp2,axis=2)
    ## Compute expected covariance matrices Sigma_y, given input x
    #########################################################################
    if nargout > 1:
        for j in range(nbStates):
            Sigma_y_tmp[:,:,0,j] = Sigma[out,out,j] -
                                   (Sigma[out,inp,j]/(Sigma[inp,inp,j])*
                                    Sigma[inp,out,j])

        beta_tmp = beta.reshape(1, 1, beta.shape)
        Sigma_y_tmp2 = np.tile(beta_tmp * beta_tmp,
                               [matlength(out), matlength(out), 1, 1]) *
                                np.tile(Sigma_y_tmp,[1, 1, nbData, 1])
        Sigma_y = np.sum(Sigma_y_tmp2, axis=3)

    # ## Slow one-by-one computation (better suited to understand the algorithm)
    # ##
    # ## Compute the influence of each GMM component, given input x
    # #########################################################################
    # for i=1:nbStates
    #   Pxi(:,i) = gaussPDF(x, Mu(in,i), Sigma(in,in,i));
    # end
    # beta = (Pxi./np.tile(sum(Pxi,2)+np.finfo(np.float64).min,1,nbStates))';
    # ## Compute expected output distribution, given input x
    # #########################################################################
    # y = zeros(length(out), nbData);
    # Sigma_y = zeros(length(out), length(out), nbData);
    # for i=1:nbData
    #   # Compute expected means y, given input x
    #   #######################################################################
    #   for j=1:nbStates
    #     yj_tmp = Mu(out,j) + Sigma(out,in,j)*inv(Sigma(in,in,j)) * (x(:,i)-Mu(in,j));
    #     y(:,i) = y(:,i) + beta(j,i).*yj_tmp;
    #   end
    #   # Compute expected covariance matrices Sigma_y, given input x
    #   #######################################################################
    #   for j=1:nbStates
    #     Sigmaj_y_tmp = Sigma(out,out,j) - (Sigma(out,in,j)*inv(Sigma(in,in,j))*Sigma(in,out,j));
    #     Sigma_y(:,:,i) = Sigma_y(:,:,i) + beta(j,i)^2.* Sigmaj_y_tmp;
    #   end
    # end
    return y, Sigma_y, beta
