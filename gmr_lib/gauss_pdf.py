import numpy as np

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

     Copyright (c) 2006 Sylvain Calinon, LASA Lab, EPFL, CH-1015 Lausanne,
                   Switzerland, http://lasa.epfl.ch

    Ported to python by Lekan Ogunmolu & Rachel Thompson
                        patlekno@icloud.com
                        August 12, 2017
    """

    nbVar, nbdata = data.shape

    data = data.T - np.tile(mu.T, [nbdata,1])
    prob = np.sum((data/sigma)*data, axis=1);
    prob = np.exp(-0.5*prob) / np.sqrt((2*np.pi)**nbVar *
                                       (np.abs(np.linalg.det(sigma))+
                                        np.finifo(np.float64).min))

    return prob
