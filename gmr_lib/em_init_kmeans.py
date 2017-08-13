import numpy as np
from scipy.cluster.vq import vq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from clfm_lib.learn_energy import matlength

def emInitKmeans(data, nbStates):
    """
     This function initializes the parameters of a Gaussian Mixture Model
     (GMM) by using k-means clustering algorithm.

     Inputs -----------------------------------------------------------------
       o data:     D x N array representing N datapoints of D dimensions.
       o nbStates: Number K of GMM components.
     Outputs ----------------------------------------------------------------
       o priors:   1 x K array representing the prior probabilities of the
                   K GMM components.
       o mu:       D x K array representing the centers of the K GMM components.
       o sigma:    D x D x K array representing the covariance matrices of the
                   K GMM components.
     Comments ---------------------------------------------------------------
       o This function uses the 'kmeans' function from the MATLAB Statistics
         toolbox. If you are using a version of the 'netlab' toolbox that also
         uses a function named 'kmeans', please rename the netlab function to
         'kmeans_netlab.m' to avoid conflicts.

     Copyright (c) 2006 Sylvain Calinon, LASA Lab, EPFL, CH-1015 Lausanne,
                   Switzerland, http://lasa.epfl.ch

     Ported to Python by Lekan Ogunmolu
                         patlekano@gmail.com
                         August 12, 2017
    """
    nbVar, nbdata = data.shape
    '''
    kmeans++:  selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence
    algorithm:  selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence
    '''
    kmeans = Kmeans(data=data.T, n_clusters=nbStates,max_iter=500,
                                 init='k-means++', n_jobs=4, random_state=0)
    data_id = kmeans.labels_
    centroids = kmeans.cluster_centers_
    mu = centroids.T
    for i in range(nbStates):
      idtmp = np.where(data_id==i)
      priors[i] = matlength(idtmp)
      sigma[:,:,i] = np.cov(
                            (np.r_[data[:,idtmp], data[:,idtmp]]).T
                            )
      # Add a tiny variance to avoid numerical instability
      sigma[:,:,i] = sigma[:,:,i] + 1e-5*np.diag(
                                                np.ones((nbVar,1))
                                                 )
    priors = priors / np.sum(priors)
    return priors, mu, sigma
