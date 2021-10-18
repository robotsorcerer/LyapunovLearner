__author__ 		= "Lekan Molu"
__copyright__ 	= "2018, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thomson (MIT), Pérez-Dattari, Rodrigo (TU Delft)"
__license__ 	= "MIT"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import logging
import numpy as np
import scipy.linalg
import scipy.linalg as LA
from scipy.cluster.vq import vq, kmeans2, whiten
from numpy.linalg import LinAlgError

logger = logging.getLogger(__name__)


def logsum(vec, axis=0, keepdims=True):
    #TODO: Add a docstring.
    maxv = np.max(vec, axis=axis, keepdims=keepdims)
    maxv[maxv == -float('inf')] = 0
    return np.log(np.sum(np.exp(vec-maxv), axis=axis, keepdims=keepdims)) + maxv

def check_sigma(A):
    """
        checks if the sigma matrix is symmetric
        positive definite before inverting via
        cholesky decomposition

        Lekan Molux. Circa, Summer 2018.
    """
    eigval = np.linalg.eigh(A)[0]
    if np.array_equal(A, A.T) and np.all(eigval>0):
        # logger.debug("sigma is pos. def. Computing cholesky factorization")
        return A
    else:
        # find lowest eigen value
        eta = 1e-6  # regularizer for matrix multiplier
        low = np.amin(np.sort(eigval))
        Anew = low * A + eta * np.eye(A.shape[0])
        return Anew

class GMM(object):
    """
        Gaussian Mixture Model.

        Includes regularization term for when cholesky
        factorization decreases.
    """
    def __init__(self, num_clusters=6, init_sequential=False,\
                    eigreg=False, warmstart=True):
        self.init_sequential = init_sequential
        self.eigreg = eigreg
        self.warmstart = warmstart
        self.sigma = None

        # Lekan June 26, 2018
        self.K    = num_clusters
        self.fail = None

        # regularization parameters
        self.eta = 1e-6
        self.delta = 1e-4
        self.eta_min = 1e-6
        self.delta_nut = 2

    def inference(self, pts):
        """
            Evaluate dynamics prior.
            Args:
                pts: A N x D array of points.
        """
        # Compute posterior cluster weights.
        logwts = self.clusterwts(pts)

        # Compute posterior mean and covariance.
        mu0, Phi = self.moments(logwts)

        # Set hyperparameters.
        m = self.N
        n0 = m - 2 - mu0.shape[0]

        # Normalize.
        m = float(m) / self.N
        n0 = float(n0) / self.N
        return mu0, Phi, m, n0

    def clusterwts(self, data):
        """
        Compute cluster weights for specified points under GMM.
        Args:
            data: An N x D array of points
        Returns:
            A K x 1 array of average cluster log probabilities.
        """
        # Compute probability of each point under each cluster.
        logobs = self.estep(data)

        # Renormalize to get cluster weights.
        logwts = logobs - logsum(logobs, axis=1)

        # Average the cluster probabilities.
        logwts = logsum(logwts, axis=0) - np.log(data.shape[0])
        return logwts.T

    def reg_sched(self, increase=False):
        # increase mu
        if increase:
            self.delta = max(self.delta_nut, self.delta * self.delta_nut)
            eta = self.eta * 1.1
        else: # decrease eta
            eta = self.eta
            eta *= 0.09
        self.eta = eta

    def estep(self, data):
        """
        Compute log observation probabilities under GMM.
        Args:
            data: A N x D array of points.
        Returns:
            logobs: A N x K array of log probabilities (for each point
                on each cluster).
        """
        # Constants.
        N, D = data.shape
        K = self.sigma.shape[0]

        logobs = -0.5*np.ones((N, K))*D*np.log(2*np.pi)

        self.fail = True
        while(self.fail):

            self.fail = False

            for i in range(K):
                # print('sigma i ', self.sigma[i].shape, np.eye(self.sigma[i].shape[-1]).shape)
                # print('eta: ', self.eta)
                self.sigma[i] += self.eta * np.eye(self.sigma[i].shape[-1])
                mu, sigma = self.mu[i], self.sigma[i]
                # logger.debug('sigma: {}\n'.format(sigma))
                try:
                    L = scipy.linalg.cholesky(sigma, lower=True)
                except LinAlgError as e:
                    logger.debug('LinAlgError: %s', e)
                    self.fail = True
                    # restart the for loop if sigma aint positive definite
                    logger.debug("sigma non-positive definiteness encountered; restarting")
                    break
                logobs[:, i] -= np.sum(np.log(np.diag(L)))
                diff = (data - mu).T
                soln = scipy.linalg.solve_triangular(L, diff, lower=True)
                logobs[:, i] -= 0.5*np.sum(soln**2, axis=0)

            if self.fail:
                old_eta = self.eta
                self.reg_sched(increase=True)
                logger.debug("Hessian became non positive definite")
                logger.debug('Increasing mu: {} -> {}'.format(old_eta, self.eta))
            else:
                # if successful, decrese mu
                old_eta = self.eta
                self.reg_sched(increase=False)
                logger.debug('Decreasing mu: {} -> {}'.format(old_eta, self.eta))

        logobs += self.logmass.T
        return logobs

    def moments(self, logwts):
        """
            Compute the moments of the cluster mixture with logwts.
            Args:
                logwts: A K x 1 array of log cluster probabilities.
            Returns:
                mu: A (D,) mean vector.
                sigma: A D x D covariance matrix.
        """
        # Exponentiate.
        wts = np.exp(logwts)

        # Compute overall mean.
        mu = np.sum(self.mu * wts, axis=0)

        # Compute overall covariance.
        diff = self.mu - np.expand_dims(mu, axis=0)
        diff_expand = np.expand_dims(self.mu, axis=1) * \
                np.expand_dims(diff, axis=2)
        wts_expand = np.expand_dims(wts, axis=2)
        sigma = np.sum((self.sigma + diff_expand) * wts_expand, axis=0)
        return mu, sigma

    def update(self, data, K=None, max_iterations=100):
        """
        Run EM to update clusters.
        Args:
            data: An N x D data matrix, where N = number of data points.
            K: Number of clusters to use.
        """
        # Constants.
        N  = data.shape[0]
        Do = data.shape[1]

        if K is None:
            K = self.K

        logger.debug('Fitting GMM with %d clusters on %d points.', K, N)

        if (not self.warmstart or self.sigma is None or K != self.sigma.shape[0]):
            # Initialization.
            logger.debug('Initializing GMM.')
            self.sigma = np.zeros((K, Do, Do))
            self.mu = np.zeros((K, Do))
            self.logmass = np.log(1.0 / K) * np.ones((K, 1))
            self.mass = (1.0 / K) * np.ones((K, 1))
            self.N = data.shape[0]
            N = self.N

            # Set initial cluster indices.
            use_kmeans = True
            if not self.init_sequential and not use_kmeans:
                cidx = np.random.randint(0, K, size=(1, N))
                for i in range(K):
                    cluster_idx = (cidx == i)[0]
                    mu = np.mean(data[cluster_idx, :], axis=0)
                    diff = (data[cluster_idx, :] - mu).T
                    sigma = (1.0 / K) * (diff.dot(diff.T))
                    self.mu[i, :] = mu
                    self.sigma[i, :, :] = sigma + np.eye(Do) * 2e-6
            else:
                # Initialize clusters with kmeans
                iter = 100000
                for j in range(iter):
                    self.mu, cidx = kmeans2(data, K)
                    for i in range(K):
                        cluster_idx = (np.reshape(cidx, [1, len(cidx)]) == i)[0]
                        sigma = np.cov(data[cluster_idx, :].T, data[cluster_idx, :].T)[:Do, :Do]
                        self.sigma[i, :, :] = sigma + np.eye(Do) * 2e-6

                    if not np.isnan(self.sigma).any():
                        break

                    if j == (iter - 1):
                        print('Initialization of gaussians in GMM failed.')
                        exit()

        prevll = -float('inf')
        for itr in range(max_iterations):
            # E-step: compute cluster probabilities.
            logobs = self.estep(data)

            # Compute log-likelihood.
            ll = np.sum(logsum(logobs, axis=1))
            logger.debug('GMM itr %d/%d. Log likelihood: %f',
                         itr, max_iterations, ll)
            if ll < prevll:
                # TODO: Why does log-likelihood decrease sometimes?
                logger.debug('Log-likelihood decreased! Ending on itr=%d/%d',
                             itr, max_iterations)
                break
            if np.abs(ll-prevll) < 1e-5*prevll:
                logger.debug('GMM converged on itr=%d/%d',
                             itr, max_iterations)
                break
            prevll = ll

            # Renormalize to get cluster weights.
            logw = logobs - logsum(logobs, axis=1)
            assert logw.shape == (N, K)

            # Renormalize again to get weights for refitting clusters.
            logwn = logw - logsum(logw, axis=0)
            assert logwn.shape == (N, K)
            w = np.exp(logwn)

            # M-step: update clusters.
            # Fit cluster mass.
            self.logmass = logsum(logw, axis=0).T
            self.logmass = self.logmass - logsum(self.logmass, axis=0)
            assert self.logmass.shape == (K, 1)
            self.mass = np.exp(self.logmass)

            # Reboot small clusters.
            w[:, (self.mass < (1.0 / K) * 1e-4)[:, 0]] = 1.0 / N
            # Fit cluster means.
            w_expand = np.expand_dims(w, axis=2)
            data_expand = np.expand_dims(data, axis=1)
            self.mu = np.sum(w_expand * data_expand, axis=0)
            # Fit covariances.
            wdata = data_expand * np.sqrt(w_expand)
            assert wdata.shape == (N, K, Do)
            for i in range(K):
                # Compute weighted outer product.
                XX = wdata[:, i, :].T.dot(wdata[:, i, :])
                mu = self.mu[i, :]
                self.sigma[i, :, :] = XX - np.outer(mu, mu)

                if self.eigreg:  # Use eigenvalue regularization.
                    raise NotImplementedError()
                else:  # Use quick and dirty regularization.
                    sigma = self.sigma[i, :, :]
                    self.sigma[i, :, :] = 0.5 * (sigma + sigma.T) + \
                            1e-6 * np.eye(Do)
