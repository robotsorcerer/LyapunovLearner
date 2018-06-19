import os
import sys
import h5py
import time
import logging
import numpy as np
from cost.cost import learnEnergy

import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt

from os.path import dirname, abspath#, join, sep
torcont = dirname(dirname(abspath(__file__))) + '/' + 'ToroboControl'
# print(torcont)
sys.path.append(torcont)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

from gmm.gmm import GMM
from gmm.gmm_utils import GMR
from config import Vxf0, options
from utils.utils import guess_init_lyap
from stabilizer.ds_stab import dsStabilizer

def main(Vxf0, options):
    gmm = GMM()

    filename = '../{}.h5'.format('torobo_processed_data')
    with h5py.File(filename, 'r+') as f:
        data = f['data/data'].value
    logging.debug(''.format(data.shape))

    Vxf0['L'] = 2   # number of asymmetric quadratic components for L >= 0
    Vxf0['d'] = int(data.shape[0]/2)
    Vxf0.update(Vxf0)

    filename = '../{}.h5'.format('torobo_processed_data')
    with h5py.File(filename, 'r+') as f:
        data = f['data/data'].value
    logging.debug(''.format(data.shape))

    Vxf0 = guess_init_lyap(data, Vxf0, b_initRandom=False)
    Vxf, J = learnEnergy(Vxf0, data, options)

    # for k, v in Vxf.items():
    #     print(k, v)
    # get gmm params
    gmm.update(data.T, K=6, max_iterations=100)
    mu, sigma, priors = gmm.mu, gmm.sigma, gmm.logmass

    logging.debug(' mu {}, sigma {}, priors: {}'.
                    format(mu.shape, sigma.shape, \
                    priors.shape))

    inp = range(0, Vxf['d'])
    out = range(Vxf['d'], 2* Vxf['d'])
    rho0, kappa0 = 1.0, 1.0

    gmr_handle = lambda x: GMR(priors, mu, sigma, x, inp, out)
    Xd, u = dsStabilizer(data, gmr_handle, Vxf, rho0, kappa0)
    logging.debug('Xd: {}, u: {}'.format(Xd.shape, u.shape))


if __name__ == '__main__':
    # A set of options that will be passed to the solver
    options = {
        'tol_mat_bias': 1e-1,
        'display': 1,
        'tol_stopping': 1e-10,
        'max_iter':500,
        'optimizePriors': True,
        'upperBoundEigenValue': True
    }
    main(Vxf0, options)
