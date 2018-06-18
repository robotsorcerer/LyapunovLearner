import h5py
import time
import logging
import numpy as np

import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

from .gmm import GMM, GMR
from .config import Vxf0, options
from .utils import guess_init_lyap
from .stabilizer import dsStabilizer

def main(Vxf0, options):
    gmm = GMM()

    filename = '../scripts/{}.h5'.format('torobo_processed_data')
    with h5py.File(filename, 'r+') as f:
        data = f['data/data'].value
    logging.debug(''.format(data.shape))

    Vxf0 = guess_init_lyap(data, Vxf0, b_initRandom=False)
    Vxf, J = learnEnergy(Vxf0, data, options)

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
    main(Vxf0, options)
