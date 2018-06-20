import os
import sys
import h5py
import time
import logging
import numpy as np

import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt

from os.path import dirname, abspath#, join, sep
torcont = dirname(dirname(abspath(__file__))) + '/' + 'ToroboControl'
# print(torcont)
sys.path.append(torcont)

from gmm.gmm import GMM
from gmm.gmm_utils import GMR
from cost.cost import learnEnergy
from config import Vxf0, options, opt_exec
from utils.utils import guess_init_lyap
from stabilizer.ds_stab import dsStabilizer
from robot_executor.execute import ToroboExecutor

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


def main(Vxf0, options):
    gmm = GMM()

    filename = '../{}.h5'.format('torobo_processed_data')
    with h5py.File(filename, 'r+') as f:
        data = f['data/data'].value
    logging.debug(''.format(data.shape))

    Vxf0['L'] = 2   # number of asymmetric quadratic components for L >= 0
    Vxf0['d'] = int(data.shape[0]/2)
    Vxf0.update(Vxf0)1

    filename = '../{}.h5'.format('torobo_processed_data')
    with h5py.File(filename, 'r+') as f:
        data = f['data/data'].value
    logging.debug(''.format(data.shape))

    Vxf0 = guess_init_lyap(data, Vxf0, b_initRandom=False)
    Vxf, J = learnEnergy(Vxf0, data, options)

    # get gmm params
    gmm.update(data.T, K=6, max_iterations=100)
    mu, sigma, priors = gmm.mu, gmm.sigma, gmm.logmass

    if options['disp']:
        logging.debug(' mu {}, sigma {}, priors: {}'.
                        format(mu.shape, sigma.shape, \
                        priors.shape))

    inp = range(0, Vxf['d'])
    out = range(Vxf['d'], 2* Vxf['d'])
    rho0, kappa0 = 1.0, 1.0

    gmr_handle = lambda x: GMR(priors, mu, sigma, x, inp, out)
    stab_handle = lambda dat: dsStabilizer(dat, gmr_handle, Vxf, rho0, kappa0)
    # Xd, u = dsStabilizer(data, gmr_handle, Vxf, rho0, kappa0)
    # logging.debug('Xd: {}, u: {}'.format(Xd.shape, u.shape))

    # do robot execution
    x0_all = data[:, :21]
    XT     = data[:, 14:]
    logger.debug('XT: {} xo: {} '.format(XT.shape, x0_all.shape)
    home_pos = [0.0]*7
    executor = ToroboExecutor(home_pos)
    x, xd = executor.execute(x0_all, XT, stab_handle, opt_exec)

if __name__ == '__main__':
    # A set of options that will be passed to the solver
    global options
    options = {
        'disp': 0,
    }
    options.update()
    main(Vxf0, options)
