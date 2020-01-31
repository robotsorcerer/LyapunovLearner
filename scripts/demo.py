from __future__ import print_function

__author__ 		= "Olalekan Ogunmolu"
__copyright__ 	= "2018, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thomson (MIT), Jethro Tan (PFN)"
__license__ 	= "MIT"
__maintainer__ 	= "Olalekan Ogunmolu"
__email__ 		= "patlekano@gmail.com"
__status__ 		= "Testing"

import os
import sys
import h5py
import time
import argparse
import logging
import numpy as np
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt

from os.path import dirname, abspath
lyap = dirname(dirname(abspath(__file__)))
sys.path.append(lyap)

from gmm.gmm import GMM
from gmm.gmm_utils import GMR
from cost.cost import Cost
from config import Vxf0, options, opt_exec
from utils.utils import guess_init_lyap, BundleType
from utils.data_prepro import format_data
from stabilizer.ds_stab import dsStabilizer

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='torobo_parser')
parser.add_argument('--silent', '-si', type=int, default=0, help='max num iterations' )
parser.add_argument('--data_type', '-dt', type=str, default='pipe_et_trumpet', help='pipe_et_trumpet | h5_data' )
args = parser.parse_args()

print(args)


def load_saved_mat_file(x, **kwargs):
    matFile = sio.loadmat(x)

    #print(matFile)
    data = matFile['Data']
    demoIdx = matFile['demoIndices']

    if ('Priors_EM' or 'Mu_EM' or 'Sigma_EM') in kwargs:
        Priors_EM, Mu_EM, Sigma_EM = matFile['Priors_EM'], matFile['Mu_EM'], matFile['Sigma_EM']
        return data, demoIdx, Priors_EM, Mu_EM, Sigma_EM
    else:
        return data, demoIdx


def main(Vxf0, urdf, options):
    gmm = GMM(num_clusters=options['num_clusters'])
    modelNames = ['w.mat', 'Sshape.mat']  # Two example models provided by Khansari
    modelNumber = 0  # could be zero or one depending on the experiment the user is running

    data, demoIdx = load_saved_mat_file(lyap + '/' + 'example_models/' + modelNames[modelNumber])

    Vxf0['d'] = int(data.shape[0]/2)
    Vxf0.update(Vxf0)

    Vxf0 = guess_init_lyap(data, Vxf0, options['int_lyap_random'])
    cost = Cost()

    # cost.success = False
    while cost.success:
        # cost.success = False
        print('Optimizing the lyapunov function')
        Vxf, J = cost.learnEnergy(Vxf0, data, options)
        # if not cost.success:
        # increase L and restart the optimization process
        old_l = Vxf0['L']
        Vxf0['L'] += 1
        print('Constraints violated. increasing the size of L from {} --> {}'.format(old_l, Vxf0['L']))
        if cost.success:
            print('optimization succeeded without violating constraints')
            break

    # get gmm params
    gmm.update(data.T, K=options['num_clusters'], max_iterations=100)
    mu, sigma, priors = gmm.mu, gmm.sigma, gmm.logmass

    if options['disp']:
        logging.debug(' mu {}, sigma {}, priors: {}'.
                        format(mu.shape, sigma.shape, priors.shape))

    inp = slice(0, Vxf['d']) #np.array(range(0, Vxf['d']))
    out = slice(Vxf['d'], 2* Vxf['d']) #np.array(range(Vxf['d'], 2* Vxf['d']))
    rho0, kappa0 = 1.0, 1.0

    gmr_handle = lambda x: GMR(priors, mu, sigma, x, inp, out)
    stab_handle = lambda dat: dsStabilizer(dat, gmr_handle, Vxf, rho0, kappa0)

    x0_all = data[0, :]
    XT     = data[-1, :]

    logger.debug('XT: {} x0: {} '.format(XT.shape, x0_all.shape))


if __name__ == '__main__':
        global options
        # options = BundleType(options)
        # A set of options that will be passed to the solver
        options['disp'] = 0
        options['args'] = args

        options.update()

        urdf = []
        main(Vxf0, urdf, options)
