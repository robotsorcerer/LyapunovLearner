from __future__ import print_function

__author__ 		= "Olalekan Ogunmolu"
__copyright__ 	= "2018, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thomson (MIT), Jethro Tan (PFN)"
__license__ 	= "MIT"
__maintainer__ 	= "Olalekan Ogunmolu"
__email__ 		= "patlekano@gmail.com"
__status__ 		= "Testing"

import sys
import argparse
import logging
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from gmm.gmm import GMM

from os.path import dirname, abspath
lyap = dirname(dirname(abspath(__file__)))
sys.path.append(lyap)

from cost import Cost
from config import Vxf0, options, opt_exec
from utils.utils import guess_init_lyap
from stabilizer import dsStabilizer

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='torobo_parser')
parser.add_argument('--silent', '-si', type=int, default=0, help='max num iterations' )
parser.add_argument('--data_type', '-dt', type=str, default='pipe_et_trumpet', help='pipe_et_trumpet | h5_data' )
args = parser.parse_args()

print(args)


def load_saved_mat_file(x, **kwargs):
    matFile = sio.loadmat(x)

    data = matFile['Data']
    demoIdx = matFile['demoIndices']

    if ('Priors_EM' or 'Mu_EM' or 'Sigma_EM') in kwargs:
        Priors_EM, Mu_EM, Sigma_EM = matFile['Priors_EM'], matFile['Mu_EM'], matFile['Sigma_EM']
        return data, demoIdx, Priors_EM, Mu_EM, Sigma_EM
    else:
        return data, demoIdx


def main(Vxf0, urdf, options):
    modelNames = ['w.mat', 'Sshape.mat']  # Two example models provided by Khansari
    modelNumber = 1  # could be zero or one depending on the experiment the user is running

    data, demoIdx = load_saved_mat_file(lyap + '/' + 'example_models/' + modelNames[modelNumber])

    Vxf0['d'] = int(data.shape[0]/2)
    Vxf0.update(Vxf0)

    Vxf0 = guess_init_lyap(data, Vxf0, options['int_lyap_random'])
    cost = Cost()

    while cost.success:
        print('Optimizing the lyapunov function')
        Vxf, J = cost.learnEnergy(Vxf0, data, options)
        old_l = Vxf0['L']
        Vxf0['L'] += 1
        print('Constraints violated. increasing the size of L from {} --> {}'.format(old_l, Vxf0['L']))
        if cost.success:
            print('optimization succeeded without violating constraints')
            break

    # Plot the result of V
    h1 = plt.plot(data[0, :], data[1, :], 'r.', label='demonstrations')

    extra = 30

    axes_limits = [np.min(data[0, :]) - extra, np.max(data[0, :]) + extra,
                   np.min(data[1, :]) - extra, np.max(data[1, :]) + extra]

    h3 = cost.energyContour(Vxf, axes_limits, np.array(()), np.array(()), np.array(()), False)
    h2 = plt.plot(0, 0, 'g*', markersize=15, linewidth=3, label='target')
    plt.title('Energy Levels of the learned Lyapunov Functions', fontsize=12)
    plt.xlabel('x (mm)', fontsize=15)
    plt.ylabel('y (mm)', fontsize=15)
    h = [h1, h2, h3]

    # Run DS
    opt_sim = dict()
    opt_sim['dt'] = 0.01
    opt_sim['i_max'] = 4000
    opt_sim['tol'] = 1

    d = data.shape[0]/2  # dimension of data
    x0_all = data[:int(d), demoIdx[0, :-1] - 1]  # finding initial points of all demonstrations

    # get gmm params
    gmm = GMM(num_clusters=options['num_clusters'])
    gmm.update(data.T, K=options['num_clusters'], max_iterations=100)
    mu, sigma, priors = gmm.mu.T, gmm.sigma.T, gmm.logmass.T

    # rho0 and kappa0 impose minimum acceptable rate of decrease in the energy
    # function during the motion. Refer to page 8 of the paper for more information
    rho0 = 1
    kappa0 = 0.1

    inp = list(range(Vxf['d']))
    output = np.arange(Vxf['d'], 2 * Vxf['d'])

    xd, _ = dsStabilizer(x0_all, Vxf, rho0, kappa0, priors, mu, sigma, inp, output, cost)

    # Evalute DS
    xT = np.array([])
    d = x0_all.shape[0]  # dimension of the model
    if not xT:
        xT = np.zeros((d, 1))

    # initialization
    nbSPoint = x0_all.shape[1]
    x = []
    #x0_all[0, 1] = -180  # modify starting point a bit to see performance in further regions
    #x0_all[1, 1] = -130
    x.append(x0_all)
    xd = []
    if xT.shape == x0_all.shape:
        XT = xT
    else:
        XT = np.tile(xT, [1, nbSPoint])   # a matrix of target location (just to simplify computation)

    t = 0  # starting time
    dt = 0.01
    for i in range(4000):
        xd.append(dsStabilizer(x[i] - XT, Vxf, rho0, kappa0, priors, mu, sigma, inp, output, cost)[0])

        x.append(x[i] + xd[i] * dt)
        t += dt

    for i in range(nbSPoint):
        # Choose one trajectory
        x = np.reshape(x, [len(x), d, nbSPoint])
        x0 = x[:, :, i]
        if i == 0:
            plt.plot(x0[:, 0], x0[:, 1], linestyle='--', label='DS eval', color='blue')
        else:
            plt.plot(x0[:, 0], x0[:, 1], linestyle='--', color='blue')
    plt.legend()
    plt.show()


if __name__ == '__main__':
        global options
        # options = BundleType(options)
        # A set of options that will be passed to the solver
        options['disp'] = 0
        options['args'] = args

        options.update()

        urdf = []
        main(Vxf0, urdf, options)
