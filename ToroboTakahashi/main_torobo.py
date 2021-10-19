from __future__ import print_function

__author__ 		= "Lekan Molu"
__copyright__ 	= "2018, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thomson (MIT), Pérez-Dattari, Rodrigo (TU Delft)"
__license__ 	= "MIT"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"
__doc__ 		= "This file for learning the stability dynamics for Torobo robot"

import os
import sys
import h5py
import time
import rospy
import argparse
import logging
import numpy as np

import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt

from os.path import dirname, abspath
takahashi = dirname(dirname(abspath(__file__))) + '/' + 'ToroboTakahashi'
sys.path.append(takahashi)

from gmm.gmm import GMM
from gmm.gmm_utils import GMR
from cost import Cost
from config import Vxf0, options, opt_exec
from utils.utils import guess_init_lyap, BundleType
from utils.data_prepro import format_data
from stabilizer.ds_stab import dsStabilizer
from robot_executor.execute import ToroboExecutor

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='torobo_parser')
parser.add_argument('--silent', '-si', type=int, default=0, help='max num iterations' )
parser.add_argument('--data_type', '-dt', type=str, default='pipe_et_trumpet', help='pipe_et_trumpet | h5_data' )
args = parser.parse_args()

print(args)


def main(Vxf0, urdf, options):
	gmm = GMM(num_clusters=options['num_clusters'])

	if options['args'].data_type == 'h5_data':
		filename = '{}.h5'.format('torobo_processed_data')
		with h5py.File(filename, 'r+') as f:
			data = f['data/data'].value
		logging.debug(''.format(data.shape))
	elif options['args'].data_type == 'pipe_et_trumpet':
		path = 'data'
		name = 'cart_pos.csv'
		data, data6d = format_data(path, name, learn_type='2d')

	if options['use_6d']:
		data = data6d

	Vxf0['d'] = int(data.shape[0]/2)
	Vxf0.update(Vxf0)

	Vxf0 = guess_init_lyap(data, Vxf0, options['int_lyap_random'])
	cost = Cost()

	# cost.success = False
	while cost.success:
		# cost.success = False
		Vxf, J = cost.learnEnergy(Vxf0, data, options)
		# if not cost.success:
		# increase L and restart the optimization process
		old_l = Vxf0['L']
		Vxf0['L'] += 1
		rospy.logwarn('Constraints violated. increasing the size of L from {} --> {}'.format(old_l, Vxf0['L']))
		if cost.success:
			rospy.logdebug('optimization succeeded without violating constraints')
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
	home_pos = [0.0]*7
	executor = ToroboExecutor(home_pos, urdf)
	x, xd = executor.optimize_traj(data6d, stab_handle, opt_exec)

if __name__ == '__main__':
	if args.silent:
		log_level = rospy.INFO
	else:
		log_level = rospy.DEBUG

	try:
		rospy.init_node('trajectory_optimization', log_level=log_level)
		rospy.loginfo('Trajectory optimization node started!')

		global options
		# options = BundleType(options)
		# A set of options that will be passed to the solver
		options['disp'] = 0
		options['args'] = args

		options.update()

		urdf = rospy.get_param('/robot_description')
		main(Vxf0, urdf, options)

		rospy.spin()
	except KeyboardInterrupt:
		rospy.logfatal("shutting down ros")
