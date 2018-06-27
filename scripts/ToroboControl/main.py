from __future__ import print_function
import os
import sys
import h5py
import time
import argparse
import logging
import numpy as np

import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt
import rospy

from os.path import dirname, abspath#, join, sep
torcont = dirname(dirname(abspath(__file__))) + '/' + 'ToroboControl'
takahashi = dirname(dirname(abspath(__file__))) + '/../' + 'ToroboTakahashi'
# print(takahashi)
sys.path.append(torcont)
sys.path.append(takahashi)

from gmm.gmm import GMM
from gmm.gmm_utils import GMR
from cost.cost import learnEnergy
from config import Vxf0, options, opt_exec
from utils.utils import guess_init_lyap, BundleType
from utils.data_prepro import format_data
from stabilizer.ds_stab import dsStabilizer
from robot_executor.execute import ToroboExecutor

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='torobo_parser')
parser.add_argument('--silent', '-si', action='store_true', default=True, help='max num iterations' )
parser.add_argument('--data_type', '-dt', type=str, default='pipe_et_trumpet', help='pipe_et_trumpet | h5_data' )
args = parser.parse_args()

print(args)

def main(Vxf0, urdf, options):
	gmm = GMM(num_clusters=options['num_clusters'])

	if options['args'].data_type == 'h5_data':
		filename = '../{}.h5'.format('torobo_processed_data')
		with h5py.File(filename, 'r+') as f:
			data = f['data/data'].value
		logging.debug(''.format(data.shape))
	elif options['args'].data_type == 'pipe_et_trumpet':
		path = '../data'
		name = 'cart_pos.csv'
		data = format_data(path, name, learn_type=options['learn_type'])

	Vxf0['d'] = int(data.shape[0]/2)
	Vxf0.update(Vxf0)

	Vxf0 = guess_init_lyap(data, Vxf0, options['int_lyap_random'])
	Vxf, J = learnEnergy(Vxf0, data, options)

	# get gmm params
	gmm.update(data.T, K=options['num_clusters'], max_iterations=100)
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

	x0_all = data[0, :]
	XT     = data[-1, :]

	logger.debug('XT: {} x0: {} '.format(XT.shape, x0_all.shape))
	home_pos = [0.0]*7
	executor = ToroboExecutor(home_pos, urdf)

	rospy.logdebug("optimizing trajectories for reaching target")
	q, qd = executor.optimize_traj(data, stab_handle, opt_exec)

	# now run the trajectory

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
		# urdf = '/robot_description'
		main(Vxf0, urdf, options)

		rospy.spin()
	except KeyboardInterrupt:
		LOGGER.critical("shutting down ros")
