__author__ 		= "Lekan Molu"
__copyright__ 	= "2018, One Hell of a Lyapunov Learner"
__credits__  	= "Rachel Thomson (MIT), Pérez-Dattari, Rodrigo (TU Delft)"
__license__ 	= "MIT"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from gmm.gmm import GMM

from os.path import join, dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__) ) ) )

from cost import Cost
from config import Vxf0, options, ds_options
from stabilizer.traj_stab import stabilizer
from utils.utils import guess_init_lyap
from utils.dataloader import load_saved_mat_file
from utils.gen_utils import *
from visualization.visualizer import Visualizer

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Learning SEDs')
parser.add_argument('--silent', '-si', action='store_true', default=True, help='silent debug print outs' )
parser.add_argument('--pause_time', '-pz', type=float, default=0.3, help='pause time between successive updates of plots' )
parser.add_argument('--rho0', '-rh', type=float, default=1.0, help='coeff. of class-Kappa function' )
parser.add_argument('--kappa0', '-kp', type=float, default=.1, help='exponential coeff. in class-Kappa function' )
parser.add_argument('--model', '-md', type=str, default='w', help='s|w ==> which model to use in training the data?' )
parser.add_argument('--visualize', '-vz', action='store_true', default=True, help='visualize ROAs?' )
args = parser.parse_args()

def main(Vxf0, options):
	# modelNames = ['w.mat', 'Sshape.mat']  # Two example models provided by Khansari
	# modelNumber = 0  # could be zero or one depending on the experiment the user is running
	models = {'w': 'w.mat', 's': 'Sshape.mat'}
	#data, demoIdx = load_saved_mat_file(join('data', modelNames[modelNumber]))
	data, demoIdx = load_saved_mat_file(join('scripts/data', models[args.model]))

	Vxf0['d'] = int(data.shape[0]/2)
	Vxf0.update(Vxf0)

	Vxf0 = guess_init_lyap(data, Vxf0, options['int_lyap_random'])
	cost = Cost()

	"Learn Lyapunov Function Strictly from Data"
	while cost.success:
		info('Optimizing the lyapunov function')
		Vxf, J = cost.learnEnergy(Vxf0, data, options)
		old_l = Vxf0['L']
		Vxf0['L'] += 1
		print('Constraints violated. increasing the size of L from {} --> {}'.format(old_l, Vxf0['L']))
		if cost.success:
			print('optimization succeeded without violating constraints')
			break

	if args.visualize:
		fontdict = {'fontsize':12, 'fontweight':'bold'}
		plt.ion()
		fig = plt.figure(figsize=(16, 9))
		plt.clf()

		# Plot the result of V
		ax1 = fig.add_subplot(1, 1, 1)
		ax1.plot(data[0, :], data[1, :], 'r.', label='Demos')
		ax1.grid('on')
		ax1.legend(loc='upper right')
		ax1.set_xlabel('X', fontdict=fontdict)
		ax1.set_ylabel('Y', fontdict=fontdict)
		ax1.set_title('Demos on the WAM Robot')
		plt.pause(args.pause_time)

	num_points = data.shape[-1] # number of linear spacing between points
	offset = 30 # offset from trajectories
	X = [np.min(data[0,:]) - offset, np.max(data[0,:]) + offset]
	Y = [np.min(data[1,:]) - offset, np.max(data[1,:]) + offset]
	Data_Limits = [X, Y]

	fig = plt.figure(figsize=(16, 9))
	plt.clf()
	viz = Visualizer(fig, savedict={'save': None})
	level_args = dict(disp=True, num_points=data.shape[-1], \
					  data=data, levels = np.array([]))
	ax = viz.level_sets(Vxf, Data_Limits, cost, **level_args)
	plt.pause(args.pause_time)

	# get gmm params
	gmm = GMM(num_clusters=options['num_clusters'])
	gmm.update(data.T, K=options['num_clusters'], max_iterations=100)
	mu, sigma, priors = gmm.mu.T, gmm.sigma.T, gmm.logmass.T

	"""
		rho0 is essentially a class-Kappa function that imposes the minimum acceptable rate
		of energy decrease in the Lyapunov function.

		kapp0 is essentially related to rho0 in that it affects the deceleration (well, sort of)
		of the Lyapunov function as we carry out our optimization algorithm.

		The mathematical relationship is:

		\rho = \rho_0 (1 - exp(-kappa_0)||xi||)

		where xi is the trajectory.
	"""
	rho0 = args.rho0
	kappa0 = 0.1

	inp = list(range(Vxf['d']))
	output = np.arange(Vxf['d'], 2 * Vxf['d'])

	"Now stabilize the learned dynamics"
	X0 = data[:Vxf['d'], demoIdx[0, :-1]]
	xd, _ = stabilizer(X0, Vxf, rho0, kappa0, priors, mu, sigma, inp, output, cost)

	'Initializations'
	xT = np.zeros((X0.shape[0], 1)) # Evalute The DS.
	nbSPoint = X0.shape[1]

	x = np.zeros((ds_options["traj_nums"], ) + (X0.shape)) #
	x[0] = X0

	xd = np.zeros((ds_options["traj_nums"], ) + (X0.shape)) #[]
	if xT.shape == X0.shape:
		XT = xT
	else:
		XT = np.tile(xT, [1, nbSPoint])   # a matrix of target location (just to simplify computation)

	# starting time
	t = 0; 	dt = ds_options['dt']
	for i in range(ds_options["traj_nums"]):
		xd[i] = stabilizer(x[i] - XT, Vxf, rho0, kappa0, priors, mu, sigma, inp, output, cost)[0]
		x[i] += xd[i] * dt
		t += dt


	f = plt.figure(figsize=(12, 7))
	plt.clf()
	f.tight_layout()
	fontdict = {'fontsize':12, 'fontweight':'bold'}
	plt.ion()

	for i in range(nbSPoint):
		# Choose one trajectory
		x = np.reshape(x, [len(x), Vxf['d'], nbSPoint])
		x0 = x[:, :, i]

		plt.clf()
		ax = f.gca()

		if i == 0:
			ax.plot(x0[:, 0], x0[:, 1], linestyle='--', linewidth=4, label='Corrected Trajectories', color='blue')
		else:
			ax.plot(x0[:, 0], x0[:, 1], linestyle='--', linewidth=4, color='blue')
		ax.set_xlabel('X', fontdict=fontdict)
		ax.set_ylabel('Y', fontdict=fontdict)
		ax.set_title('Learned Demonstrations', fontdict=fontdict)
		ax.grid('on')
		plt.legend(loc='best')
		plt.pause(args.pause_time)


if __name__ == '__main__':
		global options
		options['disp'] = 0
		options['args'] = args

		options.update()
		main(Vxf0, options)
