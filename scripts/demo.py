__author__ 		= "Lekan Molu"
__copyright__ 	= "2018, One Hell of a Lyapunov Learner"
__credits__  	= "Rachel Thomson (MIT), Pérez-Dattari, Rodrigo (TU Delft)"
__license__ 	= "MIT"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import matplotlib as mpl
mpl.use('Qt5Agg')

import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from gmm.gmm import GMM
from gmm.gaussian_reg import GMR

from os.path import join, dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__) ) ) )

from cost import Cost
from config import Vxf0, options, ds_options
from stabilizer.traj_stab import stabilizer
from stabilizer.correct_trajos import CorrectTrajectories
from utils.utils import guess_init_lyap
from utils.dataloader import load_saved_mat_file
from utils.gen_utils import *
from visualization.visualizer import Visualizer

parser = argparse.ArgumentParser(description='Learning SEDs')
parser.add_argument('--silent', '-si', action='store_true', default=True, help='silent debug print outs' )
parser.add_argument('--pause_time', '-pz', type=float, default=0.3, help='pause time between successive updates of plots' )
parser.add_argument('--rho0', '-rh', type=float, default=1.0, help='coeff. of class-Kappa function' )
parser.add_argument('--kappa0', '-kp', type=float, default=.1, help='exponential coeff. in class-Kappa function' )
parser.add_argument('--model', '-md', type=str, default='w', help='s|w ==> which model to use in training the data?' )
parser.add_argument('--off_priors', '-op', action='store_true', default=True, help='use KZ\'s offline priors or use ours')
parser.add_argument('--visualize', '-vz', action='store_true', default=True, help='visualize ROAs?' )
args = parser.parse_args()
print(args)


if args.silent:
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
else:
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
# Turn off pyplot's spurious dumps on screen
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)

def main(Vxf0, options):
	models = {'w': 'w.mat', 's': 'Sshape.mat'}
	data, demoIdx, Priors_EM, Mu_EM, Sigma_EM = load_saved_mat_file(join('scripts/data', models[args.model]))

	Vxf0['L'] = 2 if args.model == 's' else 1
	Vxf0['d'] = data.shape[0]//2
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
		fontdict = {'fontsize':16, 'fontweight':'bold'}
		# https://matplotlib.org/stable/users/interactive.html
		plt.ion()

		savedict = dict(save=True, savename='demos_w.jpg',\
                savepath=join("..", "scripts/docs"))
		viz = Visualizer(winsize=(12, 7), savedict=savedict, data=data,
		                labels=['Trajs', 'Dt(Trajs)']*2, alphas = [.15]*4,
		                fontdict=fontdict)

		level_args = dict(disp=True, levels = [], save=True)
		viz.init_demos(save=True)
		# Optimize and plot the level sets of the Lyapunov function
		viz.savedict["savename"]="level_sets_w.jpg"
		handles = viz.level_sets(Vxf, cost, **level_args)
		viz.draw()

	rho0 = args.rho0
	kappa0 = args.kappa0 #0.1

	# get gmm params
	if args.off_priors:
		mu, sigma, priors = Mu_EM, Sigma_EM, Priors_EM
	else:
		gmm = GMM(num_clusters=options['num_clusters'])
		gmm.update(data.T, K=options['num_clusters'], max_iterations=100)
		mu, sigma, priors = gmm.mu.T, gmm.sigma.T, gmm.logmass.T

	"Now stabilize the learned dynamics"
	traj = list(range(Vxf['d']))
	traj_derivs = np.arange(Vxf['d'], 2 * Vxf['d'])
	# print(f'demoIdx: {demoIdx}')
	Xinit = data[:Vxf['d'], demoIdx[0, :-1]]
	stab_args = {'time_varying': False, 'cost': cost}
	gmr_handle = lambda x: GMR(priors, mu, sigma, x, traj, traj_derivs)
	stab_handle = lambda x: stabilizer(x, gmr_handle, Vxf, rho0, kappa0, **stab_args) #, priors, mu, sigma

	ds_options['pause_time'] = args.pause_time
	traj_corr = CorrectTrajectories(Xinit, [], stab_handle, Bundle(ds_options))


if __name__ == '__main__':
		global options
		options['disp'] = 0
		options['args'] = args

		options.update()
		main(Vxf0, options)
