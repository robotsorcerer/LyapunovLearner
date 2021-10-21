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
from gmm.gaussian_reg import regress_gauss_mix

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
parser.add_argument('--traj_nums', '-tn', type=int, default=10000, help='max # of trajectory stabilizations corrections before quitting' )
parser.add_argument('--rho0', '-rh', type=float, default=1.0, help='coeff. of class-Kappa function' )
parser.add_argument('--kappa0', '-kp', type=float, default=.1, help='exponential coeff. in class-Kappa function' )
parser.add_argument('--model', '-md', type=str, default='w', help='s|w ==> which model to use in training the data?' )
parser.add_argument('--off_priors', '-op', action='store_true', default=True, help='use KZ\'s offline priors or use ours')
parser.add_argument('--visualize', '-vz', action='store_true', default=True, help='visualize ROAs?' )
args = parser.parse_args()


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

	Vxf0['L'] = 1 #if args.model == 's' else 2
	Vxf0['d'] = data.shape[0]//2
	Vxf0.update(Vxf0)

	Vxf0 = guess_init_lyap(data, Vxf0, options['int_lyap_random'])
	cost = Cost()

	"Learn Lyapunov Function Strictly from Data"
	while cost.success:
		info('Optimizing the lyapunov function')
		Vxf, J = cost.optimize_lyapunov(Vxf0, data, options)
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

		Xinit = data[:Vxf['d'], demoIdx[0, :-1]]
		level_args = dict(disp=True, levels = [], save=True)
		viz.init_demos(Xinit, save=True)
		# Optimize and plot the level sets of the Lyapunov function
		viz.savedict["savename"]="level_sets_w.jpg"
		handles = viz.level_sets(Vxf, cost, **level_args)
		viz.draw()

	rho0 = args.rho0
	kappa0 = args.kappa0

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
	stab_args = {'time_varying': False, 'cost': cost}
	gmr_handle = lambda x: regress_gauss_mix(priors, mu, sigma, x, traj, traj_derivs)
	stab_handle = lambda x: stabilizer(x, gmr_handle, Vxf, rho0, kappa0, **stab_args) #, priors, mu, sigma

	# Hardcoding this since I know what to expect from the prerecorded demos
	if args.model =='s':
		args.traj_nums = 20e3
	elif args.model == 'w':
		args.traj_nums = 50e3
		ds_options['traj_nums'] = args.traj_nums
	ds_options['pause_time'] = args.pause_time
	traj_corr = CorrectTrajectories(Xinit, [], stab_handle, Bundle(ds_options))

	# if args.visualize:
	# 	traj_corr.Xinit = Xinit
	# 	traj_corr.model = args.model
	# 	viz.corrected_trajos(traj_corr, save=True)

	x_hist = np.stack(traj_corr.x_hist)
	xd_hist = np.stack(traj_corr.xd_hist)
	t_hist = np.stack(traj_corr.t_hist)
	xT = traj_corr.XT

	plt.ioff()
	plt.close('all')
	f = plt.figure(figsize=(16, 9))
	plt.clf()
	f.tight_layout()
	_fontdict = {'fontsize':16, 'fontweight':'bold'}

	_labelsize = 18
	nbSPoint = x_hist.shape[-1]
	plt.close('all')
	cm = plt.get_cmap('ocean')
	ax = f.gca()
	ax.grid('on')
	colors =['r', 'magenta', 'cyan']

	# plot the target attractors
	ax.plot(xT[0], xT[1], 'g*',markersize=20,linewidth=1.5, label='Target Attractor')

	for j in range(nbSPoint):
	    color = cm(1.0 * j / nbSPoint)
	    ax.plot(Xinit[0, j], Xinit[1, j], 'ko', markersize=20,  linewidth=2.5)
	    ax.plot(x_hist[:, 0, j], x_hist[:, 1, j], color=colors[j], markersize=2,\
			linewidth=2.5, label=f'CLF Corrected Traj {j}')

	ax.set_xlabel('$\\xi$', fontdict=_fontdict)
	ax.set_ylabel('$\\dot{\\xi}$', fontdict=_fontdict)

	ax.xaxis.set_tick_params(labelsize=_labelsize)
	ax.yaxis.set_tick_params(labelsize=_labelsize)

	ax.legend(loc='upper left') #, bbox_to_anchor=(-1, .5))
	ax.set_title(f'Corrected Trajectories in the Interval: [{t_hist[0]:.2f}, {t_hist[-1]:.2f}] secs', fontdict=_fontdict)

	savepath=join("scripts", "docs")
	f.savefig(join(savepath, f'corrected_traj_{traj_corr.model}.jpg'),
					bbox_inches='tight',facecolor='None')

	plt.show()


if __name__ == '__main__':
		global options
		options['disp'] = 0
		options['args'] = args

		options.update()
		main(Vxf0, options)
