import time
import copy
import threading
from utils.gen_utils import *
from numpy import all, abs
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from visualization.realtime_plotter import RealTimePlotter

def run_demo(demo_func):
    demo_thread = threading.Thread(target=demo_func)
    demo_thread.daemon = True
    demo_thread.start()


def CorrectTrajectories(x0,xT,stab_handle,kwargs):
	"""
	 This function simulates motion that were learnt using SEDS, which defines
	 motions as a nonlinear time-independent asymptotically stable dynamical
	 systems:
								   xd=f(x)

	 where x is an arbitrary d dimensional variable, and xd is its first time
	 derivative.

	 The function can be called using:
		   [x xd t]=CorrectTrajectories(x0,xT,Priors,Mu,Sigma)

	 or
		   [x xd t]=CorrectTrajectories(x0,xT,Priors,Mu,Sigma,options)

	 to also send a structure of desired options.

	 Inputs -----------------------------------------------------------------
	   o x:       d x N matrix vector representing N different starting point(s)
	   o xT:      d x 1 Column vector representing the target point
	   o stab_handle:  A handle function that only gets as input a d x N matrix,
					 and returns the output matrix of the same dimension. Note
					 that the output variable is the first time derivative of
					 the input variable.

	   o options: A structure to set the optional parameters of the simulator.
				  The following parameters can be set in the options:
		   - .dt:      integration time step [default: 0.02]
		   - .i_max:   maximum number of iteration for simulator [default: i_max=1000]
		   - .plot     setting simulation graphic on (True) or off (False) [default: True]
		   - .tol:     A positive scalar defining the threshold to stop the
					   simulator. If the motions velocity becomes less than
					   tol, then simulation stops [default: 0.001]
		   - .perturbation: a structure to apply pertorbations to the robot.
							This variable has the following subvariables:
		   - .perturbation.type: A string defining the type of perturbations.
								 The acceptable values are:
								 'tdp' : target discrete perturbation
								 'tcp' : target continuous perturbation
								 'rdp' : robot discrete perturbation
								 'rcp' : robot continuous perturbation
		   - .perturbation.t0:   A positive scalar defining the time when the
								 perturbation should be applied.
		   - .perturbation.tf:   A positive scalar defining the final time for
								 the perturbations. This variable is necessary
								 only when the type is set to 'tcp' or 'rcp'.
		   - .perturbation.dx:   A d x 1 vector defining the perturbation's
								 magnitude. In 'tdp' and 'rdp', it simply
								 means a relative displacement of the
								 target/robot with the vector dx. In 'tcp' and
								 'rcp', the target/robot starts moving with
								 the velocity dx.

	 Outputs ----------------------------------------------------------------
	   o x:       d x T x N matrix containing the position of N, d dimensional
				  trajectories with the length T.

	   o xd:      d x T x N matrix containing the velocity of N, d dimensional
				  trajectories with the length T.

	   o t:       1 x N vector containing the trajectories' time.

	   o xT:      A matrix recording the change in the target position. Useful
				  only when 'tcp' or 'tdp' perturbations applied.

	   Copyright (c) Lekan Molux. https://scriptedonachip.com
	   2021.
	"""
	## parsing inputs
	if not kwargs:
		options = check_options()
	else:
		options = check_options(kwargs); # Checking the given options, and add ones are not defined.

	d=size(x0,0); #dimension of the model
	if not np.any(xT):
		xT = np.zeros((d,1), dtype=np.float64);

	if d!=size(xT,0):
		error(f'len(x0) should be equal to len(xT)!')
		x=[];xd=[];t=[];
		return

	# setting initial values
	nbSPoint=size(x0,1);

	#initialization
	x = x0.copy()
	xd = zeros(size(x))
	if size(xT) == size(x0):
		XT = xT;
	else:
		XT = np.tile(xT,[1,nbSPoint])

	t = 0; 	dt = options.dt

	i=0

	if options.plot:
		# Initialize Figure
		fig = plt.figure(figsize=options.winsize)
		gs = gridspec.GridSpec(1, 1)

		realtime_plotter = RealTimePlotter(fig, gs[0],
						labels=['$X_1$', '$X_2$', '$X_3$'], nbSPoint=nbSPoint,
			 			alphas=[.15, .15, .15], time_window=10)

		# f = plt.figure(figsize=options.winsize)
		# plt.clf()
		# f.tight_layout()

		# gs = gridspec.GridSpec(1, 1, f)
		# ax = plt.subplot(gs[0])

		plt.rcParams['toolbar'] = 'None'
		for key in plt.rcParams:
			if key.startswith('keymap.'):
				plt.rcParams[key] = ''

		# f.canvas.draw()
		# f.canvas.flush_events()
		# plots = [ax.plot([], [], 'ko', \
		# 			markersize=2, label='Trajs', \
		# 			linewidth=7.5)[0]]
	#
	x_hist, xd_hist, t_hist = [x], [xd], [t]
	#
	# def update(x, xT):
	#
	# 	# x = np.ravel(x)
	#
	# 	# ax = f.gca()
	# 	# ax.grid('on')
	# 	ax_loc = plots[0]
	# 	ax_loc.set_data(xT[0], xT[1]) #, 'g*',markersize=10,linewidth=1.5)
	# 	for j in range(nbSPoint):
	# 		ax_loc.set_data(x[0,j], x[1, j])
	#
	# 	# ax_loc.set_xlabel('X', fontdict=fontdict)
	# 	# ax_loc.set_ylabel('Y', fontdict=fontdict)
	# 	# ax_loc.legend(loc='best')
	# 	# ax_loc.set_title(f'Corrected Trajectories/Demo at time: {t:.3f}', fontdict=fontdict)
	# 	# plt.pause(options.pause_time)
	#
	# 	draw(f, ax)
	#
	# def draw(fig, ax):
	# 	ax.draw_artist(ax.patch)
	# 	for plot in plots:
	# 		ax.draw_artist(plot)
	# 	fig.canvas.flush_events()

	while True:
		#Finding xd using stab_handle.
		x_tilde = x - XT
		temp, u = stab_handle(x_tilde)
		# print(f'temp: {temp}') # xd: {xd.shape}, d: {d}, npSPoint: {nbSPoint}')
		xd = temp.reshape(d,nbSPoint)

		#############################################################################
		### Integration
		x = x + xd*options.dt
		t = t + options.dt

		# print(f't: {t}, {options.dt}, x: {x}')

		t_hist.append(t)
		x_hist.append(x)
		xd_hist.append(xd)

		# update(x, XT)
		if options.plot:
			realtime_plotter._time_window = t
			run_demo(realtime_plotter.update(x, xT))
			time.sleep(options.pause_time)

		# ax = f.gca()
		# ax.grid('on')
		# ax.plot(xT[0], xT[1], 'g*',markersize=10,linewidth=1.5)
		# for j in range(nbSPoint):
		# 	ax.set_data(x[0,j], x[1, j],  'ko', \
		# 				markersize=2, label='Trajs', \
		# 				linewidth=7.5)
		#
		# ax.set_xlabel('X', fontdict=options.fontdict)
		# ax.set_ylabel('Y', fontdict=options.fontdict)
		# ax.legend(loc='best')
		# ax.set_title(f'Corrected Trajectories/Demo at time: {t:.3f}', fontdict=options.fontdict)
		# plt.pause(options.pause_time)

		if (i > 3) and all(all(abs(xd_hist[:-3])<options.tol) or i>options.i_max-2):
			i += 1

			info(f'Traj Correction Iteration {i}')

			info(f'Final Time: {t:1.2f}')
			info(f'Final Point: {x}')
			info(f'Target Position: {xT[:,-1]}')
			info(f'########################################################')

			if i>options.i_max-2:
				info(f'Simulation stopped since it reaches the maximum number of allowed iterations {i}')
				info(f'Exiting without convergence!!! Increase the parameter ''options.i_max'' to handle this error.')
			break
		# del t_new, x_new, xd
		i += 1

	# traj_corr = Bundle(dict(x=x, xd=xd, xdot=xd, t_hist=t_hist, \
	# 				 x_hist=x_hist, xd_hist=xd_hist))

	traj_corr = Bundle(dict(XT=XT, t_hist=t_hist, \
					 x_hist=x_hist, xd_hist=xd_hist))

	return traj_corr


def  check_options(options=None):
	if not options:
		options = Bundle({})
	if not isfield(options,'dt'): # integration time step
		options.dt = 0.02
	if not isfield(options,'i_max'): # maximum number of iterations
		options.i_max = options.traj_nums
	if not isfield(options,'tol'): # convergence tolerance
		options.tol = 0.001
	if isfield(options,'plot') and options.plot: # shall simulator plot the figure
		options.winsize = (12, 7)
		options.labelsize=18
		options.linewidth=6
		options.fontdict = {'fontsize':16, 'fontweight':'bold'}

	return options
