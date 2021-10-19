import copy
from utils.gen_utils import *
from numpy import all, abs
from matplotlib import pyplot as plt

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

	Lekan Molux, MSFT Corp.
	"""
	## parsing inputs
	# options = check_options(kwargs)
	if not kwargs:
		options = check_options()
	else:
		options = check_options(kwargs); # Checking the given options, and add ones are not defined.

	# print('x0: ', x0.shape)
	d=size(x0,0); #dimension of the model
	if not np.any(xT):
		xT = np.zeros((d,1), dtype=np.float64);

	if d!=size(xT,0):
		error(f'len(x0) should be equal to len(xT)!')
		x=[];xd=[];t=[];
		return

	# setting initial values
	nbSPoint=size(x0,1); #number of starting points. This enables to simulatneously run several trajectories

	# # TODO
	# if isfield(options,'obstacle') and options.obstacle: #there is obstacle
	# 	obs_bool = True;
	# 	obs = options.obstacle;
	#     for n in range(obs):
	#         x_obs{n} = obs{n}.x0;
	#         if ~isfield(obs{n},'extra')
	#             obs{n}.extra.ind = 2;
	#             obs{n}.extra.C_Amp = 0.01;
	#             obs{n}.extra.R_Amp = 0.0;
	#         else
	#             if ~isfield(obs{n}.extra,'ind')
	#                 obs{n}.extra.ind = 2;
	#             end
	#             if ~isfield(obs{n}.extra,'C_Amp')
	#                 obs{n}.extra.C_Amp = 0.01;
	#             end
	#             if ~isfield(obs{n}.extra,'R_Amp')
	#                 obs{n}.extra.R_Amp = 0.0;
	#     b_contour = zeros(1,nbSPoint);
	# else
	#     obs_bool = False;
	#     obs = [];
	#     x_obs = np.nan #NaN

	#initialization
	x = x0[:,np.newaxis,:]
	for i in range(nbSPoint):
		x[:,0,i] = x0[:,i]

	# print('x: ', x.shape)
	xd = zeros(size(x))
	if size(xT) == size(x0):
		XT = xT;
	else:
		XT = np.tile(xT,[1,nbSPoint])
	# print('xT: ', xT.shape, ' XT: ', XT.shape)

	t = 0; 	dt = options.dt

	i=0

	if options.plot:
		plt.ion()

		f = plt.figure(figsize=options.winsize)
		plt.clf()
		f.tight_layout()
		fontdict = {'fontsize':12, 'fontweight':'bold'}
		plt.rcParams['toolbar'] = 'None'
		for key in plt.rcParams:
			if key.startswith('keymap.'):
				plt.rcParams[key] = ''
		plt.ion()

	while True:
		#Finding xd using stab_handle.
		print(f'x: {x.shape} XT: {XT.shape}, d: {d}, npSPoint: {nbSPoint}')
		x_tilde = np.squeeze(x[:,i,:])-XT
		temp = stab_handle(x_tilde)
		print(f'temp: {temp.size} xd: {xd.shape}, d: {d}, npSPoint: {nbSPoint}')
		xd[:,i,:]=np.reshape(temp,(d,1,nbSPoint))

		#############################################################################
		# # This part if for the obstacle avoidance module
		# if obs_bool
		#     # applying perturbation on the obstacles
		#     for n in range(obs):
		#         if isfield(obs[n],'perturbation'):
		#             if i >= round(obs{n}.perturbation.t0/options.dt)+1 and i <= round(obs{n}.perturbation.tf/options.dt) && length(obs{n}.perturbation.dx)==d
		#                 x_obs{n}(:,end+1) = x_obs{n}(:,end) + obs{n}.perturbation.dx*options.dt;
		#                 obs{n}.x0 = x_obs{n}(:,end);
		#                 xd_obs = obs{n}.perturbation.dx;
		#                 if options.plot #plotting options
		#                     plot_results('o',sp,x,xT,n,obs{n}.perturbation.dx*options.dt);
		#                 end
		#             else
		#                 x_obs{n}(:,end+1) = x_obs{n}(:,end);
		#                 xd_obs = 0;
		#             end
		#         else
		#             xd_obs = 0;
		#
		#     for j=1:nbSPoint
		#         [xd(:,i,j) b_contour(j)] = obs_modulation_ellipsoid(x(:,i,j),xd(:,i,j),obs,b_contour(j),xd_obs);

		#############################################################################
		### Integration
		x[:,i+1,:] = x[:,i,:] + xd[:,i,:]@options['dt']
		t[i+1] = t[i] + options['dt'];

		if i < 10:
			print(f'x: {x.shape}', end=", ")

		tStr = f'Corrected Trajectories/Demo at time: {t[i]:.3f}'
		# ax = f.gca()
		# ax.plot(xx[:,i+1,:], )
		# ax.set_xlabel('X', fontdict=fontdict)
		# ax.set_ylabel('Y', fontdict=fontdict)
		# ax.grid('on')
		# ax.set_title(tStr, fontsize=fontdict['fontsize'])

		#############################################################################
		# # Applying perturbation if any
		# if isfield(options, 'perturbations'):
		#     switch options.perturbation.type
		#         case 'tdp' #applying target discrete perturbation
		#             if i == round(options.perturbation.t0/options.dt)+1 && length(options.perturbation.dx)==d
		#                 xT(:,end+1) = xT(:,end) + options.perturbation.dx;
		#                 XT = repmat(xT(:,end),1,nbSPoint);
		#                 if options.plot #plotting options
		#                     plot_results('t',sp,x,xT);
		#             else
		#                 xT(:,end+1) = xT(:,end);
		#         case 'rdp' #applying robot discrete perturbation
		#             if i == round(options.perturbation.t0/options.dt)+1 && length(options.perturbation.dx)==d
		#                 x(:,i+1,:) = x(:,i+1,:) + repmat(options.perturbation.dx,[1 1 nbSPoint]);
		#             end
		#         case 'tcp' #applying target continuous perturbation
		#             if i >= round(options.perturbation.t0/options.dt)+1 && i <= round(options.perturbation.tf/options.dt) && length(options.perturbation.dx)==d
		#                 xT(:,end+1) = xT(:,end) + options.perturbation.dx*options.dt;
		#                 XT = repmat(xT(:,end),1,nbSPoint);
		#                 if options.plot #plotting options
		#                     plot_results('t',sp,x,xT);
		#                 end
		#             else
		#                 xT(:,end+1) = xT(:,end);
		#         case 'rcp' #applying robot continuous perturbation
		#             if i >= round(options.perturbation.t0/options.dt)+1 && i <= round(options.perturbation.tf/options.dt) && length(options.perturbation.dx)==d
		#                 x(:,i+1,:) = x(:,i+1,:) + repmat(options.perturbation.dx,[1 1 nbSPoint])*options.dt;

		# # plotting the result
		# if options.plot:
		# 	plot_results('u',sp,x,xT);

		#Checking the convergence
		if i > 3 and (all(all(all(abs(xd[:,:-3:,:])<options.tol))) or i>options.i_max-2):
			# if options.plot:
			# 	plot_results('f',sp,x,xT)
			i += 1

			x = x[:,:-1,:]
			t = t[:-1]
			info(f'Traj Correction Iteration {i:.1f}')
			#
			# tmp=' '
			# for j in range(d):
			#     tmp= tmp += ' %1.4f ' #[tmp.T #1.4f ;'];
			#
			# tmp=tmp[1:-2];
			info(f'Final Time: {t[0, -1, 0]:1.2f}')
			info(f'Final Point: {np.squeeze(x[:,-1,:])}')
			info(f'Target Position: {xT[:,-1]}')
			info(f'########################################################')

			if i>options.i_max-2:
				info(f'Simulation stopped since it reaches the maximum number of allowed iterations {i}')
				info(f'Exiting without convergence!!! Increase the parameter ''options.i_max'' to handle this error.')
			break
		i += 1

	return x, xd#, #t, xT, x_obs

def  check_options(options=None):
	if not options:
		options = Bundle({})
	if not isfield(options,'dt'): # integration time step
		options.dt = 0.02
	if not isfield(options,'winsize'): # integration time step
		options.winsize = (12, 7)
	if not isfield(options,'i_max'): # maximum number of iterations
		options.i_max = 1000
	if not isfield(options,'tol'): # convergence tolerance
		options.tol = 0.001
	if not isfield(options,'plot'): # shall simulator plot the figure
		options.plot = True
	else:
		options.plot = options.plot > 0
	# if not isfield(options,'perturbation'): # shall simulator plot the figure
	# 	options.perturbation.type = '';
	# else:
	# 	if not isfield(options.perturbation,'type') or not isfield(options.perturbation,'t0') or not isfield(options.perturbation,'dx') or \
	# 		((strcmp(options.perturbation.type,'rcp') or strcmp(options.perturbation.type,'tcp')) and not isfield(options.perturbation,'tf')) or \
	# 		(not strcmp(options.perturbation.type,'rcp') and not strcmp(options.perturbation.type,'tcp') and not strcmp(options.perturbation.type,'rdp') \
	# 		and not strcmp(options.perturbation.type,'tdp')):
	#
	# 		info('Invalid perturbation structure. The perturbation input is ignored!')
	# 		options.perturbation.type = ''

	return options


# def plot_results(mode,sp,x,xT,kwargs):
# 	if not kwargs or not kwargs[0]:
# 		b_obs = False;
# 	else:
# 		b_obs = True;
# 		obs = kwargs[0]
#
# 	d, _, nbSPoint = size(x);
# 	winsize =(16, 9) if not isfield('winsize', kwargs) else kwargs.winsize
# 	fontdict = {'fontsize':16, 'fontweight':'bold'}
#
# 	if strcmp(mode, 'i'):
# 		if d==2:
# 			sp.fig = plt.figure(figsize=winsize)#'name','2D Simulation of the task','position',[653 550 560 420]);
# 			sp.axis = f.gca()
# 			sp.x = [np.nan for i in range(nbSPoint)]
# 			sp.xT = sp.axis.plot(xT[0],xT[1],'k*', markersize=10, linewidth=1.5);
# 			sp.xT_l = sp.axis.plot(xT[0],xT[1],'k--', linewidth=1.5);
#
# 			for j in range(nbSPoint):
# 				sp.axis.plot(x[0, 0, j], x[1, 0,j], c='ko',markersize=2,linewidth=7.5)
# 				sp.x[j] = sp.axis.plot(x[0, 0, j], x[1, 0,j]);
#
# 			ax.set_xlabel('xi_1', fontdict=fontdict)
# 			ax.set_ylabel('xi_2', fontdict=fontdict)
# 			ax.grid('on')
#
# 			# if b_obs:
# 			#     [x_obs x_obs_sf] = obs_draw_ellipsoid(obs,40);
# 			#     for n=1:size(x_obs,3)
# 			#         sp.obs(n) = patch(x_obs(1,:,n),x_obs(2,:,n),0.1*ones(1,size(x_obs,2)),[0.6 1 0.6]);
# 			#         sp.obs_sf(n) = plot(x_obs_sf(1,:,n),x_obs_sf(2,:,n),'k--','linewidth',0.5);
#
# 		# elif d==3:
# 		#         sp.fig = figure('name','3D Simulation of the task','position',[653 550 560 420]);
# 		#         sp.axis = gca;
# 		#         hold on
# 		#         sp.xT = plot3(xT(1),xT(2),xT(3),'k*','EraseMode','none','markersize',10,'linewidth',1.5);
# 		#         sp.xT_l = plot3(xT(1),xT(2),xT(3),'k--','EraseMode','none','linewidth',1.5);
# 		#         for j=1:nbSPoint
# 		#             plot3(x(1,1,j),x(2,1,j),x(3,1,j),'ok','markersize',2,'linewidth',7.5)
# 		#             sp.x(j)= plot3(x(1,1,j),x(2,1,j),x(3,1,j),'EraseMode','none');
# 		#
# 		#         # if b_obs
# 		#         #     n_theta = 15;
# 		#         #     n_phi = 10;
# 		#         #     x_obs = obs_draw_ellipsoid(obs,[n_theta n_phi]);
# 		#         #     for n=1:size(x_obs,3)
# 		#         #         sp.obs(n) = surf(reshape(x_obs(1,:,n),n_phi,n_theta), reshape(x_obs(2,:,n),n_phi,n_theta), reshape(x_obs(3,:,n),n_phi,n_theta));
# 		#         #         set(sp.obs(n),'FaceColor',[0.6 1 0.6],'linewidth',0.1)
# 		#
# 		#         ax.set_xlabel('xi_1', fontdict=fontdict)
# 		# 		ax.set_ylabel('xi_2', fontdict=fontdict)
# 		# 		ax.set_zlabel('xi_3', fontdict=fontdict)
# 		# 		ax.grid('on')
# 		# else
# 		#         sp.fig = figure('name','Simulation of the task','position',[542   146   513   807]);
# 		#         for i=2:d
# 		#             sp.axis(i-1)=subplot(d-1,1,i-1);
# 		#             hold on
# 		#             sp.xT(i-1) = plot(xT(1),xT(i),'k*','EraseMode','none','markersize',10,'linewidth',1.5);
# 		#             sp.xT_l(i-1) = plot(xT(1),xT(i),'k--','EraseMode','none','linewidth',1.5);
# 		#             for j=1:nbSPoint
# 		#                 plot(x(1,1,j),x(i,1,j),'ok','markersize',2,'linewidth',7.5);
# 		#                 sp.x(i-1,j)= plot(x(1,1,j),x(i,1,j),'EraseMode','none');
# 		#             end
# 		#             ylabel(['$\xi_' num2str(i) '$'],'interpreter','latex','fontsize',12);
# 		#             grid on
# 		#             if i==d
# 		#                 xlabel(['$\xi_' num2str(1) '$'],'interpreter','latex','fontsize',12);
# 		#             end
# 		#             grid on;box on
# 		#         end
# 		#     end
#
# 		elif strcmp(mode, 'u'): #case 'u' #updating the figure
# 			if d==2:
# 				sp.fig = plt.figure(figsize=winsize)#'name','2D Simulation of the task','position',[653 550 560 420]);
# 				ax = f.gca()
# 				# for j in range(nbSPoint):
# 				#     set(sp.x(j),'XData', x(1,:,j),'YData',x(2,:,j))
#
# 				if max(x[0,-1,:])>ax.xlim(2) or min(x[0,-1,:])<ax.xlim(1) or \
# 					max(x[1,-1,:])>ax.ylim(2) or min(x(2,end,:))<ax.ylim(1):
# 					ax(sp.axis,'tight');
# 					ax=get(sp.axis);
# 					axis(sp.axis,...
# 						 [ax.xlim(1)-(ax.xlim(2)-ax.xlim(1))/10 ax.xlim(2)+(ax.xlim(2)-ax.xlim(1))/10 ...
# 						  ax.ylim(1)-(ax.ylim(2)-ax.ylim(1))/10 ax.ylim(2)+(ax.ylim(2)-ax.ylim(1))/10]);
# 				end
# 			# elseif d==3
# 			#     ax=get(sp.axis);
# 			#     for j=1:nbSPoint
# 			#         set(sp.x(j),'XData',x(1,:,j),'YData',x(2,:,j),'ZData',x(3,:,j))
# 			#     end
# 			#
# 			#     if max(x(1,end,:))>ax.xlim(2) or min(x(1,end,:))<ax.xlim(1) or max(x(2,end,:))>ax.ylim(2) or min(x(2,end,:))<ax.ylim(1) or max(x(3,end,:))>ax.zlim(2) or min(x(3,end,:))<ax.zlim(1)
# 			#         axis(sp.axis,'tight');
# 			#         ax=get(sp.axis);
# 			#         axis(sp.axis,...
# 			#              [ax.xlim(1)-(ax.xlim(2)-ax.xlim(1))/10 ax.xlim(2)+(ax.xlim(2)-ax.xlim(1))/10 ...
# 			#               ax.ylim(1)-(ax.ylim(2)-ax.ylim(1))/10 ax.ylim(2)+(ax.ylim(2)-ax.ylim(1))/10 ...
# 			#               ax.zlim(1)-(ax.zlim(2)-ax.zlim(1))/10 ax.zlim(2)+(ax.zlim(2)-ax.zlim(1))/10]);
# 			#     end
# 			# else
# 			#     for i=1:d-1
# 			#         ax=get(sp.axis(i));
# 			#         for j=1:nbSPoint
# 			#             set(sp.x(i,j),'XData',x(1,:,j),'YData',x(i+1,:,j))
# 			#         end
# 			#
# 			#         if max(x(1,end,:))>ax.xlim(2) or min(x(1,end,:))<ax.xlim(1) or max(x(i+1,end,:))>ax.ylim(2) or min(x(i+1,end,:))<ax.ylim(1)
# 			#             axis(sp.axis(i),'tight');
# 			#             ax=get(sp.axis(i));
# 			#             axis(sp.axis(i),...
# 			#                 [ax.xlim(1)-(ax.xlim(2)-ax.xlim(1))/10 ax.xlim(2)+(ax.xlim(2)-ax.xlim(1))/10 ...
# 			#                 ax.ylim(1)-(ax.ylim(2)-ax.ylim(1))/10 ax.ylim(2)+(ax.ylim(2)-ax.ylim(1))/10]);
#
#
# 		case 't' #updating the figure
# 			if gcf ~= sp.fig
# 				figure(sp.fig)
# 			end
# 			if d==2
# 				ax=get(sp.axis);
# 				set(sp.xT,'XData',xT(1,end),'YData',xT(2,end))
# 				set(sp.xT_l,'XData',xT(1,:),'YData',xT(2,:))
#
# 				if max(xT(1,end))>ax.xlim(2) or min(xT(1,end))<ax.xlim(1) or max(xT(2,end))>ax.ylim(2) or min(xT(2,end))<ax.ylim(1)
# 					axis(sp.axis,'tight');
# 					ax=get(sp.axis);
# 					axis(sp.axis,...
# 						 [ax.xlim(1)-(ax.xlim(2)-ax.xlim(1))/10 ax.xlim(2)+(ax.xlim(2)-ax.xlim(1))/10 ...
# 						  ax.ylim(1)-(ax.ylim(2)-ax.ylim(1))/10 ax.ylim(2)+(ax.ylim(2)-ax.ylim(1))/10]);
# 				end
# 			# elseif d==3
# 			#     ax=get(sp.axis);
# 			#     set(sp.xT,'XData',xT(1,end),'YData',xT(2,end),'ZData',xT(3,end))
# 			#     set(sp.xT_l,'XData',xT(1,:),'YData',xT(2,:),'ZData',xT(3,:))
# 			#
# 			#     if max(xT(1,end))>ax.xlim(2) or min(xT(1,end))<ax.xlim(1) or max(xT(2,end))>ax.ylim(2) or min(xT(2,end))<ax.ylim(1) or max(xT(3,end))>ax.zlim(2) or min(xT(3,end))<ax.zlim(1)
# 			#         axis(sp.axis,'tight');
# 			#         ax=get(sp.axis);
# 			#         axis(sp.axis,...
# 			#              [ax.xlim(1)-(ax.xlim(2)-ax.xlim(1))/10 ax.xlim(2)+(ax.xlim(2)-ax.xlim(1))/10 ...
# 			#               ax.ylim(1)-(ax.ylim(2)-ax.ylim(1))/10 ax.ylim(2)+(ax.ylim(2)-ax.ylim(1))/10 ...
# 			#               ax.zlim(1)-(ax.zlim(2)-ax.zlim(1))/10 ax.zlim(2)+(ax.zlim(2)-ax.zlim(1))/10]);
# 			#     end
# 			# else
# 			#     for i=1:d-1
# 			#         ax=get(sp.axis(i));
# 			#         set(sp.xT(i),'XData',xT(1,end),'YData',xT(i+1,end))
# 			#         set(sp.xT_l(i),'XData',xT(1,:),'YData',xT(i+1,:))
# 			#
# 			#         if max(xT(1,end))>ax.xlim(2) or min(xT(1,end))<ax.xlim(1) or max(xT(i+1,end))>ax.ylim(2) or min(xT(i+1,end))<ax.ylim(1)
# 			#             axis(sp.axis(i),'tight');
# 			#             ax=get(sp.axis(i));
# 			#             axis(sp.axis(i),...
# 			#                 [ax.xlim(1)-(ax.xlim(2)-ax.xlim(1))/10 ax.xlim(2)+(ax.xlim(2)-ax.xlim(1))/10 ...
# 			#                 ax.ylim(1)-(ax.ylim(2)-ax.ylim(1))/10 ax.ylim(2)+(ax.ylim(2)-ax.ylim(1))/10]);
#
# 		# case 'o' #updating the obstacle position
# 		#     if gcf ~= sp.fig
# 		#         figure(sp.fig)
# 		#     end
# 		#     if b_obs
# 		#         n = kwargs{1};
# 		#         dx = kwargs{2};
# 		#         if d==2
# 		#             set(sp.obs(n),'XData',get(sp.obs(n),'XData')+ dx(1))
# 		#             set(sp.obs(n),'YData',get(sp.obs(n),'YData')+ dx(2))
# 		#
# 		#             set(sp.obs_sf(n),'XData',get(sp.obs_sf(n),'XData')+ dx(1))
# 		#             set(sp.obs_sf(n),'YData',get(sp.obs_sf(n),'YData')+ dx(2))
# 		#         elseif d==3
# 		#             set(sp.obs(n),'XData',get(sp.obs(n),'XData')+ dx(1))
# 		#             set(sp.obs(n),'YData',get(sp.obs(n),'YData')+ dx(2))
# 		#             set(sp.obs(n),'ZData',get(sp.obs(n),'ZData')+ dx(2))
#
# 		case 'f' # final alighnment of axis
# 			if gcf ~= sp.fig
# 				figure(sp.fig)
# 			end
# 			if d==2
# 				axis(sp.axis,'tight');
# 				ax=get(sp.axis);
# 				axis(sp.axis,...
# 					[ax.xlim(1)-(ax.xlim(2)-ax.xlim(1))/10 ax.xlim(2)+(ax.xlim(2)-ax.xlim(1))/10 ...
# 					 ax.ylim(1)-(ax.ylim(2)-ax.ylim(1))/10 ax.ylim(2)+(ax.ylim(2)-ax.ylim(1))/10]);
# 			elseif d==3
# 				axis(sp.axis,'tight');
# 				ax=get(sp.axis);
# 				axis(sp.axis,...
# 					[ax.xlim(1)-(ax.xlim(2)-ax.xlim(1))/10 ax.xlim(2)+(ax.xlim(2)-ax.xlim(1))/10 ...
# 					 ax.ylim(1)-(ax.ylim(2)-ax.ylim(1))/10 ax.ylim(2)+(ax.ylim(2)-ax.ylim(1))/10 ...
# 					 ax.zlim(1)-(ax.zlim(2)-ax.zlim(1))/10 ax.zlim(2)+(ax.zlim(2)-ax.zlim(1))/10]);
# 			else
# 				for i=1:d-1
# 					axis(sp.axis(i),'tight');
# 					ax=get(sp.axis(i));
# 					axis(sp.axis(i),...
# 						[ax.xlim(1)-(ax.xlim(2)-ax.xlim(1))/10 ax.xlim(2)+(ax.xlim(2)-ax.xlim(1))/10 ...
# 						 ax.ylim(1)-(ax.ylim(2)-ax.ylim(1))/10 ax.ylim(2)+(ax.ylim(2)-ax.ylim(1))/10]);
# 	return sp
