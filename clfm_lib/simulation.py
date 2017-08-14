import numpy as np
from .learn_energy import matlength
import matplotlib as mpl
import matplotlib.pyplot as plt

def Simulation(x0,xT,fn_handle,*args):
    """
     This function simulates motion that were learnt using SEDS, which defines
     a motins as a nonlinear time-independent asymptotically stable dynamical
     systems:
                                   xd=f(x)

     where x is an arbitrary d dimensional variable, and xd is its first time
     derivative.

     The function can be called using:
           x, xd, t = Simulation(x0,xT,Priors,Mu,Sigma)

     or
           [x xd t]=Simulation(x0,xT,Priors,Mu,Sigma,options)

     to also send a structure of desired options.

     Inputs -----------------------------------------------------------------
       o x:       d x N matrix vector representing N different starting point(s)
       o xT:      d x 1 Column vector representing the target point
       o fn_handle:  A lambda function that only gets as input a d x N matrix,
                     and returns the output matrix of the same dimension. Note
                     that the output variable is the first time derivative of
                     the input variable.

       o options: A structure to set the optional parameters of the simulator.
                  The following parameters can be set in the options:
           - .dt:      integration time step [default: 0.02]
           - .i_max:   maximum number of iteration for simulator [default: i_max=1000]
           - .plot     setting simulation graphic on (true) or off (false) [default: true]
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


        Copyright (c) 2010 S. Mohammad Khansari-Zadeh, LASA Lab, EPFL,
              CH-1015 Lausanne, Switzerland, http://lasa.epfl.ch


     The program is free for non-commercial academic use. Please contact the
     author if you are interested in using the software for commercial purposes.
     The software must not be modified or distributed without prior permission
     of the authors. Please acknowledge the authors in any academic publications
     that have made use of this code or part of it. Please use this BibTex
     reference:

     S. M. Khansari-Zadeh and A. Billard, "Learning Stable Non-Linear Dynamical
     Systems with Gaussian Mixture Models", IEEE Transaction on Robotics, 2011.

     To get latest upadate of the software please visit
                              http://lasa.epfl.ch/khansari

     Please send your feedbacks or questions to:
                               mohammad.khansari_at_epfl.ch

    Ported to python by Lekan Ogunmolu
                        patlekano@gmail.com
                        August 13, 2017
    """
    # parsing inputs
    if not args:
        options = check_options()
    else:
        options = check_options(args[0])    # Checking the given options, and add ones are not defined.

    d= x0.shape[0] #dimension of the model
    if not xT:
        xT = np.zeros((d,1))

    if d != xT.shape[0]:
        print('Error: length(x0) should be equal to length(xT)!')
        x =  np.array(()); xd= np.array(()); t= np.array(());
        return

    ## setting initial values
    nbSPoint=x0.shape[1] #number of starting points. This enables to simulatneously run several trajectories

    if ('obstacle' in options and not options.['obstacle']): #there is obstacle
        obs_bool = True
        obs = options['obstacle']
        for n in range(matlength(obs)):
            x_obs[n] = obs[n].x0
            if not 'extra' in obs[n]:
                obs[n]['extra']['ind'] = 2
                obs[n]['extra']['C_Amp'] = 0.01
                obs[n]['extra']['R_Amp'] = 0.0
            else:
                if not 'ind' in obs[n]['extra']:
                    obs[n]['extra']['ind'] = 2
                if not 'C_Amp' in obs[n]['extra']:
                    obs[n]['extra']['C_Amp'] = 0.01
                if not 'R_Amp' in obs[n]['extra']:
                    obs[n]['extra']['R_Amp'] = 0.0

        b_contour = np.zeros((1,nbSPoint))
    else:
        obs_bool = False
        obs = np.array(())
        x_obs = np.nan

    #initialization
    for i in range(nbSPoint):
        x[:,0,i] = x0[:,i]
    xd = np.zeros(x.shape)
    if xT.shape == x0.shape:
        XT = xT
    else:
        XT = np.tile(xT, [1,nbSPoint])   #a matrix of target location (just to simplify computation)

    t=0; #starting time

    if options['plot'] #plotting options
        sp = plot_results('i',np.array(()),x,xT,obs);

    ## Simulation
    i=1;
    while True:
        #Finding xd using fn_handle.
        xd[:,i,:]=fn_handle(np.squeeze(x(:,i,:))-XT).reshape([d, 1, nbSPoint])

        # This part if for the obstacle avoidance module
        if obs_bool:
            # applying perturbation on the obstacles
            for n in range(matlength(obs)):
                if 'perturbation' in obs[n]:
                    if i >= np.round(obs[n]['perturbation']['t0']/options['dt'])+1 \
                            and i <= np.round(obs[n]['perturbation']['tf']/options['dt']) \
                            and matlength(obs[n]['perturbation']['dx'])==d:
                        x_obs[n][:,-1] = x_obs[n][:,-1] + obs[n]['perturbation']['dx']*options['dt']
                        obs[n]['x0'] = x_obs[n][:,-1]
                        xd_obs = obs[n]['perturbation']['dx']
                        if options['plot']: #plotting options
                            plot_results('o',sp,x,xT,n,obs[n]['perturbation']['dx']*options['dt']);\
                    else:
                        x_obs[n][:,-1] = x_obs[n][:,-1]
                        xd_obs = 0
                else:
                    xd_obs = 0

            for j in range(nbSPoint):
                xd[:,i,j] b_contour[j] = obs_modulation_ellipsoid(x[:,i,j], \
                                            xd[:,i,j],obs,b_contour[j],xd_obs)

        # Integration
        x[:,i+1,:]=x[:,i,:]+xd[:,i,:]*options['dt']
        t[i+1]=t[i]+options['dt']

        # Applying perturbation if any
        if options['perturbation']['type']=='tdp':
            #case 'tdp' %applying target discrete perturbation
            if (i == np.round(options['perturbation']['t0']/options['dt'])+1 \
                    and matlength(options['perturbation']['dx']))==d:
                xT[:,-1] = xT[:,-1] + options['perturbation']['dx']
                XT = np.tile(xT[:,-1], [1,nbSPoint])
                if options['plot']: #plotting options
                    plot_results('t',sp,x,xT)
            else:
                xT[:,-1] = xT[:,-1]
        elif options['perturbation']['type']== 'rdp': #applying robot discrete 'perturbation']
            if (i == np.round(options['perturbation']['t0']/options['dt'])+1 \
                    and matlength(options['perturbation']['dx']) ) ==d:
                x[:,i+1,:] = x[:,i+1,:] + np.tile(options['perturbation']['dx'],[1, 1, nbSPoint])

        elif options['perturbation']['dx']== 'tcp' #applying target continuous perturbation
            if (i >= np.round(options['perturbation']['t0']/options['dt'])+1 \
                and i <= np.round(options['perturbation']['tf']/options['dt']) \
                and matlength(options['perturbation']['dx']) )==d:
                xT[:,-1] += options['perturbation']['dx']*options['dt']
                XT = np.tile(xT(:,end), [1,nbSPoint])
                if options['plot']: #plotting options
                    plot_results('t',sp,x,xT)
            # else:
            #     xT(:,end+1) = xT(:,end);
            # end
        elif options['perturbation']['dx']== 'rcp': #applying robot continuous perturbation
            if (i >= np.round(options['perturbation']['t0']/options['dt'])+1 \
                and i <= np.round(options['perturbation']['tf']/options['dt']) \
                and matlength(options['perturbation']['dx']) ) ==d:
                x(:,i+1,:) += np.tile(options['perturbation']['dx'],[1, 1, nbSPoint])*options['dt']

        # plotting the result
        if options['plot']
            plot_results('u',sp,x,xT)

        #Checking the convergence
        if i > 3 and (np.all(np.all(np.all(np.abs(xd[:,-3:-1,:]) \
                                           < options['tol']))) \
                                           or i>options['i_max']-2):
            if options['plot']:
                plot_results('f',sp,x,xT)

            i += 1
    #         xd(:,i,:)=reshape(fn_handle(squeeze(x(:,i,:))-XT),[d 1 nbSPoint]);
            x[:,-1,:] = np.array([])
            t[-1] = np.array(())
            print('Number of Iterations: \n',i)
            tmp=''
            for j in range(d):
                tmp=np.r_[tmp, ' #1.4f ;']
            tmp=tmp[2:-2]
            print('Final Time: #1.2f (sec)\n',t[1,-1,1])
            print('Final Point: [' tmp ']\n'],np.squeeze(x[:,-1,:]))
            print('Target Position: [' tmp ']\n',xT[:,-1])
            print('## #####################################################\n\n\n')

            if i>options['i_max']-2:
                print('Simulation stopped since it reaches the maximum number of allowed iterations i_max = #1.0f\n',i)
                print('Exiting without convergence!!! Increase the parameter ''options.i_max'' to handle this error.\n')

            break
        i += 1

    return x, xd, t, xT, x_obs

def check_options(args):
    if not args:
        options = args[0]
    else:
        options['dt']=0.02 #to create the variabl

    if not 'dt' in options: #% integration time step
        options['dt'] = 0.02

    if not 'i_max' in options:  # maximum number of iterations
        options['i_max'] = 1000

    if not 'tol' in options: # convergence tolerance
        options['tol'] = 0.001

    if not 'plot' in options: #shall simulator plot the figure
        options['plot'] = 1
    else:
        options['plot'] = options['plot'] > 0

    if not 'perturbation' in options: # shall simulator plot the figure
        options['perturbation].['type'] = ''
    else:
        if  (
            (not 'type' in options['perturbation']) or \
            (not 't0' in options['perturbation']) or \
            (not 'dx' in options['perturbation']) or \
            (
            (options['perturbation']['type']=='rcp') or \
            (options['perturbation']['type']=='tcp')) and \
            (not  options['perturbation']=='tf')
            ) or \
            (not (options['perturbation']['type']=='rcp') \
            and not (options['perturbation']['type']=='tcp') \
            and not (options['perturbation']['type']=='rdp') \
            and not (options['perturbation']['type']=='tdp'))
            )

            print('Invalid perturbation structure. The perturbation input is ignored!')
            options['perturbation']['type'] = ''
return options

def plot_results(mode,sp,x,xT,args):
    if not args or not args[0]:
        b_obs = False
    else:
        b_obs = True
        obs = args[0]
    d, _, nbSPoint = x.shape
    if mode=='i':
        if d==2:
            sp['fig'] = plt.figure(num=0) #,'2D Simulation of the task',figsize=(653 550 560 420)
            plt.title(r'2D Simulation of the task')
            sp['axis'] = plt.gca()
            hold(True)
            sp['xT'] = plt.plot(xT[0],xT[1],'k*', markersize=10, linewidth=1.5)
            sp['xT_l'] = plt.plot(xT[0],xT[1],'k--', linewidth=1.5)
            for j in range(nbSPoint):
                plot(x[0,0,j],x[1,0,j],'ok',markersize=2,linewidth=7.5)
                sp['x['str(j)']']= plot(x[0,0,j],x[1,0,j])
            plt.xlabel(r'$\xi_1$',interpreter=latex,fontsize=16)
            plt.ylabel(r'$\xi_2$',interpreter=latex,fontsize=16)
            grid on;box on

            if b_obs:
                x_obs, x_obs_sf = obs_draw_ellipsoid(obs,40)
                for n in range(x_obs.shape[3]):
                    sp['obs'][n] = patch(x_obs[0,:,n],x_obs[1,:,n],0.1*np.ones((1,x_obs.shape[2])),[0.6 1 0.6])
                    sp['obs_sf'][n] = plot(x_obs_sf[0,:,n],x_obs_sf[1,:,n],'k--',linewidth=0.5)
        elif d==3:
            sp['fig'] = figure('name','3D Simulation of the task','position',[653 550 560 420]);
            sp['axis'] = gca;
            hold on
            sp['xT'] = plot3(xT(1),xT(2),xT(3),'k*','EraseMode','none','markersize',10,'linewidth',1.5);
            sp['xT_l'] = plot3(xT(1),xT(2),xT(3),'k--','EraseMode','none','linewidth',1.5);
            for j=1:nbSPoint
                plot3(x(1,1,j),x(2,1,j),x(3,1,j),'ok','markersize',2,'linewidth',7.5)
                sp['x'][j]= plot3(x(1,1,j),x(2,1,j),x(3,1,j),'EraseMode','none');
            end

            if b_obs
                n_theta = 15;
                n_phi = 10;
                x_obs = obs_draw_ellipsoid(obs,[n_theta n_phi]);
                for n=1:size(x_obs,3)
                    sp.obs(n) = surf(reshape(x_obs(1,:,n),n_phi,n_theta), reshape(x_obs(2,:,n),n_phi,n_theta), reshape(x_obs(3,:,n),n_phi,n_theta));
                    set(sp.obs(n),'FaceColor',[0.6 1 0.6],'linewidth',0.1)
                end
            end
            xlabel('$\xi_1$','interpreter','latex','fontsize',16);
            ylabel('$\xi_2$','interpreter','latex','fontsize',16);
            zlabel('$\xi_3$','interpreter','latex','fontsize',16);
            grid on
            view(-28,44)
        else
            sp.fig = figure('name','Simulation of the task','position',[542   146   513   807]);
            for i=2:d
                sp.axis(i-1)=subplot(d-1,1,i-1);
                hold on
                sp.xT(i-1) = plot(xT(1),xT(i),'k*','EraseMode','none','markersize',10,'linewidth',1.5);
                sp.xT_l(i-1) = plot(xT(1),xT(i),'k--','EraseMode','none','linewidth',1.5);
                for j=1:nbSPoint
                    plot(x(1,1,j),x(i,1,j),'ok','markersize',2,'linewidth',7.5);
                    sp.x(i-1,j)= plot(x(1,1,j),x(i,1,j),'EraseMode','none');
                end
                ylabel(['$\xi_' num2str(i) '$'],'interpreter','latex','fontsize',12);
                grid on
                if i==d
                    xlabel(['$\xi_' num2str(1) '$'],'interpreter','latex','fontsize',12);
                end
                grid on;box on
            end
        end

        case 'u' %updating the figure
            if gcf ~= sp.fig
                figure(sp.fig)
            end
            if d==2
                ax=get(sp.axis);
                for j=1:nbSPoint
                    set(sp.x(j),'XData',x(1,:,j),'YData',x(2,:,j))
                end

                if max(x(1,end,:))>ax.XLim(2) || min(x(1,end,:))<ax.XLim(1) || max(x(2,end,:))>ax.YLim(2) || min(x(2,end,:))<ax.YLim(1)
                    axis(sp.axis,'tight');
                    ax=get(sp.axis);
                    axis(sp.axis,...
                         [ax.XLim(1)-(ax.XLim(2)-ax.XLim(1))/10 ax.XLim(2)+(ax.XLim(2)-ax.XLim(1))/10 ...
                          ax.YLim(1)-(ax.YLim(2)-ax.YLim(1))/10 ax.YLim(2)+(ax.YLim(2)-ax.YLim(1))/10]);
                end
            elseif d==3
                ax=get(sp.axis);
                for j=1:nbSPoint
                    set(sp.x(j),'XData',x(1,:,j),'YData',x(2,:,j),'ZData',x(3,:,j))
                end

                if max(x(1,end,:))>ax.XLim(2) || min(x(1,end,:))<ax.XLim(1) || max(x(2,end,:))>ax.YLim(2) || min(x(2,end,:))<ax.YLim(1) || max(x(3,end,:))>ax.ZLim(2) || min(x(3,end,:))<ax.ZLim(1)
                    axis(sp.axis,'tight');
                    ax=get(sp.axis);
                    axis(sp.axis,...
                         [ax.XLim(1)-(ax.XLim(2)-ax.XLim(1))/10 ax.XLim(2)+(ax.XLim(2)-ax.XLim(1))/10 ...
                          ax.YLim(1)-(ax.YLim(2)-ax.YLim(1))/10 ax.YLim(2)+(ax.YLim(2)-ax.YLim(1))/10 ...
                          ax.ZLim(1)-(ax.ZLim(2)-ax.ZLim(1))/10 ax.ZLim(2)+(ax.ZLim(2)-ax.ZLim(1))/10]);
                end
            else
                for i=1:d-1
                    ax=get(sp.axis(i));
                    for j=1:nbSPoint
                        set(sp.x(i,j),'XData',x(1,:,j),'YData',x(i+1,:,j))
                    end

                    if max(x(1,end,:))>ax.XLim(2) || min(x(1,end,:))<ax.XLim(1) || max(x(i+1,end,:))>ax.YLim(2) || min(x(i+1,end,:))<ax.YLim(1)
                        axis(sp.axis(i),'tight');
                        ax=get(sp.axis(i));
                        axis(sp.axis(i),...
                            [ax.XLim(1)-(ax.XLim(2)-ax.XLim(1))/10 ax.XLim(2)+(ax.XLim(2)-ax.XLim(1))/10 ...
                            ax.YLim(1)-(ax.YLim(2)-ax.YLim(1))/10 ax.YLim(2)+(ax.YLim(2)-ax.YLim(1))/10]);
                    end
                end
            end

        case 't' %updating the figure
            if gcf ~= sp.fig
                figure(sp.fig)
            end
            if d==2
                ax=get(sp.axis);
                set(sp.xT,'XData',xT(1,end),'YData',xT(2,end))
                set(sp.xT_l,'XData',xT(1,:),'YData',xT(2,:))

                if max(xT(1,end))>ax.XLim(2) || min(xT(1,end))<ax.XLim(1) || max(xT(2,end))>ax.YLim(2) || min(xT(2,end))<ax.YLim(1)
                    axis(sp.axis,'tight');
                    ax=get(sp.axis);
                    axis(sp.axis,...
                         [ax.XLim(1)-(ax.XLim(2)-ax.XLim(1))/10 ax.XLim(2)+(ax.XLim(2)-ax.XLim(1))/10 ...
                          ax.YLim(1)-(ax.YLim(2)-ax.YLim(1))/10 ax.YLim(2)+(ax.YLim(2)-ax.YLim(1))/10]);
                end
            elseif d==3
                ax=get(sp.axis);
                set(sp.xT,'XData',xT(1,end),'YData',xT(2,end),'ZData',xT(3,end))
                set(sp.xT_l,'XData',xT(1,:),'YData',xT(2,:),'ZData',xT(3,:))

                if max(xT(1,end))>ax.XLim(2) || min(xT(1,end))<ax.XLim(1) || max(xT(2,end))>ax.YLim(2) || min(xT(2,end))<ax.YLim(1) || max(xT(3,end))>ax.ZLim(2) || min(xT(3,end))<ax.ZLim(1)
                    axis(sp.axis,'tight');
                    ax=get(sp.axis);
                    axis(sp.axis,...
                         [ax.XLim(1)-(ax.XLim(2)-ax.XLim(1))/10 ax.XLim(2)+(ax.XLim(2)-ax.XLim(1))/10 ...
                          ax.YLim(1)-(ax.YLim(2)-ax.YLim(1))/10 ax.YLim(2)+(ax.YLim(2)-ax.YLim(1))/10 ...
                          ax.ZLim(1)-(ax.ZLim(2)-ax.ZLim(1))/10 ax.ZLim(2)+(ax.ZLim(2)-ax.ZLim(1))/10]);
                end
            else
                for i=1:d-1
                    ax=get(sp.axis(i));
                    set(sp.xT(i),'XData',xT(1,end),'YData',xT(i+1,end))
                    set(sp.xT_l(i),'XData',xT(1,:),'YData',xT(i+1,:))

                    if max(xT(1,end))>ax.XLim(2) || min(xT(1,end))<ax.XLim(1) || max(xT(i+1,end))>ax.YLim(2) || min(xT(i+1,end))<ax.YLim(1)
                        axis(sp.axis(i),'tight');
                        ax=get(sp.axis(i));
                        axis(sp.axis(i),...
                            [ax.XLim(1)-(ax.XLim(2)-ax.XLim(1))/10 ax.XLim(2)+(ax.XLim(2)-ax.XLim(1))/10 ...
                            ax.YLim(1)-(ax.YLim(2)-ax.YLim(1))/10 ax.YLim(2)+(ax.YLim(2)-ax.YLim(1))/10]);
                    end
                end
            end

        case 'o' %updating the obstacle position
            if gcf ~= sp.fig
                figure(sp.fig)
            end
            if b_obs
                n = args{1};
                dx = args{2};
                if d==2
                    set(sp.obs(n),'XData',get(sp.obs(n),'XData')+ dx(1))
                    set(sp.obs(n),'YData',get(sp.obs(n),'YData')+ dx(2))

                    set(sp.obs_sf(n),'XData',get(sp.obs_sf(n),'XData')+ dx(1))
                    set(sp.obs_sf(n),'YData',get(sp.obs_sf(n),'YData')+ dx(2))
                elseif d==3
                    set(sp.obs(n),'XData',get(sp.obs(n),'XData')+ dx(1))
                    set(sp.obs(n),'YData',get(sp.obs(n),'YData')+ dx(2))
                    set(sp.obs(n),'ZData',get(sp.obs(n),'ZData')+ dx(2))
                end
            end

        case 'f' % final alighnment of axis
            if gcf ~= sp.fig
                figure(sp.fig)
            end
            if d==2
                axis(sp.axis,'tight');
                ax=get(sp.axis);
                axis(sp.axis,...
                    [ax.XLim(1)-(ax.XLim(2)-ax.XLim(1))/10 ax.XLim(2)+(ax.XLim(2)-ax.XLim(1))/10 ...
                     ax.YLim(1)-(ax.YLim(2)-ax.YLim(1))/10 ax.YLim(2)+(ax.YLim(2)-ax.YLim(1))/10]);
            elseif d==3
                axis(sp.axis,'tight');
                ax=get(sp.axis);
                axis(sp.axis,...
                    [ax.XLim(1)-(ax.XLim(2)-ax.XLim(1))/10 ax.XLim(2)+(ax.XLim(2)-ax.XLim(1))/10 ...
                     ax.YLim(1)-(ax.YLim(2)-ax.YLim(1))/10 ax.YLim(2)+(ax.YLim(2)-ax.YLim(1))/10 ...
                     ax.ZLim(1)-(ax.ZLim(2)-ax.ZLim(1))/10 ax.ZLim(2)+(ax.ZLim(2)-ax.ZLim(1))/10]);
            else
                for i=1:d-1
                    axis(sp.axis(i),'tight');
                    ax=get(sp.axis(i));
                    axis(sp.axis(i),...
                        [ax.XLim(1)-(ax.XLim(2)-ax.XLim(1))/10 ax.XLim(2)+(ax.XLim(2)-ax.XLim(1))/10 ...
                         ax.YLim(1)-(ax.YLim(2)-ax.YLim(1))/10 ax.YLim(2)+(ax.YLim(2)-ax.YLim(1))/10]);
                end
            end
    end
    drawnow
