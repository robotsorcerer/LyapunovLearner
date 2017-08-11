import numpy as np
from .compute_energy import computeEnergy
import matplotlib.pyplot as plt
from matplotlib import mlab, cm

def energyContour(Vxf,D,*args):
    """
     Syntax:

           h = EnergyContour(Vxf,D,*args)

     This function computes the energy value at a point x, given the energy
     (Lyapunov) function Vxf. When xd is passed as an empty variable, it also
     provides the energy gradient (i.e. Vdot = dV). Otherwise, it computes the
     rate of change in energy by moving along xd.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%         Copyright (c) 2014 Mohammad Khansari, LASA Lab, EPFL,       %%%
    %%          CH-1015 Lausanne, Switzerland, http://lasa.epfl.ch         %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     The program is free for non-commercial academic use. Please contact the
     author if you are interested in using the software for commercial purposes.
     The software must not be modified or distributed without prior permission
     of the authors. Please acknowledge the authors in any academic publications
     that have made use of this code or part of it. Please use this BibTex
     reference:

     S.M. Khansari-Zadeh and A. Billard (2014), "Learning Control Lyapunov Function
     to Ensure Stability of Dynamical System-based Robot Reaching Motions."
     Robotics and Autonomous Systems, vol. 62, num 6, p. 752-765.

     To get latest update of the software please visit
                              http://cs.stanford.edu/people/khansari/

     Please send your feedbacks or questions to:
                              khansari_at_cs.stanford.edu

    Ported to python by Lekan Ogunmolu
                        August 11, 2017
    """

    quality='low'
    b_plot_stream = False
    b_plot_color = False
    b_plot_contour = True
    sp = np.array(())
    countour_levels = np.array(())

    if args:
        quality, b_plot_stream, b_plot_stream, sp,
            countour_levels, b_plot_color = (args[i] for _ in range(5))

    if quality =='high':
        nx, ny = 600, 600
    elif quality == 'medium':
        nx, ny = 400, 400
    else:
        nx, ny = 200, 200

    x = np.arange(D[0], D[1], nx)
    y = np.arange(D[2], D[3], ny)
    X, Y = np.meshgrid(x, y)
    x = np.array(np.r_(
                       np.ravel(X), np.ravel(Y)].T
                 ) )

    V, dV = computeEnergy(x,np.array(()),Vxf, nargout = 2)

    if not countour_levels.size:
        countour_levels = np.linspace(0,np.log(np.max(V)),40)
        countour_levels = np.exp(countour_levels)
        if np.max(V)>40:
            countour_levels = np.round(countour_levels)

    V = V.reshape(ny,nx);
    if not sp.size:
        figure
        sp = gca

    if b_plot_color:
        pcolor(X,Y,V)
        shading interp

        colormap pink
        ca = caxis;
        ca(1) = 0;
        caxis(ca)

    if b_plot_contour:
    #     [~,h] = contour(sp,X,Y,log(V),countour_levels);
        [~,h] = contour(sp,X,Y,V,countour_levels);
        set(h,'ShowText','on','color','k','labelspacing',200);%,'fill','on'
    #     set(h,'ShowText','on','TextStep',get(h,'LevelStep')*2,'color','g')
    end

    if b_plot_stream:
        h_S = streamslice(sp,X,Y,reshape(-dV(1,:),ny,nx),reshape(-dV(2,:),ny,nx),0.5,'method','cubic');
        set(h_S,'color','m');
    end
    plot(sp,0,0,'k*','markersize',12,'linewidth',2)
    axis(sp,'equal');axis(sp,'tight');box(sp,'on')
    # grid on
    return h
