#! /usr/bin/env python3
import sys
import argparse
import numpy as np
# import scipy as sp
import scipy.io as sio
import scipy.linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use(['presentation', 'fivethirtyeight'])

# path imports
from os.path import dirname, abspath#, join, sep
lyap = dirname(dirname(abspath(__file__))) + '/' + 'LyapunovLearner'
sys.path.append(lyap)

from config import Vxf0, options
from clfm_lib.learn_energy import learnEnergy
from clfm_lib.ds_stabilizer import dsStabilizer
from clfm_lib.compute_energy import computeEnergy
from gmr_lib.gmr import GMR

"""
Source code for implementing the Learning Control Lyapunov Function paper by Khansari-Zadeh

This script illustrates how to learn an arbitrary model from a set of demonstrations

Reference:
    S.M. Khansari-Zadeh and A. Billard (2014), "Learning Control Lyapunov Function to Ensure Stability
    of Dynamical System-based Robot Reaching Motions." Robotics and Autonomous Systems, vol. 62, num 6, p. 752-765.

Ported from the bitbucket repo: https://bitbucket.org/khansari/clfdm

Written by: Olalekan Ogunmolu
Date:     August 04, 2017
"""

def loadSavedMatFile(x, **kwargs):
    matFile = sio.loadmat(x)

    # print(matFile)
    data = matFile['Data']
    demoIdx = matFile['demoIndices']

    if ('Priors_EM' or 'Mu_EM' or 'Sigma_EM') in kwargs:
        Priors_EM, Mu_EM, Sigma_EM = matFile['Priors_EM'], matFile['Mu_EM'],
        matFile['Sigma_EM']
        return data, demoIdx, Priors_EM, Mu_EM, Sigma_EM
    else:
        return data, demoIdx

def guess_init_lyap(data, Vxf0, b_initRandom=False):
    """
    This function guesses the initial lyapunov function
    """

    # allocate spaces for incoming arrays
    Vxf0['Mu']  =  np.zeros(( Vxf0['d'], Vxf0['L']+1 )) # will be 2x2
    Vxf0['P']   =  np.zeros(( Vxf0['d'], Vxf0['d'], Vxf0['L']+1)) # wil be 2x2x3

    if b_initRandom:
        temp = data[0:Vxf0['d'],:].T
        tempvar = np.var(temp, axis=0)
        lengthScale = np.sqrt(tempvar)
        lengthScale = np.ravel(lengthScale)
        '''
         If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
        '''
        tempcov = np.cov(temp, rowvar=False)
        lengthScaleMatrix = LA.sqrtm(tempcov)
        Vxf0['Priors'] = np.random.rand(Vxf0['L']+1,1)

        for l in range(Vxf0['L']+1):
            tempMat = np.random.randn(Vxf0['d'], Vxf0['d'])
            # print(np.random.randn(Vxf0['d'],1).shape, 'ls: ', lengthScale.shape, 'Vxf0[\'Mu\']: ', Vxf0['Mu'].shape)
            Vxf0['Mu'][:,l] = np.multiply(np.random.randn(Vxf0['d'],1), lengthScale)
            # print('tempMat: ', tempMat.shape, 'lSM: ', lengthScaleMatrix.shape)
            Vxf0['P'][:,:,l] = lengthScaleMatrix.dot((tempMat * tempMat.T)).dot(lengthScaleMatrix)
    else:
        Vxf0['Priors'] = np.ones((Vxf0['L']+1, 1))
        Vxf0['Priors'] = Vxf0['Priors']/np.sum(Vxf0['Priors'])
        # Vxf0['Mu'] = np.zeros((Vxf0['d'], Vxf0['d'], Vxf0['L']+1))

        # allocate Vxf0['P']
        # Vxf0['P']   =  np.zeros(( Vxf0[ 'd'], Vxf0['d'], Vxf0['L']+1)) # wil be 2x2x3
        for l in range(Vxf0['L']+1):
            Vxf0['P'][:,:,l] = np.eye((Vxf0['d']))

    return Vxf0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelNumber', type=int, default=1, help="can be 0 or 1")
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    modelNames = ['w.mat', 'Sshape.mat']   # Two example models provided by Khansari
    modelNumber = args.modelNumber  # could be zero or one depending on the experiment the user is running

    data, demoIdx = loadSavedMatFile(lyap + '/' + 'example_models/' + modelNames[modelNumber])

    global Vxf0
    Vxf0['L'] = modelNumber
    Vxf0['d'] = int(data.shape[0]/2)

    Vxf0 = guess_init_lyap(data, Vxf0)
    # for k, v in Vxf0.items():
    #     print(k, v)
    Vxf = learnEnergy(Vxf0, data, options)

    # plot the result
    fig = mpl.figure.Figure(figsize=(6.40,4.80), dpi=128, facecolor=darkslategray,
                            edgecolor=blue, linewidth=2.4, frameon=None,
                            subplotpars=None, tight_layout=True)
    fig = plt.figure()
    axes = plt.gca()
    plt.hold(True)

    h1 = plt.plot(data[0:,:], data[1,:], 'r.', label='demonstrations')
    axes.set_xlim([ax.XLim(1)-(ax.XLim(2)-ax.XLim(1))/10, ax.XLim(2)+(ax.XLim(2)-ax.XLim(1))/10])
    axes.set_ylim([ax.YLim(1)-(ax.YLim(2)-ax.YLim(1))/10, ax.YLim(2)+(ax.YLim(2)-ax.YLim(1))/10])

    h3 = energyContour(Vxf, axis, np.array(()), np.array(()), axes, np.array(()), False, label='energy levels')
    h2 = plt.plot(0,0, 'g*', markersize=15, linewidth=3, label='target')
    plt.title('Energy Levels of the learned Lyapunov Functions')
    plt.xlabel('x (mm)','fontsize',15)
    plt.ylabel('y (mm)','fontsize',15)
    h = [h1, h2, h3]
    plt.legend(handles=h,loc=3)

    # if args.verbose:
    #     print('demoIdx: ', demoIdx)
    #     for k, v in Vxf0.items():
    #         # print(k, v)
    #         pass

    # Simulation

    # A set of options that will be passed to the Simulator. Please type
    # 'doc preprocess_demos' in the MATLAB command window to get detailed
    # information about each option.
    opt_sim = dict()
    opt_sim['dt']   = 0.01
    opt_sim['i_max']    = 4000
    opt_sim['tol']  = 1
    d = Data.shape[0]/2  #dimension of data
    x0_all = Data[:d,demoIndices[:-2]]; #finding initial points of all demonstrations

    # load(['ExampleModels/' modelNames{modelNumber}],'Priors_EM','Mu_EM','Sigma_EM')
    data, demoIdx, Priors_EM, Mu_EM, Sigma_EM = loadSavedMatFile(lyap + '/' + 'example_models/' +
                                     modelNames[modelNumber], Priors_EM=None,
                                     Mu_EM=None, Sigma_EM=None)

    # rho0 and kappa0 impose minimum acceptable rate of decrease in the energy
    # function during the motion. Refer to page 8 of the paper for more information
    rho0 = 1
    kappa0 = 0.1

    inp = list(range(Vxf['d']))
    output = Vxf['d']+ np.array([0, 1]) * Vxf['d']
    gmr_handle = lambda x: gmr(Priors_EM, Mu_EM, Sigma_EM, x, inp,output)
    fn_handle = lambda x: dsStabilizer(x,gmr_handle,Vxf,rho0,kappa0)

    x, xd = Simulation(x0_all,np.array(()),fn_handle,opt_sim) #running the simulator

    for i in x.shape[2]:
        h4 = plt.plot(x[0,:,i],x[1,:,i],'b',linewidth=1.5)
        h += h4
    lg = legend(h, ['demonstrations','target','energy levels','reproductions',
                    'location','southwest','orientation','horizontal']);
    # set(lg,'position',[0.0673    0.9278    0.8768    0.0571])

if __name__ == '__main__':
    main()
