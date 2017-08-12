#! /usr/bin/env python3
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

import sys
import argparse
import numpy as np
# import scipy as sp
import scipy.io as sio
import scipy.linalg as LA

# path imports
from os.path import dirname, abspath#, join, sep
lyap = dirname(dirname(abspath(__file__))) + '/' + 'LyapunovLearner'
sys.path.append(lyap)

from config import Vxf0, options
from clfm_lib.learn_energy import learnEnergy
from clfm_lib.compute_energy import computeEnergy

def loadSavedMatFile(x):
    matFile = sio.loadmat(x)

    # print(matFile)
    data = matFile['Data']
    demoIdx = matFile['demoIndices']

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

    if args.verbose:
        print('demoIdx: ', demoIdx)
        for k, v in Vxf0.items():
            # print(k, v)
            pass

if __name__ == '__main__':
    main()
