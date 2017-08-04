#! /usr/bin/env/python3
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
import math
import numpy as np
import argparse
import scipy.io as sio
import scipy.linalg as spla

from config import Vxf0, Options

# path imports
sys.path.insert(0, "CLFM_lib")
sys.path.insert(1, "GMR_lib")

def loadSavedMatFile(x):
    matFile = sio.loadmat(x)

    # print(matFile)
    data = matFile['Data']
    demoIdx = matFile['demoIndices']

    return data, demoIdx

def guess_init(data, Vxf0):
    """
    This function guesses the initial lyapunov function
    """
    b_initRandom = False

    if b_initRandom:
        temp = np.transpose(data[0:Vxf0['d'],:])
        tempvar = np.var(temp, axis=0)
        lengthScale = np.sqrt(tempvar)
        '''
         If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
        '''
        tempcov = np.cov(temp, rowvar=False)
        lengthScaleMatrix = spla.sqrtm(tempcov)
        Vxf0['Priors'] = np.random.rand(Vxf0['L']+1,1);

        # allocate spaces for incoming arrays
        Vxf0['Mu']  =  np.empty([ Vxf0['d'], Vxf0['d'], Vxf0['L'] ])
        Vxf0['P']   =  np.empty([ Vxf0['d'], Vxf0['d'], Vxf0['L'] ])

        for l in range(Vxf0['L']+1):
            tempMat = np.random.randn(Vxf0['d'], Vxf0['d'])
            Vxf0['Mu'][:,l] = np.random.randn(Vxf0['d'],1) * lengthScale
            Vxf0['P'][:,:,l] = lengthScaleMatrix * (tempMat * np.transpose(tempMat)) * lengthScaleMatrix
    else:
        Vxf0['Priors'] = np.ones((Vxf0['L']+1, 1))
        Vxf0['Priors'] = Vxf0['Priors']/sum(Vxf0['Priors'])
        Vxf0['Mu'] = np.zeros((Vxf0['d'],Vxf0['L']+1))
        for l in range(Vxf0['L']+1):
            Vxf0['P'][:,:,l] = np.eye((Vxf0['d']))

    return Vxf0

def main(Vxf0):
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelNumber', type=int, default=2, help="'can be 1 or 2'")
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    modelNames = ['w.mat', 'Sshape.mat']   # Two example models provided by Khansari
    modelNumber = args.modelNumber  # could be zero or one depending on the experiment the user is running

    data, demoIdx = loadSavedMatFile('ExampleModels/' + modelNames[modelNumber])

    Vxf0['L'] = modelNumber
    Vxf0['d'] = int(data.shape[0]/2)

    Vxf0 = guess_init(data, Vxf0)

    if args.verbose:
        print('demoIdx: ', demoIdx)
        for k, v in Vxf0.items():
            print(k, v)

if __name__ == '__main__':
    main(Vxf0)
