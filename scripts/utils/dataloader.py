__author__ 		= "Lekan Molu"
__copyright__ 	= "2018, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thomson (MIT), Jethro Tan (PFN), Pérez-Dattari, Rodrigo (TU Delft)"
__license__ 	= "MIT"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import scipy.io as sio

def load_saved_mat_file(x):
    """
        Loads a matlab file from a subdirectory.

        Inputs:
            x: path to data on HDD

            
       Copyright (c) Lekan Molux. https://scriptedonachip.com
       2021.
    """

    matFile = sio.loadmat(x)

    data = matFile['Data']
    demoIdx = matFile['demoIndices']-1

    Priors_EM, Mu_EM, Sigma_EM = matFile['Priors_EM'], matFile['Mu_EM'], matFile['Sigma_EM']
    return data, demoIdx, Priors_EM, Mu_EM, Sigma_EM
