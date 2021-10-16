__author__ 		= "Lekan Molu"
__copyright__ 	= "2018, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thomson (MIT), Jethro Tan (PFN), Pérez-Dattari, Rodrigo (TU Delft)"
__license__ 	= "MIT"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import scipy.io as sio

def load_saved_mat_file(file_path, **kwargs):
    matFile = sio.loadmat(file_path)

    data = matFile['Data']
    demoIdx = matFile['demoIndices']

    if ('Priors_EM' or 'Mu_EM' or 'Sigma_EM') in kwargs:
        Priors_EM, Mu_EM, Sigma_EM = matFile['Priors_EM'], matFile['Mu_EM'], matFile['Sigma_EM']
        return data, demoIdx, Priors_EM, Mu_EM, Sigma_EM
    else:
        return data, demoIdx
