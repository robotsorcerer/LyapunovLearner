__author__ 		= "Olalekan Ogunmolu"
__copyright__ 	= "2018, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thomson (MIT), Jethro Tan (PFN)"
__license__ 	= "MIT"
__maintainer__ 	= "Olalekan Ogunmolu"
__email__ 		= "patlekano@gmail.com"
__status__ 		= "Testing"

import numpy as np

Vxf0 = {
    'L': 4,
    'd': 2,
    'w': 1e-4, #A positive scalar weight regulating the priority between the two objectives of the opitmization. Please refer to the page 7 of the paper for further information.
    'Mu': np.array(()),
    'P': np.array(()),
    'SOS': False
}

options = {
    'tol_mat_bias': 1e-1,
    'disp': True,
    'use_6d': True,
    'num_clusters': 10, # number of gmm clusters
    'tol_stopping': 1e-10,
    'max_iter': 500,
    'int_lyap_random': True,
    'optimizePriors': True,
    'upperBoundEigenValue': True,
}

opt_exec = {
'dt': 0.1,
'i_max': 4000,
'tol': 1,
'stop_tol': 1e-4,
}


hyperparams = {
    'use_cvxopt': True, #whether to use cvxopt package, fmincon or otherwise
}
