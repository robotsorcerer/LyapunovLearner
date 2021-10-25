__author__ 		= "Lekan Molu"
__copyright__ 	= "2018, One Hell of a Lyapunov Learning Solver"
__credits__  	= "Rachel Thomson (MIT), Pérez-Dattari, Rodrigo (TU Delft)"
__license__ 	= "MIT"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import numpy as np

"""
    Contains the following parameters:
    'L':  the number of asymmetric quadratic components L>=0.
    'd': the number of asymmetric quadratic components L>=0.
    'w':  A positive scalar weight regulating the priority between the two objectives of the opitmization. Please refer to the page 7 of the paper for further information.
    'Mu': The mean of the Gaussian parameterization,
    'P': A SPD Hurwitz matrix
    'SOS': Whether we are using  Sum of Squares for the optimization
"""
Vxf0 = {
    'w': 1e-4, # A positive scalar weight regulating the priority between the two objectives of the opitmization. Please refer to the page 7 of the paper for further information.
    'Mu': np.array(()),
    'P': np.array(()),
    'SOS': False
}

# A set of options that will be passed to the solver.
options = {
    'tol_mat_bias': 1e-1,
    'disp': True,
    'num_clusters': 5,  # number of gmm clusters
    'tol_stopping': 1e-10,
    'max_iter': 500,
    'int_lyap_random': False,
    'optimizePriors': True,
    'upperBoundEigenValue': True,
}

"""
    Stabilization Options: After estimating the trajos, we now must
    stabilize those trajos at each time step.
"""
stab_options = {
    'traj_nums': 10000, # Why did KZ choose this?
    'dt': 0.01,
    'tol': 1,
    'plot': False
    }

# These for the Torobo Robot
opt_exec = {
'dt': 0.1,
'i_max': 10000,
'tol': 1,
'stop_tol': 1e-4,
}
