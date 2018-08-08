from __future__ import print_function

__author__ 		= "Olalekan Ogunmolu"
__copyright__ 	= "2018, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thomson (MIT), Jethro Tan (PFN)"
__license__ 	= "MIT"
__maintainer__ 	= "Olalekan Ogunmolu"
__email__ 		= "patlekano@gmail.com"
__status__ 		= "Testing"

import rospy
import logging, time
import numpy as np
import numpy.random as npr
import scipy.linalg as LA
from scipy.optimize import minimize, linprog
from gmm import gmm_2_parameters, parameters_2_gmm, \
                shape_DS, gmr_lyapunov
log_level = rospy.logdebug


class Cost(object):

    def __init__(self):
        """
            Class that estimates lyapunov energy function
        """
        self.success = True   # boolean indicating if constraints were violated

    def matVecNorm(self, x):
        return np.sqrt(np.sum(x**2, axis=0))

    def obj(self, p, x, xd, d, L, w, options):
        if L == -1:
            Vxf = dict()
            Vxf['d'] = d
            Vxf['n'] = int(np.sqrt(len(p)/d**2))
            Vxf['P'] = p.reshape(Vxf['n']*d,Vxf['n']*d)
            Vxf['SOS'] = 1
        else:
            Vxf         = shape_DS(p,d,L,options)
        V, Vx       = self.computeEnergy(x, [], Vxf)
        Vdot        = np.sum(Vx*xd, axis=0)  # derivative of J w.r.t. xd
        norm_Vx     = np.sqrt(np.sum(Vx * Vx, axis=0))
        norm_xd     = np.sqrt(np.sum(xd * xd, axis=0))
        # fix nans in Vdot and bot
        bot         = norm_Vx * norm_xd
        bot[bot==0] = options['tol_mat_bias']
        J           = Vdot/bot

        # projections onto positive orthant
        J[np.where(norm_xd==0)] = 0
        J[np.where(norm_Vx==0)] = 0
        J[np.where(Vdot>0)]     = J[np.where(Vdot>0)]**2      # solves psi(t,n)**2
        J[np.where(Vdot<0)]     = -w*J[np.where(Vdot<0)]**2   # solves -psi(t,n)**2

        J                       = np.sum(J)
        dJ                      = None

        return J#, dJ

    def optimize(self, obj_handle, ctr_handle, p0):
        a_ub = ctr_handle(p0)
        # print('a_ub: {} p0: {}'.format(a_ub.shape, p0.shape))
        a_ub = np.tile(a_ub, [len(a_ub), 1])
        A_ub = np.zeros((p0.shape[0], p0.shape[0]))
        A_ub[:a_ub.shape[0], :a_ub.shape[0]] = a_ub
        c = obj_handle(p0)
        objective = np.tile(c, len(A_ub))
        b_ub    = np.array([(0.0) for _ in range(len(A_ub))])
        # rospy.logdebug('A_ub: {}, obj: {}, b_ub: {}'.format(A_ub.shape, objective.shape, b_ub.shape))
        opt = linprog(
                        objective,
                        A_ub    = A_ub.T, #None,
                        b_ub    = b_ub,
                        bounds  = None,
                        method  = 'interior-point',
                        options = {'tol': 1e-8, 'disp': True}
                    )
        return opt

    def ctr_eigenvalue(self, p,d,L,options):
        Vxf = dict()
        if L == -1: # SOS
            Vxf['d'] = d
            Vxf['n'] = int(np.sqrt(matlength(p)/d**2))
            Vxf['P'] = p.reshape(Vxf['n']*d,Vxf['n']*d)
            Vxf['SOS'] = 1
            c  = np.zeros((Vxf['n']*d))
            ceq = []
        else:
            Vxf = shape_DS(p,d,L,options)
            if L > 0:
                c  = np.zeros([(L+1)*d+(L+1)*options['optimizePriors']])  #+options.variableSwitch
                if options['upperBoundEigenValue']:
                    ceq = np.zeros((L+1))
                else:
                    ceq = []
            else:
                c  = np.zeros((d))
                ceq = Vxf['P'].ravel().T.dot(Vxf['P'].ravel()) - 2

        dc = []
        dceq = []

        if L == -1:  # SOS
            c = -LA.eigvals(Vxf['P'] + Vxf['P'].T - np.eye(Vxf['n']*d)*options['tol_mat_bias'])
        else:
            for k in range(L):
                lambder = LA.eigvals(Vxf['P'][:,:,k+1] + (Vxf['P'][:,:,k+1]).T)/2.0
                c[k*d:(k+1)*d] = -lambder.real + options['tol_mat_bias']
                if options['upperBoundEigenValue']:
                    ceq[k+1] = 1.0 - np.sum(lambder.real)

        if L > 0 and options['optimizePriors']:
            c[(L+1)*d:(L+1)*d+L+1] = -Vxf['Priors'].squeeze()

        return c

    def check_constraints(self, p, ctr_handle, d, L, options):
        c = -ctr_handle(p)

        if L > 0:
            c_P = c[:L*d].reshape(d, L).T
        else:
            c_P = c

        i = np.where(c_P <= 0)
        # self.success = True

        if i:
            rospy.logerr('Error in P constraints')
            rospy.logerr('Eigen values of P^k violating constraints at ')
            rospy.logfatal('{}'.format(c_P[i]))
            self.success = False
        else:
            self.success = True

        if L > 1:
            if options['optimizePriors']:
                c_Priors = c[L*d+1:L*d+L]
                i = np.nonzero(c_Priors < 0)

                if i:
                    rospy.logerr('Errors in constraints on priors')
                    rospy.logerr('Values of the priors violates the constraints')
                    rospy.logfatal('{}'.format(c_Priors[i]))
                    self.success = False
                else:
                    self.success = True

            if len(c) > L*d+L:
                c_x_sw = c[L*d+L+1]
                if c_x_sw <= 0:
                    rospy.logerr('error in x_sw constraints')
                    rospy.logefatal('c_x_sw , %f', c_x_sw)
                    self.success = False
                else:
                    self.success = True

        if self.success:
            rospy.loginfo('Optimization finished successfully')
            rospy.loginfo(' ')
        else:
            rospy.logwarn('Optimization did not reach optimal point')
            rospy.logwarn('Some constraints were slightly violated')
            rospy.logwarn('Rerun the optimization w/diff initial; guesses to handle this issue')
            rospy.logwarn('increasing the # of P could help')

    def computeEnergy(self, X,Xd,Vxf, nargout=2):
        d = X.shape[0]
        nDemo = 1
        if nDemo>1:
            X = X.reshape(d,-1)
            Xd = Xd.reshape(d,-1)

        if Vxf['SOS']:
            V, dV = sos_lyapunov(X, Vxf['P'], Vxf['d'], Vxf['n'])
            if 'p0' in Vxf:
                V -= Vxf['p0']
        else:
            V, dV = gmr_lyapunov(X, Vxf['Priors'], Vxf['Mu'], Vxf['P'])

        if nargout > 1:
            if not Xd:
                Vdot = dV
            else:
                Vdot = np.sum(Xd*dV, axis=0)
        if nDemo>1:
            V = V.reshape(-1, nDemo).T
            if nargout > 1:
                Vdot = Vdot.reshape(-1, nDemo).T

        return V, Vdot

    def learnEnergy(self, Vxf0, Data, options):
        d = Vxf0['d']
        x = Data[:d,:]
        xd = Data[d:,:]

        # Transform the Lyapunov model to a vector of optimization parameters
        if Vxf0['SOS']:
            p0 = np.random.randn(d*Vxf0['n'], d*Vxf0['n']);
            p0 = p0.dot(p0.T)
            p0 = np.ravel(p0)
            Vxf0['L'] = -1; # to distinguish sos from other methods
        else:
            for l in range(Vxf0['L']):
                Vxf0['P'][:,:,l] = np.linalg.lstsq(Vxf0['P'][:,:,l], np.eye(d), rcond=None)[0]

            # in order to set the first component to be the closest Gaussian to origin
            to_sort     = self.matVecNorm(Vxf0['Mu'])
            idx         = np.argsort(to_sort, kind='mergesort')
            Vxf0['Mu']  = Vxf0['Mu'][:,idx]
            Vxf0['P']   = Vxf0['P'][:,:,idx]
            p0          = gmm_2_parameters(Vxf0,options)

        # account for targets in x and xd
        x               = x - np.expand_dims(x[:, -1], 1)
        xd              = xd - np.expand_dims(xd[:, -1], 1)

        obj_handle      = lambda p: self.obj(p, x, xd, d, Vxf0['L'], Vxf0['w'], options)
        ctr_handle      = lambda p: self.ctr_eigenvalue(p, d, Vxf0['L'], options)

        optim_res       = self.optimize(obj_handle, ctr_handle, p0)
        popt, J         = optim_res.x, optim_res.fun

        while not optim_res.success:
            optim_res   = self.optimize(obj_handle, ctr_handle, p0)
            popt, J     = optim_res.x, optim_res.fun

        if Vxf0['SOS']:
            Vxf['d']    = d
            Vxf['n']    = Vxf0['n']
            Vxf['P']    = popt.reshape(Vxf['n']*d,Vxf['n']*d)
            Vxf['SOS']  = 1
            Vxf['p0']   = self.compute_Energy(zeros(d,1),[],Vxf)
            self.check_constraints(popt,ctr_handle,d,0,options)
        else:
            # transforming back the optimization parameters into the GMM model
            Vxf             = parameters_2_gmm(popt,d,Vxf0['L'],options)
            Vxf['Mu'][:,0]  = 0
            Vxf['L']        = Vxf0['L']
            Vxf['d']        = Vxf0['d']
            Vxf['w']        = Vxf0['w']
            self.check_constraints(popt,ctr_handle,d,Vxf['L'],options)


        sumDet = 0
        for l in range(Vxf['L']+1):
            sumDet += np.linalg.det(Vxf['P'][:,:,l])

        # tol_mat_bias is a numrical stability check
        Vxf['P'][:,:,0] = Vxf['P'][:,:,0]/(sumDet + options['tol_mat_bias'])
        Vxf['P'][:,:,1:] = Vxf['P'][:,:,1:]/(np.sqrt(abs(sumDet)) + options['tol_mat_bias'])

        return Vxf, J
