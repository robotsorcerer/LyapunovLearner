from __future__ import print_function
import rospy
import logging, time
import numpy as np
import numpy.random as npr
import scipy.linalg as LA
from scipy.optimize import minimize, linprog
from gmm import gmm_2_parameters, parameters_2_gmm, \
                shape_DS, gmr_lyapunov#, sos_lyapunov

LOGGER = logging.getLogger(__name__)

def matVecNorm(x):
    return np.sqrt(np.sum(x**2, axis=0))

def obj(p, x, xd, d, L, w, options):
    if L == -1:
        Vxf = dict()
        Vxf['d'] = d
        Vxf['n'] = int(np.sqrt(matlength(p)/d**2))
        Vxf['P'] = p.reshape(Vxf['n']*d,Vxf['n']*d)
        Vxf['SOS'] = 1
    else:
        Vxf         = shape_DS(p,d,L,options)
    V, Vx       = computeEnergy(x, [], Vxf)
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

# def optimize(obj_handle, ctr_handle, p0):
#     opt = minimize(
#             obj_handle,
#             x0=p0,
#             method='L-BFGS-B',
#             jac=False,
#             bounds= [(0.1, None) for _ in range(len(p0))], # # no negative p values
#             options={'ftol': 1e-8, 'disp': False}
#             )
#     return opt

def optimize(obj_handle, ctr_handle, p0):
    a_ub = ctr_handle(p0)
    # print('a_ub: {} p0: {}'.format(a_ub.shape, p0.shape))
    a_ub = np.tile(a_ub, [len(a_ub), 1])
    A_ub = np.zeros((p0.shape[0], p0.shape[0]))
    A_ub[:a_ub.shape[0], :a_ub.shape[0]] = a_ub
    c = obj_handle(p0)
    objective = np.tile(c, len(A_ub))
    b_ub    = np.array([(0.0) for _ in range(len(A_ub))])
    print('A_ub: {}, obj: {}, b_ub: {}'.format(A_ub.shape, objective.shape, b_ub.shape))
    opt = linprog(
                    objective,
                    A_ub    = A_ub.T, #None,
                    b_ub    = b_ub,
                    bounds  = None,
                    method  = 'interior-point',
                    options = {'tol': 1e-8, 'disp': True}
                )
    return opt

def ctr_eigenvalue(p,d,L,options):
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

    # print('c: {}, ceq: {}'.format(c.shape, ceq.shape))
    return c #, ceq

def ctr_eigenvalue2(p,d,L,options):
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
            c  = np.zeros_like(p)#([(L+1)*d+(L+1)*options['optimizePriors']])  #+options.variableSwitch
            if options['upperBoundEigenValue']:
                ceq = np.zeros((L+1))
            else:
                ceq = []
        else:
            c  = np.zeros_like(p)#((d))
            ceq = Vxf['P'].T.ravel().dot(Vxf['P'].ravel()) - 2

    dc = []
    dceq = []

    if L == -1:  # SOS
        c = -LA.eigvals(Vxf['P'] + Vxf['P'].T - np.eye(Vxf['n']*d)*options['tol_mat_bias'])
    else:
        for k in range(L):
            lambder = LA.eigvals(Vxf['P'][:,:,k+1] + (Vxf['P'][:,:,k+1]).T)/2.0
            # print('c[k*d:(k+1)*d]: ', c[k*d:(k+1)*d].shape, '-lambder.real: ', -lambder.real.shape)
            c[k*d:(k+1)*d] = np.expand_dims(-lambder.real + options['tol_mat_bias'], 1)
            if options['upperBoundEigenValue']:
                ceq[k+1] = 1.0 - np.sum(lambder.real)

    if L > 0 and options['optimizePriors']:
        c[(L+1)*d:(L+1)*d+L+1] = -Vxf['Priors']#.squeeze()

    return c #, ceq

def check_constraints(p, ctr_handle, d, L, options):
    c = -ctr_handle(p)

    if L > 0:
        c_P = c[:L*d].reshape(d, L).T
    else:
        c_P = c

    i = np.where(c_P <= 0)
    success = True

    if i:
        rospy.logerr('Error in P constraints')
        rospy.logerr('eigen values of P^k violating constraints at ')
        rospy.logfatal('{}'.format(c_P[i]))
        success = False

    if L > 1:
        if options['optimizePriors']:
            c_Priors = c[L*d+1:L*d+L]
            i = np.nonzero(c_Priors < 0)

            if i:
                rospy.logerr('Errors in constraints on priors')
                rospy.logerr('values of the priors violates the constraints')
                rospy.logfatal('{}'.format(c_Priors[i]))
                success = False

        if len(c) > L*d+L:
            c_x_sw = c[L*d+L+1]
            if c_x_sw <= 0:
                rospy.logerr('error in x_sw constraints')
                rospy.logefatal('c_x_sw , %f', c_x_sw)
                success = False

    if success:
        rospy.loginfo('Optimization finished successfully')
        rospy.loginfo(' ')
    else:
        rospy.loginfo('Optimization did not reach optimal point')
        rospy.loginfo('Some constraints were slightly violated')
        rospy.loginfo('Rerun the optimization w/diff initial; guesses to hanfdle this issue')
        rospy.loginfo('increasing the # of P could help')

def learnEnergy(Vxf0, Data, options):
    d = Vxf0['d'] #int(Data.shape[0]/2)
    x = Data[:d,:]
    xd = Data[d:,:]

    # Transform the Lyapunov model to a vector of optimization parameters
    if Vxf0['SOS']:
        p0 = npr.randn(d*Vxf0['n'], d*Vxf0['n']);
        p0 = p0.dot(p0.T)
        p0 = np.ravel(p0)
        Vxf0['L'] = -1; # to distinguish sos from other methods
    else:
        for l in range(Vxf0['L']):
            Vxf0['P'][:,:,l] = np.linalg.lstsq(Vxf0['P'][:,:,l], np.eye(d), rcond=None)[0]

        # in order to set the first component to be the closest Gaussian to origin
        to_sort = matVecNorm(Vxf0['Mu'])
        idx = np.argsort(to_sort, kind='mergesort')
        Vxf0['Mu'] = Vxf0['Mu'][:,idx]
        Vxf0['P']  = Vxf0['P'][:,:,idx]
        p0 = gmm_2_parameters(Vxf0,options)

    # account for targets in x and xd
    x  = x - np.expand_dims(x[:, -1], 1)
    xd = xd - np.expand_dims(xd[:, -1], 1)

    obj_handle = lambda p: obj(p, x, xd, d, Vxf0['L'], Vxf0['w'], options)
    ctr_handle = lambda p: ctr_eigenvalue(p, d, Vxf0['L'], options)

    optim_res = optimize(obj_handle, ctr_handle, p0)
    popt, J = optim_res.x, optim_res.fun

    while not optim_res.success:
        optim_res = optimize(obj_handle, ctr_handle, p0)
        popt, J = optim_res.x, optim_res.fun

    if Vxf0['SOS']:
        Vxf['d']    = d
        Vxf['n']    = Vxf0['n']
        Vxf['P']    = popt.reshape(Vxf['n']*d,Vxf['n']*d)
        Vxf['SOS']  = 1
        Vxf['p0']   = compute_Energy(zeros(d,1),[],Vxf)
        check_constraints(popt,ctr_handle,d,0,options)
    else:
        # transforming back the optimization parameters into the GMM model
        Vxf             = parameters_2_gmm(popt,d,Vxf0['L'],options)
        Vxf['Mu'][:,0]  = 0
        Vxf['L']        = Vxf0['L']
        Vxf['d']        = Vxf0['d']
        Vxf['w']        = Vxf0['w']
        check_constraints(popt,ctr_handle,d,Vxf['L'],options)

    sumDet = 0
    for l in range(Vxf['L']+1):
        sumDet += np.linalg.det(Vxf['P'][:,:,l])

    Vxf['P'][:,:,0] = Vxf['P'][:,:,0]/sumDet
    Vxf['P'][:,:,1:] = Vxf['P'][:,:,1:]/np.sqrt(abs(sumDet))

    return Vxf, J

def computeEnergy(X,Xd,Vxf, nargout=2):
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
