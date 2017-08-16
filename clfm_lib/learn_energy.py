import sys
import logging
import numpy as np
import scipy as sp
import numpy.random as npr
import scipy.linalg as linalg

from inspect import currentframe, getframeinfo
import cvxpy as cvx
from cvxopt import solvers, matrix, spdiag, div # to solve convex problems


from .compute_energy import computeEnergy
from config import hyperparams # config is in the sys path since demo is launched from root package

frameinfo = getframeinfo(currentframe())
LOGGER = logging.getLogger(__name__)

# global vars
p,x,xd,d,Vxf0,options = (None for _ in range(6))

use_convex = hyperparams['use_cvxopt']

def matVecNorm(x):
    return np.sqrt(np.sum(x**2, axis=0))

def matlength(x):
  # find the max of a numpy matrix dims
  return np.max(x.shape)


def ctr_eigenvalue(p,d,L,options):
  # This function computes the derivative of the constrains w.r.t.
  # optimization parameters.
  Vxf = dict()
  if L == -1: # SOS
      Vxf['d'] = d
      Vxf['n'] = int(np.sqrt(matlength(p)/d**2))
      Vxf['P'] = p.reshape(Vxf['n']*d,Vxf['n']*d)
      Vxf['SOS'] = 1
      c  = np.zeros(( Vxf['n']*d, 1 ))
      ceq = np.array(())
  else:
      Vxf = shape_DS(p,d,L,options)
      if L > 0:
          c  = np.zeros(((L+1)*d+(L+1)*options['optimizePriors'],1))  #+options.variableSwitch
          if options['upperBoundEigenValue']:
              ceq = np.zeros((L+1,1))
          else:
              ceq = np.array(()) # zeros(L+1,1);
      else:
          c  = np.zeros((d,1))
          ceq = (np.ravel(Vxf['P']).T).dot(np.ravel(Vxf['P'])) -2

  dc = np.array(())
  dceq = np.array(())

  if L == -1:  # SOS
    c = -np.linalg.eigvals(Vxf['P'] + Vxf['P'].T - np.eye(Vxf['n']*d)*options['tol_mat_bias'])
  else:
    for k in range(L):
      lambder = sp.linalg.eigvals(Vxf['P'][:,:,k+1] + (Vxf['P'][:,:,k+1]).T)
      #   print("check that div 2 works in line %d in file %s ".format(frameinfo.lineno, frameinfo.filename))
      lambder = np.divide(lambder.real, 2.0)
      lambder = np.expand_dims(lambder, axis=1)
      # print('lambder: ', lambder.shape, c[k*d:(k+1)*d].shape)
      c[k*d:(k+1)*d] = -lambder.real + options['tol_mat_bias']
      if options['upperBoundEigenValue']:
        ceq[k+1] = 1.0 - np.sum(lambder.real) # + Vxf.P(:,:,k+1)'

  if L > 0 and options['optimizePriors']:
    #   print(Vxf['Priors'].shape)
    #   print(c.shape)
    #   print(c[(L+1)*d:(L+1)*d+L+1].shape)
      c[(L+1)*d:(L+1)*d+L+1] = -Vxf['Priors']
      #   was: c[(L+1)*d+1:(L+1)*d+L+1] = -Vxf['Priors']

  return c, ceq, dc, dceq

def gmm_2_parameters(Vxf, options):
  # transforming optimization parameters into a column vector
  d = Vxf['d']
  if Vxf['L'] > 0:
      if options['optimizePriors']:
          p0 = np.vstack((
                           np.expand_dims(np.ravel(Vxf['Priors']), axis=1), # will be a x 1
                           np.expand_dims(Vxf['Mu'][:,1:], axis=1).reshape(Vxf['L']*d,1)
                        ))
      else:
          p0 = Vxf['Mu'][:,2:].reshape(Vxf['L']*d, 1) #print(p0) # p0 will be 4x1
  else:
      p0 = np.array(())

  for k in range(Vxf['L']):

      p0 = np.vstack((
                      p0,
                      Vxf['P'][:,:,k+1].reshape(d**2,1)
                    ))

  return p0

def parameters_2_gmm(popt, d, L, options):
  # transforming the column of parameters into Priors, Mu, and P
  Vxf = shape_DS(popt, d, L, options)

  return Vxf


def check_options(*args):
    options = args[0] if args else None
    if (not args) or ('tol_mat_bias' not in options):
        options['tol_mat_bias'] = 1e-15
    if 'tol_stopping' not in options:
        options['tol_stopping'] = 10^-10
    if 'max_iter' not in options:
        options['max_iter'] = 1000
    if not 'display' in options:
        options['display'] = 1
    else:
        options['display'] = options['display'] > 0
    if not 'optimizePriors' in options:
        options['optimizePriors'] = True
    else:
        options['optimizePriors'] = options['optimizePriors'] > 0

    if not 'upperBoundEigenValue' in options:
        options['upperBoundEigenValue'] = True
    else:
        options['upperBoundEigenValue'] = options['upperBoundEigenValue'] > 0

    return options


def check_constraints(p,ctr_handle,d,L,options):
# checking if every thing goes well here. Sometimes if the parameter
# 'options['cons_penalty']' is not big enough, the constrains may be violated.
# Then this function notifies the user to increase 'options['cons_penalty']'.

  c = -ctr_handle[p]

  if L > 0:
      c_P = c[0:L*d].reshape(d,L).T
  else:
      c_P = c

  idx = np.nonzero(c_P<=0)
  bool_success = True
  if idx.size != 0:
      idx = np.sort(idx);
      err = np.concatenate((np.ravel(idx), c_P[idx,:]), axis=1)
      sys.stdout.write('Error in the constraints on P!\n')
      sys.stdout.write('Eigenvalues of the P^k that violates the constraints:\n')
      sys.stdout.write(err)
      sys.stdout.flush()
      bool_success = False


  if L>1:
      if options['optimizePriors']:
          c_Priors = c[L*d+1:L*d+L]
          idx = c_Priors[c_Priors<0]
          if idx.size==0:
              err = np.concatenate((np.ravel(idx), c_Priors[idx]), axis=1);
              sys.stdout.write('Error in the constraints on Priors!')
              sys.stdout.write('Values of the Priors that violates the constraints:')
              sys.stdout.write(err)
              sys.stdout.flush()
              bool_success = False;

      if matlength(c) > L*d + L:
          c_x_sw = c[L * d + L + 1]
          if c_x_sw <= 0:
              sys.stdout.write('Error in the constraints on x_sw!')
              sys.stdout.write('x_sw = %f',c_x_sw)
              bool_success = False;

  if bool_success:
      sys.stdout.write('Optimization finished successfully.')
      sys.stdout.write(' ')
      sys.stdout.write(' ')
  else:
      sys.stdout.write('Optimization did not reach to an optimal point.')
      sys.stdout.write('Some constraints were slightly violated.')
      sys.stdout.write('Re-run the optimization with different initial guess to handle this issue.')
      sys.stdout.write('Increasing the number of P could be also helpful.')
      sys.stdout.write(' ')
      sys.stdout.write(' ')

def shape_DS(p,d,L,options):
  # transforming the column of parameters into Priors, Mu, and P
  P = np.zeros((d,d,L+1))
  optimizePriors = options['optimizePriors']
  # print('options', optimizePriors)
  if L == 0:
      Priors = 1
      Mu = np.zeros((d,1))
      i_c = 1
  else:
      if optimizePriors: #options['optimizePriors']:
          Priors = p[:L+1]
          i_c = L+1
      else:
          Priors = np.ones((L+1,1))
          i_c = 0

      Priors = np.divide(Priors, np.sum(Priors))
      Mu = np.hstack((np.zeros((d,1)), p[[i_c+ x for x in range(d*L)]].reshape(d,L)))
      i_c = i_c+d*L+1

  for k in range(L):
    # print('p shape: ', p.shape)
    # print('i_c: {}, k: {}, d: {}, '.format(i_c, k, d))
    # print('range: ', range(i_c+k*(d**2),i_c+(k+1)*(d**2)-1))
    # print('p range: \n', p[range(i_c+k*(d**2),i_c+(k+1)*(d**2)-1)])
    P[:,:,k+1] = p[range(i_c+k*(d**2)-1,i_c+(k+1)*(d**2)-1)].reshape(d,d)

  Vxf           = dict()
  Vxf['Priors'] = Priors
  Vxf['Mu']     = Mu
  Vxf['P']      = P
  Vxf['SOS']    = 0

  return Vxf


def learnEnergy(Vxf0, Data, options):
  """

  This function builds an estimate of a Lyapunov (energy) function from
  demonstrations.

   Syntax:
         [Vxf J] = learnEnergy(Vxf0,Data,options)

   to also pass a structure of desired options.

   Important NOTE: Both the demonstration data, and the model estimation
   should be in the target frame of reference. In other words, this codes
   assumes that the target is at the origin!

   Inputs -----------------------------------------------------------------

     o Vxf_0:     A structure variable representing the initial guess for the
                  energy function. Please refer to the Output variable for
                  further details about its fields.

     o Data:      A 2d x N_Total matrix containing all demonstration data points.
                  Rows 1:d corresponds to trajectories and the rows d+1:2d
                  are their first time derivatives. Each column of Data stands
                  for a datapoint. All demonstrations are put next to each other
                  along the second dimension. For example, if we have 3 demos
                  D1, D2, and D3, then the matrix Data is:
                                   Data = [[D1] [D2] [D3]]

     o options: A structure to set the optional parameters of the solver.
                The following parameters can be set in the options:
         - .tol_mat_bias:     a very small positive scalar to avoid
                              having a zero eigen value in matrices P^l [default: 10^-15]

         - .tol_stopping:     A small positive scalar defining the stoppping
                              tolerance for the optimization solver [default: 10^-10]

         - .i_max:            maximum number of iteration for the solver [default: i_max=1000]

         - .display:          An option to control whether the algorithm
                              displays the output of each iterations [default: true]

         - .optimizePriors    This is an added feature that is not reported in the paper. In fact
                              the new CLFDM model now allows to add a prior weight to each quadratic
                              energy term. IF optimizePriors sets to false, unifrom weight is considered;
                              otherwise, it will be optimized by the sovler. [default: true]

         - .upperBoundEigenValue     This is also another added feature that is impelemnted recently.
                                     When set to true, it forces the sum of eigenvalues of each P^l
                                     matrix to be equal one. [default: true]


   Outputs ----------------------------------------------------------------

     o Vxf:      A structure variable representing the energy function. It
                 is composed of the following fields:

         - .d:       Dimension of the state space, d>0.

         - .L:       The number of asymmetric quadratic components L>=0.

         - .Priors:  1 x K array representing the prior weight of each energy
                    component. Prioris are positive scalars between 0 and 1.

         - .Mu:      Each Mu(:,i) is a vector of R^d and represent the center of
                    the energy component i. Note that by construction Mu(:,1)=0!

         - .P:       Each P(:,:,i), i=1:L+1, is a d x d positive definite matrix.
                    P(:,:,1) corresponds to the symmetric energy component P^0.
                    P^1 to P^{L+1} are asymmetric quadratic terms. Note that the
                    matrices P^i are not necessarily symmetric.

         - .w:      A positive scalar weight regulating the priority between the
                    two objectives of the opitmization. Please refer to the
                    page 7 of the paper for further information.

         - .SOS:    This is an internal variable, and is automatically set to false

     o J:      The value of the objective function at the optimized point

  ###########################################################################
  ##         Copyright (c) 2014 Mohammad Khansari, LASA Lab, EPFL,         ##
  ##          CH-1015 Lausanne, Switzerland, http://lasa.epfl.ch           ##
  ###########################################################################

  Ported to Python by Lekan Ogunmolu
          August 5, 2017
  """
  # augment undefined options by updating the options dict
  if not options:
    options = check_options()
  else:
    options = check_options(options)
  # for k, v in Vxf0.items():
  #     print(k, v)
  d = int(Data.shape[0]/2)  # dimension of model
  x = Data[:d,:]     # state space
  xd = Data[d:2*d,:]    # derivatives of the state space
  print('x: ', x.shape, 'xd ', xd.shape)
  Vxf0['SOS'] = False

  # Optimization
  # Transform the Lyapunov model to a vector of optimization parameters
  if Vxf0['SOS']:
    p0 = npr.randn(d*Vxf0['n'], d*Vxf0['n']);
    p0 = p0.dot(p0.T)
    p0 = np.ravel(p0)
    Vxf0['L'] = -1; # to distinguish sos from other methods
  else:
    for l in range(Vxf0['L']):
      try:
        Vxf0['P'][:,:,l+1] = sp.linalg.solve( Vxf0['P'][:,:,l+1], sp.eye(d))
      except sp.linalg.LinAlgError as e:
        LOGGER.debug('LinAlgError: %s', e)

    # in order to set the first component to be the closest Gaussian to origin
    to_sort = matVecNorm(Vxf0['Mu'])
    idx = np.argsort(to_sort, kind='mergesort')
    Vxf0['Mu'] = Vxf0['Mu'][:,idx]
    Vxf0['P']  = Vxf0['P'][:,:,idx]
    p0 = gmm_2_parameters(Vxf0,options)

  c,ceq, dc, dceq = ctr_eigenvalue(p0,d,Vxf0['L'],options)

  """
    popt is value of minimization
    J is value of cost at the optimal solution
    c are the ineq constraints
    ceq are the equality constraints
    dc and dceq are the corresponding derivatives
  """
  def optimize(p0, d, L, w, options):
    # print('Vxf', Vxf)
    # n, T = Vxf['n'], options['max_iter']

    # x = cvx.Variable(n, T+1)   # states of the system
    # for t in range(T):
    if L == -1: #SOS
      Vxf['n']    = np.sqrt(matlength(p)/d**2)
      Vxf['d']    = d
      Vxf['P']    = p.reshape(Vxf['n']*d,Vxf['n']*d)
      Vxf['SOS']  = 1
    else:
      Vxf = shape_DS(p0,d,L,options)
      Vxf.update(Vxf)
    _, Vx         = computeEnergy(x,np.array(()), Vxf, nargout=2)
    # xd will be 2 x 750
    # Vx should be (2, 750),
    # Vdot  (750,) for expt 0,
    Vdot          = np.sum(Vx*xd, axis=0)  #derivative of J w.r.t. xd
    norm_Vx       = np.sqrt(np.sum(Vx * Vx, axis=0))
    norm_xd       = np.sqrt(np.sum(xd * xd, axis=0))
    # print('Vx: {}, Vdot, {} norm_Vx, {}, xd: {}, norm_xd: {}, butt: {}'.format(Vx.shape, Vdot.shape,
    #                   norm_Vx.shape, (xd).shape, norm_xd.shape, butt.shape) )
    # x:  (2, 750), xd  (2, 750)
    # Vx: (2, 750), Vdot, (750,) norm_Vx, (750,), norm_xd: (750,), xd: (2, 750), butt: (750, 750)
    # expand arrays to fit suppose shape
    Vdot      = np.expand_dims(Vdot, axis=0)
    norm_Vx   = np.expand_dims(norm_Vx, axis=0)
    norm_xd   = np.expand_dims(norm_xd, axis=0)
    butt          = norm_Vx * norm_xd
    # w was added by Lekan to regularize the invalid values in butt
    J             = Vdot / (butt + w)
    J[np.where(norm_xd==0)] = 0
    J[np.where(norm_Vx==0)] = 0
    J[np.where(Vdot>0)]     = J[np.where(Vdot>0)]**2      # solves psi(t,n)**2
    J[np.where(Vdot<0)]     = -w*J[np.where(Vdot<0)]**2   # # J should be (1, 750)
    # print('J: ', J.shape)
    J = np.sum(J, axis=1) # Jsum would be of shape (1,)
    print('J sum: ', J[0])

    # print('Vxf: ', -Vxf['P'], 'L: ', L)
    constraints = []
    for l in range(L):
        constraints.append(cvx.Parameter(Vxf['P'][:,:,l]>=0))
    # The 'minimize' objective must resolve to a scalar.
    J_var = cvx.Variable(cvx.vec(J))
    obj   = cvx.Minimize(sum( J ) )
    prob  = cvx.Problem(obj, constraints)
    optionsAlg = {
        'maxiters': options['max_iter'],
        'show_progress': True,
        'refinement': 1,
        'abstol': 1e-12,
        'reltol':  1e-10,
        'feastol': 1e-7,
      }
    prob.solve(solver=CVXOPT, verbose=True, options=optionsAlg)
    # prob.solve()

    return prob#.status, prob.value, J.value

  w = Vxf0['w']
  L = Vxf0['L']

  opt_res = optimize(p0, Vxf0['d'], Vxf0['L'], Vxf0['w'], options)
  print('status: {}, value: {}', opt_res.status, opt_res.value)

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
  Vxf['P'][:,:,1:] = Vxf['P'][:,:,1:]/np.sqrt(sumDet)

  return Vxf, J
