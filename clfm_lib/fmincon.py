import sys
import numpy as np
import scipy as sp
import scipy.sparse as spr

# from .classify_bounds_on_vars import classifyBoundsOnVars

def matlength(x):
  return np.max(x.shape)

def checkbbounds(xin,lbin,ubin,nvars):

  #CHECKBOUNDS Verify that the bounds are valid with respect to initial point.
  #
  # This is a helper function.

  #   [X,LB,UB,X,FLAG] = CHECKBOUNDS(X0,LB,UB,nvars)
  #   checks that the upper and lower
  #   bounds are valid (LB <= UB) and the same length as X (pad with -inf/inf
  #   if necessary); warn if too short or too long.  Also make LB and UB vectors
  #   if not already. Finally, inf in LB or -inf in UB throws an error.

  #   Copyright 1990-2012 The MathWorks, Inc.

  msg = np.array(())
  # Turn into column vectors
  lb = np.ravel(lbin)
  ub = np.ravel(ubin)
  xin = np.ravel(xin)

  lenlb = matlength(lb);
  lenub = matlength(ub);

  # Check lb length
  if lenlb > nvars:
      sys.stdout.write('optimlib:checkbounds:IgnoringExtraLbs')
      lb = lb[:nvars]
      lenlb = nvars
  elif lenlb < nvars: # includes empty lb case
      if lenlb > 0:
          # lb is non-empty and length(lb) < nvars.
          sys.stdout.write('optimlib:checkbounds:PadLbWithMinusInf')

      lb = np.vstack(
                      lb, 
                      -np.inf*np.ones((nvars-lenlb,1))
                    )
      lenlb = nvars
  else:
    pass

  # Check ub length
  if lenub > nvars:
      sys.stdout.write('optimlib:checkbounds:IgnoringExtraUbs')
      ub = ub[:nvars]
      lenub = nvars
  elif lenub < nvars: # includes empty ub case
      if lenub > 0:
          # ub is non-empty and length(ub) < nvars.
          sys.stdout.write('optimlib:checkbounds:PadUbWithInf')
      
      ub = np.vstack(ub,
                     np.inf*np.ones((nvars-lenub,1))
                    )
      lenub = nvars
  

  # Check feasibility of bounds
  length = np.min([lenlb,lenub]);
  if np.any( [lb[1:length].T > ub[1:length].T] ):
      count = (np.sum[lb>ub]).todense()
      if count == 1:
          msg = 'Exiting due to infeasibility:  lower bound exceeds the' +
              ' corresponding upper bound by ' + str(count)
      else:
          msg = 'Exiting due to infeasibility: lower bounds exceeds the' +
              ' corresponding upper bound by ' + str(count)

  # Check if -inf in ub or inf in lb
  if np.any( [ub==-np.inf)] ):
      sys.stdout.write('optimlib:checkbounds:MinusInfUb')
  elif np.any( [lb==np.inf] ):
      sys.stdout.write('optimlib:checkbounds:PlusInfLb')

  x = xin

  return x, lb, ub, msg

def prepareOptionsForSolver(options, solverName):
  # This portion of the code does not seem relevant to me
  if (isinstance(options, dict) and options['TolFun'] and options['TolFunValue'] ):
    options['TolFunValue'] = options['TolFun']
  
def startx( ub, lb, xstart=np.array(()), xstartOutOfBounds_idx=np.array(()) ):
  
  #STARTX Box-centered start point.
  #
  # This is a helper function.

  # xstart = STARTX(ub,lb,xstart,xstartOutOfBounds_idx) sets the components
  # that violate the bounds to a centered value.
      
  arg1 = np.logical_and( (ub < inf),  (lb == -inf) )
  arg2 = np.logical_and( (ub == inf), (lb > -inf)  )
  arg3 = np.logical_and( (ub < inf),  (lb > -inf)  )
  arg4 = np.logical_and( (ub == inf), (lb == -inf) )

  if not xstart.size:
      arg1 = np.logical_and(arg1, xstartOutOfBounds_idx)
      arg2 = np.logical_and(arg2, xstartOutOfBounds_idx)
      arg3 = np.logical_and(arg3, xstartOutOfBounds_idx)
      arg4 = np.logical_and(arg4, xstartOutOfBounds_idx)
  else:
      n = matlength(ub);
      xstart = np.zeros((n,1))

  w = np.max( [np.abs(ub),1] );
  xstart[arg1] = ub[arg1] - .5*w[arg1]
  
  ww = np.max( [np.abs(lb),1] )
  xstart[arg2] = lb[arg2] + .5*ww[arg2]
  
  xstart[arg3] = (ub[arg3] + lb[arg3])/2s

  xstart[arg4] = 1

  return xstart

def classifyBoundOnVars(lb,ub,nVar,findFixedVar):
  #classifyBoundsOnVars Helper function that identifies variables
  # that are fixed, and that have finite bounds. 
  # Set empty vector bounds to vectors of +Inf or -Inf
  if not lb.size:
      lb = -np.inf*np.ones((nVar,1))

  if not ub.size:
      ub = np.inf*np.ones((nVar,1))

  # Check for NaN
  if (np.any(np.isnan(lb)) or np.any(np.isnan(ub)) ):
      print('optimlib:classifyBoundsOnVars:NaNBounds')

  # Check for +Inf lb and -Inf ub
  if np.any( lb == np.inf ):
      print('optimlib:classifyBoundsOnVars:PlusInfLb')

  if np.any(ub == -np.inf):
      print('optimlib:classifyBoundsOnVars:MinusInfUb')

  # Check for fixed variables equal to Inf or -Inf
  if np.any( np.logical_and( (np.ravel(lb) == np.ravel(ub)), (np.isinf(np.ravel(lb))) ) ):
      print('optimlib:classifyBoundsOnVars:FixedVarsAtInf')


  # Fixed variables
  if findFixedVar:
      xIndices['fixed'] = equalFloat(lb,ub,eps)
  else: # Do not attempt to detect fixed variables
      xIndices['fixed'] = False * np.ones((nVar,1))


  # Finite lower and upper bounds; exclude fixed variables
  xIndices['finiteLb'] = np.logical_not( np.logical_and(
                               xIndices['fixed'], np.isfinite(np.ravel(lb)) 
                            ) 
                    )
  xIndices['finiteUb'] = np.logical_not( np.logical_and(
                               xIndices['fixed'], np.isfinite(np.ravel(ub)) 
                            ) 
                    )
  return xIndices

def equalFloat(v1,v2,tolerance):
  # equalFloat Helper function that compares two vectors
  # using a relative difference and returns a boolean
  # vector.

  # Indices for which both v1 and v2 are finite
  finiteRange_idx = np.logical_and(np.isfinite(np.ravel(v1)), np.isfinite(np.ravel(v2)))

  # Indices at which v1 and v2 are (i) finite and (ii) equal in a 
  # floating point sense
  isEqual_idx = np.logical_and(finiteRange_idx, np.abs( np.ravel(v1)- np.ravel(v2) ) <= \
                 tolerance * np.max( 1, np.max([np.abs(np.ravel(v1)), np.abs(np.ravel(v2))]) ))
  return isEqual_idx

def fmincon(FUN,X,A,B,Aeq,Beq,LB,UB,NONLCON,options,*args):
  #
  # FMINCON finds a constrained minimum of a function of several variables.
  #   FMINCON attempts to solve problems of the form:
  #    min F(X)  subject to:  A*X  <= B, Aeq*X  = Beq (linear constraints)
  #     X                     C(X) <= 0, Ceq(X) = 0   (nonlinear constraints)
  #                              LB <= X <= UB        (bounds)
  #    
  #   FMINCON implements four different algorithms: interior point, SQP,
  #   active set, and trust region reflective. Choose one via the option
  #   Algorithm: for instance, to choose SQP, set OPTIONS =
  #   optimoptions('fmincon','Algorithm','sqp'), and then pass OPTIONS to
  #   FMINCON.
  #                                                           
  #   X = FMINCON(FUN,X0,A,B) starts at X0 and finds a minimum X to the 
  #   function FUN, subject to the linear inequalities A*X <= B. FUN accepts 
  #   input X and returns a scalar function value F evaluated at X. X0 may be
  #   a scalar, vector, or matrix. 
  #   #   X = FMINCON(FUN,X0,A,B,Aeq,Beq) minimizes FUN subject to the linear 
  #   equalities Aeq*X = Beq as well as A*X <= B. (Set A=[] and B=[] if no 
  #   inequalities exist.)
  #   #   X = FMINCON(FUN,X0,A,B,Aeq,Beq,LB,UB) defines a set of lower and upper
  #   bounds on the design variables, X, so that a solution is found in 
  #   the range LB <= X <= UB. Use empty matrices for LB and UB
  #   if no bounds exist. Set LB(i) = -Inf if X(i) is unbounded below; 
  #   set UB(i) = Inf if X(i) is unbounded above.
  #   #   X = FMINCON(FUN,X0,A,B,Aeq,Beq,LB,UB,NONLCON) subjects the minimization
  #   to the constraints defined in NONLCON. The function NONLCON accepts X 
  #   and returns the vectors C and Ceq, representing the nonlinear 
  #   inequalities and equalities respectively. FMINCON minimizes FUN such 
  #   that C(X) <= 0 and Ceq(X) = 0. (Set LB = [] and/or UB = [] if no bounds
  #   exist.)
  #   #   X = FMINCON(FUN,X0,A,B,Aeq,Beq,LB,UB,NONLCON,OPTIONS) minimizes with
  #   the default optimization parameters replaced by values in OPTIONS, an
  #   argument created with the OPTIMOPTIONS function. See OPTIMOPTIONS for
  #   details. For a list of options accepted by FMINCON refer to the
  #   documentation.
  #  
  #   X = FMINCON(PROBLEM) finds the minimum for PROBLEM. PROBLEM is a
  #   structure with the function FUN in PROBLEM.objective, the start point
  #   in PROBLEM.x0, the linear inequality constraints in PROBLEM.Aineq
  #   and PROBLEM.bineq, the linear equality constraints in PROBLEM.Aeq and
  #   PROBLEM.beq, the lower bounds in PROBLEM.lb, the upper bounds in 
  #   PROBLEM.ub, the nonlinear constraint function in PROBLEM.nonlcon, the
  #   options structure in PROBLEM.options, and solver name 'fmincon' in
  #   PROBLEM.solver. Use this syntax to solve at the command line a problem 
  #   exported from OPTIMTOOL. 
  #   #   [X,FVAL] = FMINCON(FUN,X0,...) returns the value of the objective 
  #   function FUN at the solution X.
  #   #   [X,FVAL,EXITFLAG] = FMINCON(FUN,X0,...) returns an EXITFLAG that
  #   describes the exit condition. Possible values of EXITFLAG and the
  #   corresponding exit conditions are listed below. See the documentation
  #   for a complete description.
  #   
  #   All algorithms:
  #     1  First order optimality conditions satisfied.
  #     0  Too many function evaluations or iterations.
  #    -1  Stopped by output/plot function.
  #    -2  No feasible point found.
  #   Trust-region-reflective, interior-point, and sqp:
  #     2  Change in X too small.
  #   Trust-region-reflective:
  #     3  Change in objective function too small.
  #   Active-set only:
  #     4  Computed search direction too small.
  #     5  Predicted change in objective function too small.
  #   Interior-point and sqp:
  #    -3  Problem seems unbounded.
  #   #   [X,FVAL,EXITFLAG,OUTPUT] = FMINCON(FUN,X0,...) returns a dictionary 
  #   OUTPUT with information such as total number of iterations, and final 
  #   objective function value. See the documentation for a complete list.
  #   #   [X,FVAL,EXITFLAG,OUTPUT,LAMBDA] = FMINCON(FUN,X0,...) returns the 
  #   Lagrange multipliers at the solution X: LAMBDA.lower for LB, 
  #   LAMBDA.upper for UB, LAMBDA.ineqlin is for the linear inequalities, 
  #   LAMBDA.eqlin is for the linear equalities, LAMBDA.ineqnonlin is for the
  #   nonlinear inequalities, and LAMBDA.eqnonlin is for the nonlinear 
  #   equalities.
  #   #   [X,FVAL,EXITFLAG,OUTPUT,LAMBDA,GRAD] = FMINCON(FUN,X0,...) returns the 
  #   value of the gradient of FUN at the solution X.
  #   #   [X,FVAL,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = FMINCON(FUN,X0,...) 
  #   returns the value of the exact or approximate Hessian of the Lagrangian
  #   at X. 
  #   #   Examples
  #     FUN can be specified using @:
  #        X = fmincon(@humps,...)
  #     In this case, F = humps(X) returns the scalar function value F of 
  #     the HUMPS function evaluated at X.
  #   #     FUN can also be an anonymous function:
  #        X = fmincon(@(x) 3*sin(x(1))+exp(x(2)),[1;1],[],[],[],[],[0 0])
  #     returns X = [0;0].
  #   #   If FUN or NONLCON are parameterized, you can use anonymous functions to
  #   capture the problem-dependent parameters. Suppose you want to minimize 
  #   the objective given in the function myfun, subject to the nonlinear 
  #   constraint mycon, where these two functions are parameterized by their 
  #   second argument a1 and a2, respectively. Here myfun and mycon are 
  #   MATLAB file functions such as
  #   #        function f = myfun(x,a1)      
  #        f = x(1)^2 + a1*x(2)^2;       
  #                                      
  #        function [c,ceq] = mycon(x,a2)
  #        c = a2/x(1) - x(2);
  #        ceq = [];
  #   #   To optimize for specific values of a1 and a2, first assign the values 
  #   to these two parameters. Then create two one-argument anonymous 
  #   functions that capture the values of a1 and a2, and call myfun and 
  #   mycon with two arguments. Finally, pass these anonymous functions to 
  #   FMINCON:
  #   #        a1 = 2; a2 = 1.5; define parameters first
  #        options = optimoptions('fmincon','Algorithm','interior-point'); run interior-point algorithm
  #        x = fmincon(@(x) myfun(x,a1),[1;2],[],[],[],[],[],[],@(x) mycon(x,a2),options)
  #   #   See also OPTIMOPTIONS, OPTIMTOOL, FMINUNC, FMINBND, FMINSEARCH, @, FUNCTION_HANDLE.

  #    Copyright 1990-2015 The MathWorks, Inc.

  # ported to python by Lekan Ogunmolu
  # Date: August 06, 2017

  numberOfVariables = 1 # default. To be overidden
  numberOfEqualities = 1
  numberOfBounds = 1

  defaultopt = {
      'Algorithm': 'interior-point', 
      'AlwaysHonorConstraints': 'bounds', 
      'DerivativeCheck': 'off', 
      'Diagnostics': 'off', 
      'DiffMaxChange': np.inf, 
      'DiffMinChange': 0, 
      'Display': 'final', 
      'FinDiffRelStep': : np.inf,
      'FinDiffType': 'forward', 
      'FunValCheck': 'off', 
      'GradConstr': 'off', 
      'GradObj': 'off', 
      'HessFcn': np.inf,
      'Hessian': np.inf,
      'HessMult': np.inf,
      'HessPattern': spr.csr_matrix(np.ones((numberOfVariables, numberOfVariables))), 
      'InitBarrierParam': 0.1, 
      'InitTrustRegionRadius': np.sqrt(numberOfVariables), 
      'MaxFunEvals': np.inf,...
      'MaxIter': np.inf,...
      'MaxPCGIter': np.max([1, np.floor(numberOfVariables/2)]),
      'MaxProjCGIter': 2*(numberOfVariables-numberOfEqualities), 
      'MaxSQPIter': 10*np.max([numberOfVariables,numberOfInequalities+numberOfBounds]),
      'ObjectiveLimit': -1e20,
      'OutputFcn': np.inf,
      'PlotFcns': np.inf,
      'PrecondBandWidth',0,
      'RelLineSrchBnd': np.inf,
      'RelLineSrchBndDuration': 1, 
      'ScaleProblem': None, 
      'SubproblemAlgorithm': 'ldl-factorization', 
      'TolCon': 1e-6, 
      'TolConSQP': 1e-6, 
      'TolFun': 1e-6, 
      'TolFunValue': 1e-6, 
      'TolPCG': 0.1, 
      'TolProjCG': 1e-2, 
      'TolProjCGAbs': 1e-10, 
      'TolX': np.inf,
      'TypicalX': np.ones(numberOfVariables,1), 
      'UseParallel': False 
      }

  # If just 'defaults' passed in, return the default options in X
  nargin = len(args)
  nargout = _get_nargout()


  if nargin==1 and nargout <= 1 and FUN == 'defaults':
     X = defaultopt

  if nargin < 10:
      options = np.array(())
      if nargin < 9:
          NONLCON = np.array(())
          if nargin < 8:
              UB = np.array(())
              if nargin < 7:
                  LB = np.array(())
                  if nargin < 6:
                      Beq = np.array(())
                      if nargin < 5:
                          Aeq = np.array(())
                          if nargin < 4:
                              B = np.array(())
                              if nargin < 3:
                                  A = np.array(())

  if nargin == 1:
      if isinstance(FUN, dict):
          templist = [FUN,X,A,B,Aeq,Beq,LB,UB,NONLCON,options] 
          for i in range(len(templist)):
            templist[i] = Fun[str(templist[i])]
      else: 
          sys.stdout.write('optimlib:fmincon:InputArg')

  if nargout > 4:
     computeLambda = True
  else :
     computeLambda = False

  activeSet = 'active-set'
  sqp = 'sqp'
  trustRegionReflective = 'trust-region-reflective'
  interiorPoint = 'interior-point'
  sqpLegacy = 'sqp-legacy'

  XOUT = np.ravel(X)
  sizes = dict()
  sizes['xShape'] = X.shape;
  sizes['nVar'] = np.max(XOUT.shape);
  # Check for empty X
  if sizes['nVar'] == 0:
     sys.stdout.write('optimlib:fmincon:EmptyX')

  display = defaultopt['Display']
  # flags.detailedExitMsg = ~isempty(strfind(display,'detailed'));
  # switcher {    
  #   'off',
  #   'none'    verbosity = 0;
  #   'notify','notify-detailed'
  #       verbosity = 1;
  #   'final','final-detailed'
  #       verbosity = 2;
  #   'iter','iter-detailed'
  #       verbosity = 3;
  #   'testing'
  #       verbosity = 4;
  #   otherwise
  #       verbosity = 2;
  # } #display

  # % Set linear constraint right hand sides to column vectors
  # % (in particular, if empty, they will be made the correct
  # % size, 0-by-1)
  B = np.ravel(B)
  Beq = np.ravel(Beq)

  # % Check for consistency of linear constraints, before evaluating
  # % (potentially expensive) user functions 

  # % Set empty linear constraint matrices to the correct size, 0-by-n
  if not Aeq.size:
      Aeq = Aeq.reshape(0,sizes['nVar']);

  if not A.size:
      A = A.reshape(sizes['nVar']);   

  lin_eq, Aeqcol = Aeq.shape
  lin_ineq, Acol = A.shape
  # These sizes checks assume that empty matrices have already been made the correct size
  if Aeqcol != sizes['nVar']:
     sys.stdout.write('optimlib:fmincon:WrongNumberOfColumnsInAeq %d', sizes['nVar'])

  if lin_eq != length(Beq):
      sys.stdout.write('optimlib:fmincon:AeqAndBeqInconsistent')

  if Acol != sizes['nVar']:
      sys.stdout.write('optimlib:fmincon:WrongNumberOfColumnsInA %d', sizes['nVar'])

  if lin_ineq != np.max(B.shape):
       sys.stdout.write('optimlib:fmincon:AeqAndBinInconsistent')

  # End of linear constraint consistency check

  Algorithm = defaultopt['Algorithm'] 

  # Option needed for processing initial guess
  AlwaysHonorConstraints = defaultopt['AlwaysHonorConstraints'] 

  # Determine algorithm user chose via options. (We need this now
  # to set OUTPUT['algorithm'] in case of early termination due to 
  # inconsistent bounds.) 
  if (Algorithm != ('activeSet' or 'sqp' or 'trustRegionReflective' or 'interiorPoint' or 'sqpLegacy')):
      sys.stdout.write('optimlib:fmincon:InvalidAlgorithm')
  
  OUTPUT = dict()
  OUTPUT['algorithm'] = Algorithm
    
  XOUT,l,u,msg = checkbounds(XOUT,LB,UB,sizes['nVar']);
  if not msg.size:
     EXITFLAG = -2
     FVAL,LAMBDA,GRAD,HESSIAN = (np.array(()) for _ in range(4))
     
     OUTPUT['iterations'] = 0
     OUTPUT['funcCount'] = 0
     OUTPUT['stepsize'] = np.array(())
     if (OUTPUT['algorithm']=='activeSet') or (OUTPUT['algorithm']=='sqp')or (OUTPUT['algorithm']=='sqpLegacy'):
         OUTPUT['lssteplength'] = np.array(())
     else: #% trust-region-reflective, interior-point
         OUTPUT['cgiterations'] = np.array(())

     if (OUTPUT['algorithm']=='interiorPoint' or OUTPUT['algorithm']=='activeSet' or 
          OUTPUT['algorithm']=='sqp' or OUTPUT['algorithm']=='sqpLegacy' ):
          OUTPUT['constrviolation'] = np.array(())

     OUTPUT['firstorderopt'] = np.array(())
     OUTPUT['message'] = msg
     
     np.ravel(X) = XOUT
     # if verbosity > 0
     #    disp(msg)
     # end

  # % Get logical list of finite lower and upper bounds
  finDiffFlags = dict()
  finDiffFlags['hasLBs'] = np.isfinite(l);
  finDiffFlags['hasUBs'] = np.isfinite(u);

  lFinite = l[finDiffFlags['hasLBs']]
  uFinite = u[finDiffFlags['hasUBs']]

  # Create structure of flags and initial values, initialize merit function
  # type and the original shape of X.
  flags, initVals = dict(), dict()
  flags['meritFunction'] = 0;
  initVals['xOrigShape'] = X;

  diagnostics = defaultopt['Diagnostics'] = 'on' 
  funValCheck = defaultopt['FunValCheck'] = 'on' 
  derivativeCheck = defaultopt['DerivativeCheck'] = 'on' 

  # Gather options needed for finite differences
  # Write checked DiffMaxChange, DiffMinChage, FinDiffType, FinDiffRelStep,
  # GradObj and GradConstr options back into struct for later use
  options['DiffMinChange'] = defaultopt['DiffMinChange'] #optimget(options,'DiffMinChange',defaultopt,'fast');
  options['DiffMaxChange'] = defaultopt['DiffMaxChange'] #optimget(options,'DiffMaxChange',defaultopt,'fast');
  if options['DiffMinChange'] >= options['DiffMaxChange']:
      sys.stdout.write('optimlib:fmincon:DiffChangesInconsistent %0.5g, %0.5g', options['DiffMinChange'], options['DiffMaxChange'] )

  # Read in and error check option TypicalX
  typicalx = defaultopt['TypicalX'] = np.ones((sizes['nVar'], 1)) 

  # checkoptionsize('TypicalX', size(typicalx), sizes.nVar);
  options['TypicalX'] = typicalx
  options['FinDiffType'] = defaultopt['FinDiffType'] #optimget(options,'FinDiffType',defaultopt,'fast');
  # options = validateFinDiffRelStep(sizes.nVar,options,defaultopt);
  options['GradObj'] = defaultopt['GradObj'] 
  options['GradConstr'] = defaultopt['GradConstr']

  flags['grad'] = options['GradObj'] = 'on'

  # Notice that defaultopt.Hessian = [], so the variable "hessian" can be empty
  hessian = defaultopt['Hessian'] 
  # If calling trust-region-reflective with an unavailable Hessian option value, 
  # issue informative error message
  if ((OUTPUT['algorithm']=='trustRegionReflective') and \
          not ( hessian.size ) or (hessian=='on') or (hessian=='user-supplied') or \
             (hessian=='off') or (hessian=='fin-diff-grads')  ):
      print('optimlib:fmincon:BadTRReflectHessianValue')

  if ( not(hessian) and (hessian=='user-supplied') or (hessian=='on') ):
      flags['hess'] = True
  else:
      flags['hess'] = False

  if NONLCON:
     flags['constr'] = False
  else:
     flags['constr'] = True

  # Process objective function
  # if FUN:  # will detect empty string, empty matrix, empty cell array
  #    # constrflag in optimfcnchk set to false because we're checking the objective, not constraint
  #    funfcn = optimfcnchk(FUN,'fmincon',length(varargin),funValCheck,flags['grad'],flags['hess'],False,Algorithm);
  # else:
  #    sys.stdout.write('optimlib:fmincon:InvalidFUN')
  
  # Process constraint function
  # if flags.constr % NONLCON is non-empty
  #    flags.gradconst = strcmpi(options.GradConstr,'on');
  #    % hessflag in optimfcnchk set to false because hessian is never returned by nonlinear constraint 
  #    % function
  #    %
  #    % constrflag in optimfcnchk set to true because we're checking the constraints
  #    confcn = optimfcnchk(NONLCON,'fmincon',length(varargin),funValCheck,flags.gradconst,false,true);
  # else
  #    flags.gradconst = false; 
  #    confcn = {'','','','',''};
  # end

  rowAeq,colAeq = Aeq.shape

  if OUTPUT['algorithm']=='activeSet' or OUTPUT['algorithm']=='sqp' or OUTPUT['algorithm']=='sqpLegacy':
      # See if linear constraints are sparse and if user passed in Hessian
      if spr.issparse(Aeq) or spr.issparse(A):
          sys.stdout.write('optimlib:fmincon:ConvertingToFull %s', Algorithm)
      
      if flags['hess']: # conflicting options
          flags['hess'] = False
          sys.stdout.write('optimlib:fmincon:HessianIgnoredForAlg %s', Algorithm));
          # if strcmpi(funfcn{1},'fungradhess')
          #     funfcn{1}='fungrad';
          # elseif  strcmpi(funfcn{1},'fun_then_grad_then_hess')
          #     funfcn{1}='fun_then_grad';
          # end
  elif OUTPUT['algorithm']=='trustRegionReflective':
      # % Look at constraint type and supplied derivatives, and determine if
      # % trust-region-reflective can solve problem
      isBoundedNLP = NONLCON.size and A.size and Aeq.size  #problem has only bounds and no other constraints 
      isLinEqNLP = (NONLCON.size and A.size and lFinite.size 
                    and uFinite.size and colAeq) > rowAeq;
      if isBoundedNLP and flags['grad']:
        pass
          #if only l and u then call sfminbx
      elif isLinEqNLP and flags['grad']:
          # if only Aeq beq and Aeq has more columns than rows, then call sfminle
          pass
      else:
          # linkToDoc = addLink('Choosing the Algorithm', 'optim', 'helptargets.map', ...
          #                     'choose_algorithm', false);
          # if ((not isBoundedNLP) and (not isLinEqNLP)):
          #     print('optimlib:fmincon:ConstrTRR', linkToDoc))            
          # else
          #     % The user has a problem that satisfies the TRR constraint
          #     % restrictions but they haven't supplied gradients.
          #     error(message('optimlib:fmincon:GradOffTRR', linkToDoc))
          pass

  # % Process initial point 
  shiftedX0 = False # boolean that indicates if initial point was shifted
  if (OUTPUT['algorithm'] ==('activeSet' or 'sqp' or 'sqpLegacy')):
     if OUTPUT['algorithm']=='sqpLegacy':
         # Classify variables: finite lower bounds, finite upper bounds
         xIndices = classifyBoundsOnVars(l,u,sizes.nVar,false);
         # xIndices = NotImplemented
     
     # Check that initial point strictly satisfies the bounds on the variables.
     violatedLowerBnds_idx = XOUT[finDiffFlags['hasLBs']] < l[finDiffFlags['hasLBs']]
     violatedUpperBnds_idx = XOUT[finDiffFlags['hasUBs']] > u[finDiffFlags['hasUBs']]
     if np.any(violatedLowerBnds_idx) or np.any(violatedUpperBnds_idx):
         finiteLbIdx = np.nonzero(finDiffFlags['hasLBs'])
         finiteUbIdx = np.nonzero(finDiffFlags['hasUBs'])
         XOUT[finiteLbIdx[violatedLowerBnds_idx]] = l[finiteLbIdx[violatedLowerBnds_idx]];
         XOUT[finiteUbIdx[violatedUpperBnds_idx]] = u[finiteUbIdx[violatedUpperBnds_idx]];
         np.ravel(X) = XOUT
         shiftedX0 = True
  elif OUTPUT['algorithm']=='trustRegionReflective':
     #
     # If components of initial x not within bounds, set those components  
     # of initial point to a "box-centered" point
     #
     if not Aeq.size:
         arg = (u >= 1e10); arg2 = (l <= -1e10);
         u[arg] = np.inf;
         l[arg2] = -np.inf;
         xinitOutOfBounds_idx = np.logical_or(XOUT < l | XOUT > u)
         if np.any(xinitOutOfBounds_idx):
             shiftedX0 = True
             XOUT = startx(u,l,XOUT,xinitOutOfBounds_idx);
             np.ravel(X) = XOUT
     else:
        # Phase-1 for sfminle nearest feas. pt. to XOUT. Don't print a
        # message for this change in X0 for sfminle. 
         XOUT = NotImplemented #feasibl(Aeq,Beq,XOUT)
         np.ravel(X) = XOUT
    
  elif OUTPUT['algorithm'] == 'interiorPoint':
      # Variables: fixed, finite lower bounds, finite upper bounds
      xIndices = classifyBoundsOnVars(l,u,sizes.nVar,true);

      # If honor bounds mode, then check that initial point strictly satisfies the
      # simple inequality bounds on the variables and exactly satisfies fixed variable
      # bounds.
      if (AlwaysHonorConstraints=='bounds') or (AlwaysHonorConstraints=='bounds-ineqs'):
        violatedFixedBnds_idx = XOUT[xIndices['fixed'] != l[xIndices['fixed']]
        violatedLowerBnds_idx = XOUT[xIndices['finiteLb'] <= l[xIndices['finiteLb']]
        violatedUpperBnds_idx = XOUT[xIndices['finiteUb'] >= u[xIndices['finiteUb']]
        if np.any(violatedLowerBnds_idx) or np.any(violatedUpperBnds_idx) or np.any(violatedFixedBnds_idx):
          XOUT = shiftInitPtToInterior(sizes['nVar'], XOUT, l, u, np.inf);
          np.ravel(X) = XOUT;
          shiftedX0 = True;

  % Display that x0 was shifted in order to honor bounds
  if shiftedX0
      if verbosity >= 3
          if strcmpi(OUTPUT.algorithm,interiorPoint) 
              fprintf(getString(message('optimlib:fmincon:ShiftX0StrictInterior')));
              fprintf('\n');
          else
              fprintf(getString(message('optimlib:fmincon:ShiftX0ToBnds')));
              fprintf('\n');
          end
      end
  end
      
  % Evaluate function
  initVals.g = zeros(sizes.nVar,1);
  HESSIAN = []; 

  switch funfcn{1}
  case 'fun'
     try
        initVals.f = feval(funfcn{3},X,varargin{:});
     catch userFcn_ME
          optim_ME = MException('optimlib:fmincon:ObjectiveError', ...
              getString(message('optimlib:fmincon:ObjectiveError')));
          userFcn_ME = addCause(userFcn_ME,optim_ME);
          rethrow(userFcn_ME)
     end
  case 'fungrad'
     try
        [initVals.f,initVals.g] = feval(funfcn{3},X,varargin{:});
     catch userFcn_ME
          optim_ME = MException('optimlib:fmincon:ObjectiveError', ...
              getString(message('optimlib:fmincon:ObjectiveError')));
          userFcn_ME = addCause(userFcn_ME,optim_ME);
          rethrow(userFcn_ME)
     end
  case 'fungradhess'
     try
        [initVals.f,initVals.g,HESSIAN] = feval(funfcn{3},X,varargin{:});
     catch userFcn_ME
          optim_ME = MException('optimlib:fmincon:ObjectiveError', ...
              getString(message('optimlib:fmincon:ObjectiveError')));
          userFcn_ME = addCause(userFcn_ME,optim_ME);
          rethrow(userFcn_ME)
     end
  case 'fun_then_grad'
     try
        initVals.f = feval(funfcn{3},X,varargin{:});
     catch userFcn_ME
          optim_ME = MException('optimlib:fmincon:ObjectiveError', ...
              getString(message('optimlib:fmincon:ObjectiveError')));
          userFcn_ME = addCause(userFcn_ME,optim_ME);
          rethrow(userFcn_ME)
     end
     try
        initVals.g = feval(funfcn{4},X,varargin{:});
     catch userFcn_ME
          optim_ME = MException('optimlib:fmincon:GradientError', ...
              getString(message('optimlib:fmincon:GradientError')));
          userFcn_ME = addCause(userFcn_ME,optim_ME);
          rethrow(userFcn_ME)
     end
  case 'fun_then_grad_then_hess'
     try
        initVals.f = feval(funfcn{3},X,varargin{:});
     catch userFcn_ME
          optim_ME = MException('optimlib:fmincon:ObjectiveError', ...
              getString(message('optimlib:fmincon:ObjectiveError')));
          userFcn_ME = addCause(userFcn_ME,optim_ME);
          rethrow(userFcn_ME)
     end
     try
        initVals.g = feval(funfcn{4},X,varargin{:});
     catch userFcn_ME
          optim_ME = MException('optimlib:fmincon:GradientError', ...
              getString(message('optimlib:fmincon:GradientError')));
          userFcn_ME = addCause(userFcn_ME,optim_ME);
          rethrow(userFcn_ME)
     end
     try
        HESSIAN = feval(funfcn{5},X,varargin{:});
     catch userFcn_ME
          optim_ME = MException('optimlib:fmincon:HessianError', ...
              getString(message('optimlib:fmincon:HessianError')));            
          userFcn_ME = addCause(userFcn_ME,optim_ME);
          rethrow(userFcn_ME)
     end
  otherwise
     error(message('optimlib:fmincon:UndefinedCallType'));
  end

  % Check that the objective value is a scalar
  if numel(initVals.f) ~= 1
     error(message('optimlib:fmincon:NonScalarObj'))
  end

  % Check that the objective gradient is the right size
  initVals.g = initVals.g(:);
  if numel(initVals.g) ~= sizes.nVar
     error('optimlib:fmincon:InvalidSizeOfGradient', ...
         getString(message('optimlib:commonMsgs:InvalidSizeOfGradient',sizes.nVar)));
  end

  % Evaluate constraints
  switch confcn{1}
  case 'fun'
      try
          [ctmp,ceqtmp] = feval(confcn{3},X,varargin{:});
      catch userFcn_ME
          if strcmpi('MATLAB:maxlhs',userFcn_ME.identifier)
                  error(message('optimlib:fmincon:InvalidHandleNonlcon'))
          else
              optim_ME = MException('optimlib:fmincon:NonlconError', ...
                  getString(message('optimlib:fmincon:NonlconError')));
              userFcn_ME = addCause(userFcn_ME,optim_ME);
              rethrow(userFcn_ME)
          end
      end
      initVals.ncineq = ctmp(:);
      initVals.nceq = ceqtmp(:);
      initVals.gnc = zeros(sizes.nVar,length(initVals.ncineq));
      initVals.gnceq = zeros(sizes.nVar,length(initVals.nceq));
  case 'fungrad'
     try
        [ctmp,ceqtmp,initVals.gnc,initVals.gnceq] = feval(confcn{3},X,varargin{:});
     catch userFcn_ME
         optim_ME = MException('optimlib:fmincon:NonlconError', ...
             getString(message('optimlib:fmincon:NonlconError')));           
         userFcn_ME = addCause(userFcn_ME,optim_ME);
         rethrow(userFcn_ME)
     end
     initVals.ncineq = ctmp(:);
     initVals.nceq = ceqtmp(:);
  case 'fun_then_grad'
      try
          [ctmp,ceqtmp] = feval(confcn{3},X,varargin{:});
      catch userFcn_ME
          optim_ME = MException('optimlib:fmincon:NonlconError', ...
              getString(message('optimlib:fmincon:NonlconError')));
          userFcn_ME = addCause(userFcn_ME,optim_ME);
          rethrow(userFcn_ME)
      end
      initVals.ncineq = ctmp(:);
      initVals.nceq = ceqtmp(:);
      try
          [initVals.gnc,initVals.gnceq] = feval(confcn{4},X,varargin{:});
      catch userFcn_ME
          optim_ME = MException('optimlib:fmincon:NonlconFunOrGradError', ...
              getString(message('optimlib:fmincon:NonlconFunOrGradError')));
          userFcn_ME = addCause(userFcn_ME,optim_ME);
          rethrow(userFcn_ME)
      end
  case ''
     % No nonlinear constraints. Reshaping of empty quantities is done later
     % in this file, where both cases, (i) no nonlinear constraints and (ii)
     % nonlinear constraints that have one type missing (equalities or
     % inequalities), are handled in one place
     initVals.ncineq = [];
     initVals.nceq = [];
     initVals.gnc = [];
     initVals.gnceq = [];
  otherwise
     error(message('optimlib:fmincon:UndefinedCallType'));
  end

  % Check for non-double data typed values returned by user functions 
  if ~isempty( isoptimargdbl('FMINCON', {'f','g','H','c','ceq','gc','gceq'}, ...
     initVals.f, initVals.g, HESSIAN, initVals.ncineq, initVals.nceq, initVals.gnc, initVals.gnceq) )
      error('optimlib:fmincon:NonDoubleFunVal',getString(message('optimlib:commonMsgs:NonDoubleFunVal','FMINCON')));
  end

  sizes.mNonlinEq = length(initVals.nceq);
  sizes.mNonlinIneq = length(initVals.ncineq);

  % Make sure empty constraint and their derivatives have correct sizes (not 0-by-0):
  if isempty(initVals.ncineq)
      initVals.ncineq = reshape(initVals.ncineq,0,1);
  end
  if isempty(initVals.nceq)
      initVals.nceq = reshape(initVals.nceq,0,1);
  end
  if isempty(initVals.gnc)
      initVals.gnc = reshape(initVals.gnc,sizes.nVar,0);
  end
  if isempty(initVals.gnceq)
      initVals.gnceq = reshape(initVals.gnceq,sizes.nVar,0);
  end
  [cgrow,cgcol] = size(initVals.gnc);
  [ceqgrow,ceqgcol] = size(initVals.gnceq);

  if cgrow ~= sizes.nVar || cgcol ~= sizes.mNonlinIneq
     error(message('optimlib:fmincon:WrongSizeGradNonlinIneq', sizes.nVar, sizes.mNonlinIneq))
  end
  if ceqgrow ~= sizes.nVar || ceqgcol ~= sizes.mNonlinEq
     error(message('optimlib:fmincon:WrongSizeGradNonlinEq', sizes.nVar, sizes.mNonlinEq))
  end

  if diagnostics
     % Do diagnostics on information so far
     diagnose('fmincon',OUTPUT,flags.grad,flags.hess,flags.constr,flags.gradconst,...
        XOUT,sizes.mNonlinEq,sizes.mNonlinIneq,lin_eq,lin_ineq,l,u,funfcn,confcn);
  end

  % Create default structure of flags for finitedifferences:
  % This structure will (temporarily) ignore some of the features that are
  % algorithm-specific (e.g. scaling and fault-tolerance) and can be turned
  % on later for the main algorithm.
  finDiffFlags.fwdFinDiff = strcmpi(options.FinDiffType,'forward');
  finDiffFlags.scaleObjConstr = false; % No scaling for now
  finDiffFlags.chkFunEval = false;     % No fault-tolerance yet
  finDiffFlags.chkComplexObj = false;  % No need to check for complex values
  finDiffFlags.isGrad = true;          % Scalar objective


  % For parallel finite difference (if needed) we need to send the function
  % handles now to the workers. This avoids sending the function handles in
  % every iteration of the solver. The output from 'setOptimFcnHandleOnWorkers' 
  % is a onCleanup object that will perform cleanup task on the workers.
  UseParallel = optimget(options,'UseParallel',defaultopt,'fast');
  cleanupObj = setOptimFcnHandleOnWorkers(UseParallel,funfcn,confcn); %#ok<NASGU>

  % Check derivatives
  if derivativeCheck && ...               % User wants to check derivatives...
     (flags.grad || ...                   % of either objective or ...
     flags.gradconst && sizes.mNonlinEq+sizes.mNonlinIneq > 0) % nonlinear constraint function.
      validateFirstDerivatives(funfcn,confcn,X, ...
          l,u,options,finDiffFlags,sizes,varargin{:});
  end

  % call algorithm
  if strcmpi(OUTPUT.algorithm,activeSet) % active-set
      defaultopt.MaxIter = 400; defaultopt.MaxFunEvals = '100*numberofvariables'; defaultopt.TolX = 1e-6;
      defaultopt.Hessian = 'off';
      problemInfo = []; % No problem related data
      [X,FVAL,LAMBDA,EXITFLAG,OUTPUT,GRAD,HESSIAN]=...
          nlconst(funfcn,X,l,u,full(A),B,full(Aeq),Beq,confcn,options,defaultopt, ...
          finDiffFlags,verbosity,flags,initVals,problemInfo,optionFeedback,varargin{:});
  elseif strcmpi(OUTPUT.algorithm,trustRegionReflective) % trust-region-reflective
     if (strcmpi(funfcn{1}, 'fun_then_grad_then_hess') || strcmpi(funfcn{1}, 'fungradhess'))
        Hstr = [];
     elseif (strcmpi(funfcn{1}, 'fun_then_grad') || strcmpi(funfcn{1}, 'fungrad'))
        n = length(XOUT); 
        Hstr = optimget(options,'HessPattern',defaultopt,'fast');
        if ischar(Hstr) 
           if strcmpi(Hstr,'sparse(ones(numberofvariables))')
              Hstr = sparse(ones(n));
           else
              error(message('optimlib:fmincon:InvalidHessPattern'))
           end
        end
        checkoptionsize('HessPattern', size(Hstr), n);
     end
     
     defaultopt.MaxIter = 400; defaultopt.MaxFunEvals = '100*numberofvariables'; defaultopt.TolX = 1e-6;
     defaultopt.Hessian = 'off';
     % Trust-region-reflective algorithm does not compute constraint
     % violation as it progresses. If the user requests the output structure,
     % we need to calculate the constraint violation at the returned
     % solution.
     if nargout > 3
         computeConstrViolForOutput = true;
     else
         computeConstrViolForOutput = false;
     end

     if isempty(Aeq)
        [X,FVAL,LAMBDA,EXITFLAG,OUTPUT,GRAD,HESSIAN] = ...
           sfminbx(funfcn,X,l,u,verbosity,options,defaultopt,computeLambda,initVals.f,initVals.g, ...
           HESSIAN,Hstr,flags.detailedExitMsg,computeConstrViolForOutput,optionFeedback,varargin{:});
     else
        [X,FVAL,LAMBDA,EXITFLAG,OUTPUT,GRAD,HESSIAN] = ...
           sfminle(funfcn,X,sparse(Aeq),Beq,verbosity,options,defaultopt,computeLambda,initVals.f, ...
           initVals.g,HESSIAN,Hstr,flags.detailedExitMsg,computeConstrViolForOutput,optionFeedback,varargin{:});
     end
  elseif strcmpi(OUTPUT.algorithm,interiorPoint)
      defaultopt.MaxIter = 1000; defaultopt.MaxFunEvals = 3000; defaultopt.TolX = 1e-10;
      defaultopt.Hessian = 'bfgs';
      mEq = lin_eq + sizes.mNonlinEq + nnz(xIndices.fixed); % number of equalities
      % Interior-point-specific options. Default values for lbfgs memory is 10, and 
      % ldl pivot threshold is 0.01
      options = getIpOptions(options,sizes.nVar,mEq,flags.constr,defaultopt,10,0.01); 

      [X,FVAL,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = barrier(funfcn,X,A,B,Aeq,Beq,l,u,confcn,options.HessFcn, ...
          initVals.f,initVals.g,initVals.ncineq,initVals.nceq,initVals.gnc,initVals.gnceq,HESSIAN, ...
          xIndices,options,optionFeedback,finDiffFlags,varargin{:});
  elseif strcmpi(OUTPUT.algorithm,sqp)
      defaultopt.MaxIter = 400; defaultopt.MaxFunEvals = '100*numberofvariables'; 
      defaultopt.TolX = 1e-6; defaultopt.Hessian = 'bfgs';
      % Validate options used by sqp
      options = getSQPOptions(options,defaultopt,sizes.nVar);
      optionFeedback.detailedExitMsg = flags.detailedExitMsg;    
      % Call algorithm
      [X,FVAL,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = sqpInterface(funfcn,X,full(A),full(B),full(Aeq),full(Beq), ...
          full(l),full(u),confcn,initVals.f,full(initVals.g),full(initVals.ncineq),full(initVals.nceq), ...
          full(initVals.gnc),full(initVals.gnceq),sizes,options,finDiffFlags,verbosity,optionFeedback,varargin{:});
  else % sqpLegacy
      defaultopt.MaxIter = 400; defaultopt.MaxFunEvals = '100*numberofvariables'; 
      defaultopt.TolX = 1e-6; defaultopt.Hessian = 'bfgs';
      % Validate options used by sqp
      options = getSQPOptions(options,defaultopt,sizes.nVar);
      optionFeedback.detailedExitMsg = flags.detailedExitMsg;
      % Call algorithm
      [X,FVAL,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = sqpLineSearch(funfcn,X,full(A),full(B),full(Aeq),full(Beq), ...
          full(l),full(u),confcn,initVals.f,full(initVals.g),full(initVals.ncineq),full(initVals.nceq), ...
          full(initVals.gnc),full(initVals.gnceq),xIndices,options,finDiffFlags,verbosity,optionFeedback,varargin{:});
  end

  % Force a cleanup of the handle object. Sometimes, MATLAB may
  % delay the cleanup but we want to be sure it is cleaned up.
  clear cleanupObj

return (X,FVAL,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN)