import numpy as np

def optimfcnchk(funstr,caller,lenVarIn,funValCheck, 
                gradflag,hessflag,constrflag,Algorithm,ntheta, nargin=9):
# 

# OPTIMFCNCHK Pre- and post-process function expression for FCNCHK.
# 
#  This is a helper function.

#    [ALLFCNS,idandmsg] = OPTIMFCNCHK(FUNSTR,CALLER,lenVarIn,GRADFLAG) takes
#    the (nonempty) function handle or expression FUNSTR from CALLER with
#    lenVarIn extra arguments, parses it according to what CALLER is, then
#    returns a string or inline object in ALLFCNS.  If an error occurs,
#    this message is put in idandmsg.
# 
#    ALLFCNS is a cell array:
#    ALLFCNS{1} contains a flag
#    that says if the objective and gradients are together in one function
#    (calltype=='fungrad') or in two functions (calltype='fun_then_grad')
#    or there is no gradient (calltype=='fun'), etc.
#    ALLFCNS{2} contains the string CALLER.
#    ALLFCNS{3}  contains the objective (or constraint) function
#    ALLFCNS{4}  contains the gradient function
#    ALLFCNS{5}  contains the hessian function (not used for constraint function).
# 
#    If funValCheck is 'on', then we update the funfcn's (fun/grad/hess) so
#    they are called through CHECKFUN to check for NaN's, Inf's, or complex
#    values. Add a wrapper function, CHECKFUN, to check for NaN/complex
#    values without having to change the calls that look like this:
#    f = funfcn(x,varargin{:});
#    CHECKFUN is a nested function so we can get the 'caller', 'userfcn', and
#    'ntheta' (for fseminf constraint functions) information from this function
#    and the user's function and CHECKFUN can both be called with the same
#    arguments.
#    NOTE: we assume FUNSTR is nonempty.

#    Copyright 1990-2015 The MathWorks, Inc.

    #  Initialize
    if nargin < 9:
        ntheta = 0
        if nargin < 8:
            Algorithm = np.array(())
            if nargin < 7:
                constrflag = False;
                if nargin < 6:
                    hessflag = False;
                    if nargin < 5:
                        gradflag = False;

    if caller == 'fseminf':
        nonlconType = 'SEMINFCON'
    else:
        nonlconType = 'NONLCON'


    allfcns = {}
    gradfcn = np.array(())
    hessfcn = [np.array(())
    if gradflag and hessflag:
        if caller=='fmincon' and Algorithm=='interior-point':
            # fmincon interior-point doesn't take Hessian as 3rd argument 
            # of objective function - it's passed as a separate function
            calltype = 'fungrad'
        else:
            calltype = 'fungradhess'
    elif gradflag:
        calltype = 'fungrad';
    else: # ~gradflag & ~hessflag,   OR  ~gradflag & hessflag: this problem handled later
        calltype = 'fun'


    if ~isa(funstr,'cell')  
        # The documented syntax (and most common):
        # funstr is a function handle, string expression, function name string or inline object
        [funfcn, idandmsg] = fcnchk(funstr,lenVarIn); %#ok<DFCNCHK>
        if funValCheck:
            userfcn = funfcn;
            funfcn = @checkfun; #caller and userfcn are in scope in nested checkfun
        

        if ~isempty(idandmsg)
            if constrflag
                error(message('optimlib:optimfcnchk:ConstrNotAFunction', nonlconType));
            else
                error(message(idandmsg.identifier));
            end
        end
        if gradflag % gradient and function in one function/MATLAB file
            gradfcn = funfcn; # Do this so graderr will print the correct name
        end
        if hessflag and not gradflag:
            sys.stdout.write('optimlib:optimfcnchk:GradientOptionOffFunstrNotACell')
    else:
        # Input funstr is a cell-array of function handles/strings/inlines
        # NOTE: it is assumed that the first cell/function is nonempty. These
        # checks are performed in the caller.
        numFunctions = length(funstr);
        
        % Determine derivative level given by the user
        calltype = 'fun';
     
        if numFunctions > 1 && ~isempty(funstr{2})
            calltype = [calltype '_then_grad'];
        end
        if numFunctions > 2 && ~isempty(funstr{3})
            calltype = [calltype '_then_hess'];
            % Error if the user gives this input: {fun, [], hess}
            if isempty(funstr{2}) 
                error(message('optimlib:optimfcnchk:NoGradientWithHessian'))
            end
        end        
        
        % NOTE: when the user passes more than 3 cells/functions, we silently
        % ignore anything passed the 3rd cell. Previously, we would error with
        % the message "FUN must be a function handle".
        
    	% Error if gradient flag is on, but no gradient function supplied
        if gradflag && strcmpi(calltype,'fun')    
            if constrflag
                error( message('optimlib:optimfcnchk:NoConstraintGradientFunction', ...
                    '(OPTIONS.SpecifyConstraintGradient = true)') );
            else
                error( message('optimlib:optimfcnchk:NoGradientFunction', ...
                    '(OPTIONS.SpecifyObjectiveGradient = true)') );
            end
        end
        
        % Error if the Hessian flag is on, but no Hessian is supplied
        % TODO: why only throw when the number of cells is 3? Why not
        % anytime hessflag is on and there is no Hessian function?
        if hessflag && ( numFunctions == 3 && isempty(funstr{3}) )
            error(message('optimlib:optimfcnchk:NoHessianFunction'))
        end    
        
        % Do not evaluate the gradient if user gave a gradient function, but
        % turned gradient option off or if the user gave a Hessian function but
        % turned the Hessian option off. 
        % NOTE: these were accompanied by warnings, but this syntax is
        % undocumented and so the warning is not needed. Also, no warning is
        % issued in the same scenario when the gradient/Hessian are calculated
        % inside the objective/constraint function
        if ~gradflag && strncmpi(calltype,'fun_then_grad',13)
            calltype = 'fun';
        elseif ~hessflag && strcmpi(calltype,'fun_then_grad_then_hess')
            calltype = 'fun_then_grad';
        end

        % Check the objective: the first element of the cell-array
        [funfcn, idandmsg] = fcnchk(funstr{1},lenVarIn); %#ok<DFCNCHK>
        % Insert call to nested function checkfun which calls user funfcn
        if funValCheck
            userfcn = funfcn;
            funfcn = @checkfun; %caller and userfcn are in scope in nested checkfun
        end
        if ~isempty(idandmsg)
            if constrflag % Constraint, not objective, function, so adjust error message
                error(message('optimlib:optimfcnchk:ConstrNotAFunction', nonlconType));
            else
                error(message(idandmsg.identifier));
            end
        end
        
        if strncmpi(calltype,'fun_then_grad',13)
            %  {fun, grad} OR {fun, grad, []}
            [gradfcn, idandmsg] = fcnchk(funstr{2},lenVarIn); %#ok<DFCNCHK>
            if funValCheck
                userfcn = gradfcn;
                gradfcn = @checkfun; %caller and userfcn are in scope in nested checkfun
            end
            if ~isempty(idandmsg)
                if constrflag
                    error(message('optimlib:optimfcnchk:ConstrNotAFunction', nonlconType));
                else
                    error(message(idandmsg.identifier));
                end
            end

        end
        
        if strcmpi(calltype,'fun_then_grad_then_hess')
            % {fun, grad, hess}
            [hessfcn, idandmsg] = fcnchk(funstr{3},lenVarIn); %#ok<DFCNCHK>
            if funValCheck
                userfcn = hessfcn;
                hessfcn = @checkfun; %caller and userfcn are in scope in nested checkfun
            end
            
            if ~isempty(idandmsg)
                if constrflag
                    error(message('optimlib:optimfcnchk:ConstrNotAFunction', nonlconType));
                else
                    error(message(idandmsg.identifier));
                end
            end
        end
    end
    allfcns{1} = calltype;
    allfcns{2} = caller;
    allfcns{3} = funfcn;
    allfcns{4} = gradfcn;
    allfcns{5} = hessfcn;

        %------------------------------------------------------------
        function [varargout] = checkfun(x,varargin)
        % CHECKFUN checks for complex, Inf, or NaN results from userfcn.
        % Inputs CALLER, USERFCN, and NTHETA come from the scope of OPTIMFCNCHK.
        % We do not make assumptions about f, g, or H. For generality, assume
        % they can all be matrices.
       
            if nargout == 1
                f = userfcn(x,varargin{:});
                if any(any(isnan(f)))
                    error(message('optimlib:optimfcnchk:checkfun:NaNFval', functiontostring( userfcn ), upper( caller )));
                elseif ~isreal(f)
                    error(message('optimlib:optimfcnchk:checkfun:ComplexFval', functiontostring( userfcn ), upper( caller )));
                elseif any(any(isinf(f)))
                    error(message('optimlib:optimfcnchk:checkfun:InfFval', functiontostring( userfcn ), upper( caller )));
                else
                    varargout{1} = f;
                end

            elseif nargout == 2 % Two output could be f,g (from objective fcn) or c,ceq (from NONLCON)
                [f,g] = userfcn(x,varargin{:});
                if any(any(isnan(f))) || any(any(isnan(g)))
                    error(message('optimlib:optimfcnchk:checkfun:NaNFval', functiontostring( userfcn ), upper( caller )));
                elseif ~isreal(f) || ~isreal(g)
                    error(message('optimlib:optimfcnchk:checkfun:ComplexFval', functiontostring( userfcn ), upper( caller )));
                elseif any(any(isinf(f))) || any(any(isinf(g)))
                    error(message('optimlib:optimfcnchk:checkfun:InfFval', functiontostring( userfcn ), upper( caller )));
                else
                    varargout{1} = f;
                    varargout{2} = g;
                end

            elseif nargout == 3 % This case only happens for objective functions
                [f,g,H] = userfcn(x,varargin{:});
                if any(any(isnan(f))) || any(any(isnan(g))) || any(any(isnan(H)))
                    error(message('optimlib:optimfcnchk:checkfun:NaNFval', functiontostring( userfcn ), upper( caller )));
                elseif ~isreal(f) || ~isreal(g) || ~isreal(H)
                    error(message('optimlib:optimfcnchk:checkfun:ComplexFval', functiontostring( userfcn ), upper( caller )));
                elseif any(any(isinf(f))) || any(any(isinf(g))) || any(any(isinf(H)))
                    error(message('optimlib:optimfcnchk:checkfun:InfFval', functiontostring( userfcn ), upper( caller )));
                else
                    varargout{1} = f;
                    varargout{2} = g;
                    varargout{3} = H;
                end
            elseif nargout == 4 && ~strcmpi(caller,'fseminf')
                % In this case we are calling NONLCON, e.g. for FMINCON, and
                % the outputs are [c,ceq,gc,gceq]
                [c,ceq,gc,gceq] = userfcn(x,varargin{:}); 
                if any(any(isnan(c))) || any(any(isnan(ceq))) || any(any(isnan(gc))) || any(any(isnan(gceq)))
                    error(message('optimlib:optimfcnchk:checkfun:NaNFval', functiontostring( userfcn ), upper( caller )));
                elseif ~isreal(c) || ~isreal(ceq) || ~isreal(gc) || ~isreal(gceq)
                    error(message('optimlib:optimfcnchk:checkfun:ComplexFval', functiontostring( userfcn ), upper( caller )));
                elseif any(any(isinf(c))) || any(any(isinf(ceq))) || any(any(isinf(gc))) || any(any(isinf(gceq))) 
                    error(message('optimlib:optimfcnchk:checkfun:InfFval', functiontostring( userfcn ), upper( caller )));
                else
                    varargout{1} = c;
                    varargout{2} = ceq;
                    varargout{3} = gc;
                    varargout{4} = gceq;
                end
            else % fseminf constraints have a variable number of outputs, but at 
                 % least 4: see semicon.m
                % Also, don't check 's' for NaN as NaN is a valid value
                T = cell(1,ntheta);
                [c,ceq,T{:},s] = userfcn(x,varargin{:});
                nanfound = any(any(isnan(c))) || any(any(isnan(ceq)));
                complexfound = ~isreal(c) || ~isreal(ceq) || ~isreal(s);
                inffound = any(any(isinf(c))) || any(any(isinf(ceq))) || any(any(isinf(s)));
                for ii=1:length(T) % Elements of T are matrices
                    if nanfound || complexfound || inffound
                        break
                    end
                    nanfound = any(any(isnan(T{ii})));
                    complexfound = ~isreal(T{ii});
                    inffound = any(any(isinf(T{ii})));
                end
                if nanfound
                    error(message('optimlib:optimfcnchk:checkfun:NaNFval', functiontostring( userfcn ), upper( caller )));
                elseif complexfound
                    error(message('optimlib:optimfcnchk:checkfun:ComplexFval', functiontostring( userfcn ), upper( caller )));
                elseif inffound
                    error(message('optimlib:optimfcnchk:checkfun:InfFval', functiontostring( userfcn ), upper( caller )));
                else
                    varargout{1} = c;
                    varargout{2} = ceq;
                    varargout(3:ntheta+2) = T;
                    varargout{ntheta+3} = s;
    return allfcns,idandmsg