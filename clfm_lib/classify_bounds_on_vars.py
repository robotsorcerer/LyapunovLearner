import numpy as np

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
