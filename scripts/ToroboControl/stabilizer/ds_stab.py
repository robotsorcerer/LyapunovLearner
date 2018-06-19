import numpy as np
from inspect import getfullargspec
from cost.cost import computeEnergy

def dsStabilizer(X, fn_handle, Vxf, rho0, kappa0):
    d = Vxf['d']
    if X.shape[0] == 2*d:
        Xd     = X[d:2*d,:]
        X      = X[:d,:]
    else:
        if (len(getfullargspec(fn_handle).args) == 1):
            Xd, _, _ = fn_handle(X)
        elif (len(getfullargspec(fn_handle).args) == 2):
            t  = X[d,:]
            X  = X[d:]
            Xd, _, _ = fn_handle(t,X)
        else:
            logger.CRITICAL('Unknown function handle!')

    V,Vx    = computeEnergy(X,[],Vxf)

    norm_Vx = np.sum(V ** 2, axis=0)
    norm_x  = np.sum(X ** 2,axis=0)

    Vdot    = np.sum(Vx * Xd,axis=0)
    rho     = rho0 * (1-np.exp(-kappa0*norm_x)) * np.sqrt(norm_Vx)
    ind     = (Vdot + rho) >= 0
    u       = Xd * 0

    print('Vdot: ', Vdot)
    
    if np.sum(ind)>0:
        lambder   = (Vdot[ind] + rho[ind]) / norm_Vx[ind]
        u[:,ind]  = -np.tile(lambder,[d,1]) * Vx[:,ind]
        Xd[:,ind] = Xd[:,ind] + u[:,ind]

#     if args:
#         dt = args[0]
#         xn = X + np.dot(Xd, dt)
#         Vn = computeEnergy(xn,[],Vxf)
#         ind = (Vn >= V)
#         i = 0

#         while(np.any(ind) and i < 10):
#             alpha = np.divide(V[ind], Vn[ind])
#             Xd[:,ind] = np.tile(alpha,[d,1]) * Xd[:,ind] - \
#                         np.tile(alpha * np.sum(Xd[:,ind] * \
#                         Vx[:,ind], axis=0)/norm_Vx[ind],[d,1])*Vx[:,ind]
#             xn = X + np.dot(Xd,dt)
#             Vn = computeEnergy(xn,np.array(()),Vxf)
#             ind = (Vn >= V)
#             i = i + 1

    return Xd, u
