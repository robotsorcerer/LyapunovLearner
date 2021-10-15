import numpy as np
import matplotlib.pyplot as plt

def energyContour(Vxf, D, cost_obj, *args):
    """
        
    """
    quality='low'
    b_plot_contour = True
    contour_levels = np.array([])

    if quality == 'high':
        nx, ny = 0.1, 0.1
    elif quality == 'medium':
        nx, ny = 1, 1
    else:
        nx, ny = 2, 2

    x = np.arange(D[0], D[1], nx)
    y = np.arange(D[2], D[3], ny)
    x_len = len(x)
    y_len = len(y)
    X, Y = np.meshgrid(x, y)
    x = np.stack([np.ravel(X), np.ravel(Y)])

    V, dV = cost_obj.computeEnergy(x, np.array(()), Vxf, nargout=2)

    if not contour_levels.size:
        contour_levels = np.arange(0, np.log(np.max(V)), 0.5)
        contour_levels = np.exp(contour_levels)
        if np.max(V) > 40:
            contour_levels = np.round(contour_levels)

    V = V.reshape(y_len, x_len)

    if b_plot_contour:
        h = plt.contour(X, Y, V, contour_levels, colors='k', origin='upper', linewidths=2, labelspacing=200)

    return h
