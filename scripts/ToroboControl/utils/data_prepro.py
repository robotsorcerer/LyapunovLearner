from __future__ import print_function
import os
import numpy as np
from os.path import join, expanduser


def format_data(filepath, name, learn_type='2d'):
    """
        filepath:      path to file that was saved with kdl c++ parser
        name:          name of the file in question
        learn_type:    the dimension of the data that we are using
                       to learn the control lyapunov function

        Returns: 2D Numpy array data
    """
    filename = join(filepath, name)

    with open(filename, 'r+') as foo:
        data = foo.readlines()

    # gather data into right shape
    proper_data = []

    for i in range(len(data)):
        temp = data[i].rsplit(',')
        temp[-1] = temp[-1].rsplit('\n')[0]
        temp = [float(x) for x in temp]
        proper_data.append(temp)

    proper_data = np.array(proper_data)

    assert proper_data.ndim == 2, "data shape needs to be 2d"

    if learn_type == '2d':
        data2d = np.hstack([proper_data[:, :2], proper_data[:, 3:5]])

        # data2d returned as [x1, y1, x1d, y1d].T eh to"" because control only works in 2d
        return data2d.T, proper_data.T

    elif learn_type == '6d':
        # data6d returned as [x1, y1, z1, x1d, y1d, z1d].T eh to
        return proper_data.T

    else:
        print("learning type not understood")
