import numpy as np


def forward_kinematics(theta):
    (c1, c2, c3, c4, c5, c6, c7) = np.cos(np.radians(theta))
    (s1, s2, s3, s4, s5, s6, s7) = np.sin(np.radians(theta))
    d3 = 310.0
    d5 = 310.0
    d7 = 160.0
    T = np.array([[
        ((((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*c6 - ((-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4)*s6)*c7 + (-((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*s5 + (-s1*c3 - s3*c1*c2)*c5)*s7,
        -((((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*c6 - ((-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4)*s6)*s7 + (-((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*s5 + (-s1*c3 - s3*c1*c2)*c5)*c7,
        (((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 + ((-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4)*c6,
        d3*s2*c1 - d5*(-(-s1*s3 + c1*c2*c3)*s4 - s2*c1*c4) - d7*(-(((-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - ((-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4)*c6)
    ], [
        ((((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*c6 - ((s1*c2*c3 + s3*c1)*s4 + s1*s2*c4)*s6)*c7 + (-((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*s5 + (-s1*s3*c2 + c1*c3)*c5)*s7,
        -((((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*c6 - ((s1*c2*c3 + s3*c1)*s4 + s1*s2*c4)*s6)*s7 + (-((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*s5 + (-s1*s3*c2 + c1*c3)*c5)*c7,
        (((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 + ((s1*c2*c3 + s3*c1)*s4 + s1*s2*c4)*c6,
        d3*s1*s2 - d5*(-(s1*c2*c3 + s3*c1)*s4 - s1*s2*c4) - d7*(-(((s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - ((s1*c2*c3 + s3*c1)*s4 + s1*s2*c4)*c6)
    ], [
        (((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*c6 - (-s2*s4*c3 + c2*c4)*s6)*c7 + (-(-s2*c3*c4 - s4*c2)*s5 + s2*s3*c5)*s7,
        -(((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*c6 - (-s2*s4*c3 + c2*c4)*s6)*s7 + (-(-s2*c3*c4 - s4*c2)*s5 + s2*s3*c5)*c7,
        ((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 + (-s2*s4*c3 + c2*c4)*c6,
        d3*c2 - d5*(s2*s4*c3 - c2*c4) - d7*(-((-s2*c3*c4 - s4*c2)*c5 + s2*s3*s5)*s6 - (-s2*s4*c3 + c2*c4)*c6)
    ], [
        0.0, 0.0, 0.0, 1.0
    ]])
    return T
