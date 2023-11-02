from consts import *


def t_1layer(n, d, f):
    omega = 2 * pi * f
    t12, t23 = 2 / (n + 1), 2 * n / (n + 1)

    t_ = t12 * t23 * np.exp(1j * omega * d * n / c_thz)

    return t_


def t_2layer(n, d, f):
    omega = 2 * pi * f
    t12, t23, t34 = 2 / (1 + n[0]), 2 * n[0] / (n[0] + n[1]), 2 * n[1] / (n[1] + 1)

    t_ = t12 * t23 * t34 * np.exp(1j * omega * d[0] * n[0] / c_thz) * np.exp(1j * omega * d[1] * n[1] / c_thz)

    return t_
